// @sylefeb 2022-01-04
/* ---------------------------------------------------------------------

Main file, creates a small graphical GUI (OpenGL+ImGUI) around a 
simulated design. If the design has VGA signals, displays the result 
using a texture. Allows to select between GPU/CPU simulation.

 ----------------------------------------------------------------------- */
/*
BSD 3-Clause License

Copyright (c) 2022, Sylvain Lefebvre (@sylefeb)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// --------------------------------------------------------------

#include <iostream>
#include <ctime>
#include <cmath>
#include <string>
#include <getopt.h>

#include <LibSL/LibSL.h>
#include <LibSL/LibSL_gl4core.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// cuda_runtime.h must be included before including this one.
#include <helper_cuda.h>
#include <helper_functions.h>

#include "read.h"
#include "analyze.h"
#include "profile.h"
#include "optimize.h"
#include "simul_cpu.h"
#include "simul_gpu.h"
#include "simul_cuda.h"

#include "dummy.h"

// --------------------------------------------------------------

#include <imgui.h>
#include <LibSL/UIHelpers/BindImGui.h>

// --------------------------------------------------------------
// Command line parameters

bool    randomOrder            = false;
bool    cuthillMckee           = false;
int     maxFanout              = -1;
string  dumpEdgesFilename;
string  reorderFilename;
string  netlistFilename;

// --------------------------------------------------------------

using namespace std;


static void check(CUresult result, char const *const func,
                  const char *const file, int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line,
            static_cast<unsigned int>(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaDrvErrors(val) check((val), #val, __FILE__, __LINE__)

// --------------------------------------------------------------

#define SCREEN_W   (640) // screen width and height
#define SCREEN_H   (480)

// --------------------------------------------------------------

// Shader to visualize LUT outputs
#include "sh_visu.h"
AutoBindShader::sh_visu g_ShVisu;

// Output ports
map<string, v2i> g_OutPorts; // name, LUT id and rank in g_OutPortsValues
Array<int>       g_OutPortsValues; // output port values

int              g_Cycle = 0;

vector<int>      g_step_starts;
vector<int>      g_step_ends;
vector<t_lut>    g_luts;
vector<int>      g_ones;
vector<int>      g_cpu_fanout;
vector<uchar>    g_cpu_depths;
vector<uchar>    g_cpu_outputs;
vector<int>      g_cpu_computelists;

AutoPtr<GLMesh>  g_Quad;
GLTimer          g_GPU_timer;

bool             g_Use_CUDA = true;

// --------------------------------------------------------------

bool designHasVGA()
{
  return (g_OutPorts.count("out_video_vs") > 0);
}

/* -------------------------------------------------------- */

ImageRGBA_Ptr g_Framebuffer;
Tex2DRGBA_Ptr g_FramebufferTex;
int    g_X  = 0;
int    g_Y  = 0;
int    g_HS = 0;
int    g_VS = 0;
double g_Hz = 0;
double g_UsecPerCycle = 0;
string g_OutPortString;
int    g_OutportCycle = 0;

/* -------------------------------------------------------- */

void simulCUDANextWait()
{
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaEventRecord(start));

  while (1) {
    simulCycle_cuda(g_luts, g_step_starts, g_step_ends);
    if (simulReadback_cuda()) break;
  }

  checkCudaErrors(cudaEventRecord(stop));

  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);

  g_Hz = (double)CYCLE_BUFFER_LEN / ((double)ms / 1000.0);
  g_UsecPerCycle = (double)ms * 1000.0 / (double)CYCLE_BUFFER_LEN;
  g_OutportCycle = 0;
}

/* -------------------------------------------------------- */

void simulCUDANext()
{
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaEventRecord(start));

  simulCycle_cuda(g_luts, g_step_starts, g_step_ends);
  bool datain = simulReadback_cuda();

  checkCudaErrors(cudaEventRecord(stop));

  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);

  g_Hz = (double)1 / ((double)ms / 1000.0);
  g_UsecPerCycle = (double)ms * 1000.0 / (double)1;

  if (datain) {
    g_OutportCycle = 0;
  } else {
    ++g_OutportCycle;
  }

}

/* -------------------------------------------------------- */

void updateFrame(int vs, int hs, int r, int g, int b)
{
  if (vs) {
    if (hs) {
      if (g_X >= 48 && g_Y >= 34) {
        g_Framebuffer->pixel<Clamp>(g_X - 48, g_Y - 34) = v4b(r << 2, g << 2, b << 2, 255);
      }
      ++g_X;
    } else {
      g_X = 0;
      if (g_HS) {
        ++g_Y;
        g_FramebufferTex = Tex2DRGBA_Ptr(new Tex2DRGBA(g_Framebuffer->pixels()));
      }
    }
  } else {
    g_X = g_Y = 0;
  }
  g_VS = vs;
  g_HS = hs;
}

/* -------------------------------------------------------- */

void simulCUDA()
{
  if (designHasVGA()) { // design has VGA output, display it
    simulCUDANextWait(); // simulates a number of cycles and wait
    // read the output of the simulated cycles
    ForIndex(cy, CYCLE_BUFFER_LEN) {
      int offset = cy * (int)g_OutPorts.size();
      int vs = g_OutPortsValues[offset + g_OutPorts["out_video_vs"][0]];
      int hs = g_OutPortsValues[offset + g_OutPorts["out_video_hs"][0]];
      int r  = 0;
      ForIndex(i, 6) {
        r = r | ((g_OutPortsValues[offset + g_OutPorts["out_video_r[" + to_string(i) + "]"][0]]) << i);
      }
      int g = 0;
      ForIndex(i, 6) {
        g = g | ((g_OutPortsValues[offset + g_OutPorts["out_video_g[" + to_string(i) + "]"][0]]) << i);
      }
      int b = 0;
      ForIndex(i, 6) {
        b = b | ((g_OutPortsValues[offset + g_OutPorts["out_video_b[" + to_string(i) + "]"][0]]) << i);
      }
      updateFrame(vs, hs, r, g, b);
    }
  } else { // design has no VGA, show the output ports
    simulCUDANext(); // step one cycle
    // make the output string
    g_OutPortString = "";
    int offset = g_OutportCycle * (int)g_OutPorts.size();
    for (auto op : g_OutPorts) {
      g_OutPortString = (g_OutPortsValues[offset + op.second[0]] ? "1" : "0") + g_OutPortString;
    }
  }
}

/* -------------------------------------------------------- */

uchar simulCPU_output(std::string o)
{
  int pos      = g_OutPorts.at(o)[1];
  int lut      = pos >> 1;
  int q_else_d = pos & 1;
  uchar bit = (g_cpu_outputs[lut] >> q_else_d) & 1;
  return bit;
}

/* -------------------------------------------------------- */

void simulCPU()
{
  if (designHasVGA()) {
    // multiple steps
    int num_measures = 0;
    Elapsed el;
    while (num_measures++ < 100) {
      simulCycle_cpu(g_luts, g_cpu_depths, g_step_starts, g_step_ends, g_cpu_fanout, g_cpu_computelists, g_cpu_outputs);
      simulPosEdge_cpu(g_luts, g_cpu_depths, (int)g_step_starts.size(), g_cpu_fanout, g_cpu_computelists, g_cpu_outputs);
      int vs = simulCPU_output("out_video_vs");
      int hs = simulCPU_output("out_video_hs");
      int r = 0;
      ForIndex(i, 6) {
        r = r | (simulCPU_output("out_video_r[" + to_string(i) + "]") << i);
      }
      int g = 0;
      ForIndex(i, 6) {
        g = g | (simulCPU_output("out_video_g[" + to_string(i) + "]") << i);
      }
      int b = 0;
      ForIndex(i, 6) {
        b = b | (simulCPU_output("out_video_b[" + to_string(i) + "]") << i);
      }
      updateFrame(vs, hs, r, g, b);
    }
    auto ms = el.elapsed();
    g_Hz = (double)100 / ((double)ms / 1000.0);
    g_UsecPerCycle = (double)ms * 1000.0 / (double)100;
  } else {
    // step
    Elapsed el;
    simulCycle_cpu(g_luts, g_cpu_depths, g_step_starts, g_step_ends, g_cpu_fanout, g_cpu_computelists, g_cpu_outputs);
    simulPosEdge_cpu(g_luts, g_cpu_depths, (int)g_step_starts.size(), g_cpu_fanout, g_cpu_computelists, g_cpu_outputs);
    auto ms = el.elapsed();
    if (ms > 0) {
      g_Hz = (double)100 / ((double)ms / 1000.0);
      g_UsecPerCycle = (double)ms * 1000.0 / (double)100;
    } else {
      g_Hz = -1;
      g_UsecPerCycle = -1;
    }
    // make the output string
    g_OutPortString = "";
    for (auto op : g_OutPorts) {
      g_OutPortString = (simulCPU_output(op.first) ? "1" : "0") + g_OutPortString;
    }
  }
}

/* -------------------------------------------------------- */

void mainRender()
{

  // simulate
  if (g_Use_CUDA) {
    simulCUDA();
  } else {
    simulCPU();
  }

  // basic rendering
  LibSL::GPUHelpers::clearScreen(LIBSL_COLOR_BUFFER | LIBSL_DEPTH_BUFFER, 0.2f, 0.2f, 0.2f);

  // render display
  if (designHasVGA()) {
    // -> texture for VGA display
    GLBasicPipeline::getUniqueInstance()->begin();
    GLBasicPipeline::getUniqueInstance()->setProjection(orthoMatrixGL<float>(0, 1, 1, 0, -1, 1));
    GLBasicPipeline::getUniqueInstance()->setModelview(m4x4f::identity());
    GLBasicPipeline::getUniqueInstance()->setColor(v4f(1));
    if (!g_FramebufferTex.isNull()) {
      g_FramebufferTex->bind();
    }
    GLBasicPipeline::getUniqueInstance()->enableTexture();
    GLBasicPipeline::getUniqueInstance()->bindTextureUnit(0);
    g_Quad->render();
    GLBasicPipeline::getUniqueInstance()->end();
  }

  // render LUTs+FF
  if (g_Use_CUDA) {
    GLProtectViewport vp;
    glViewport(0, 0, SCREEN_H*2/3, SCREEN_H*2/3);
    g_ShVisu.begin();
    g_Quad->render();
    g_ShVisu.end();
  }

  // -> GUI
  ImGui::SetNextWindowSize(ImVec2(300, 150), ImGuiCond_Once);
  ImGui::Begin("Status");
  ImGui::Checkbox("Simulate using CUDA", &g_Use_CUDA);
  ImGui::Text("%5.1f KHz %5.1f usec / cycle", g_Hz/1000.0, g_UsecPerCycle);
  ImGui::Text("simulated cycle: %6d", g_Cycle);
  ImGui::Text("simulated LUT4+FF %7zu", g_luts.size());
  ImGui::Text("screen row %3d",g_Y);
  if (!g_OutPortString.empty()) {
    ImGui::Text("outputs: %s", g_OutPortString.c_str());
  }
  ImGui::End();

  ImGui::Render();
}

/* -------------------------------------------------------- */

void printHelp()
{
    fprintf(stderr, 
        "silixel_cuda [options] <netlist> \n"
        "   netlist                     a file in .blif format\n"
        "   -r, --random                randomize node IDs after loading\n"
        "   -c, --cuthill               perform Cuthill-McKee optimization on graph\n"
        "   -d, --dump <filename>       dump graph connections to the specified file as input for Louvain algorithm\n"
        "                               Program will exit after dumping.\n"
        "   -f, --fanout <max fanout>   specify the maximum node fanout for the Louvain or Cuthill-McKee option.\n"
        "                               Default is -1, in which case no high fanout nodes are ignored.\n"
        "   -o, --order <filename>      Apply file with vertex reordering information.\n"
        "   -h, --help                  This help message.\n"
        "\n");
}

void processArgs(int argc, char** argv)
{
    const char* const short_opts = "rc::d:f:o:";
    const option long_opts[] = {
            {"random",  no_argument,       nullptr, 'r'},
            {"cuthill", optional_argument, nullptr, 'c'},
            {"dump",    required_argument, nullptr, 'd'},
            {"fanout",  required_argument, nullptr, 'f'},
            {"order",   required_argument, nullptr, 'o'},
            {"help",    no_argument,       nullptr, 'h'},
            {nullptr,   no_argument,       nullptr,  0 }
    };

    while (true) {
        const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

        if (-1 == opt)
            break;

        switch(opt){
            case 'r':
                fprintf(stderr, "random ordering enabled.\n");
                randomOrder = true;
                break;
            case 'c':
                cuthillMckee = true;
                fprintf(stderr, "Cuthill-McKee enabled.\n");
                break;
            case 'f':
                maxFanout = atoi(optarg);
                fprintf(stderr, "max fanout = %d\n", maxFanout);
                break;
            case 'd':
                dumpEdgesFilename = optarg;
                fprintf(stderr, "Dumping edges to '%s'\n", dumpEdgesFilename.c_str());
                break;
            case 'o':
                reorderFilename = optarg;
                fprintf(stderr, "Reorder file: '%s'\n", reorderFilename.c_str());
                break;
            case 'h':
                printHelp();
                exit(0);
            default:
                printHelp();
                exit(-1);
        }
    }

    if ((optind+1) == argc){
        netlistFilename = argv[optind];
        fprintf(stderr, "Use netlist file '%s'.\n", netlistFilename.c_str());
    }
    else{
        fprintf(stderr, "No netlist file specified.\n");
        exit(-1);
    }

}


CUdevice g_cuDevice;

void initCuda(int argc, char **argv)
{

  g_cuDevice = findCudaDevice(argc, (const char **)argv);
  if (g_cuDevice == -1){
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char **argv)
{
  processArgs(argc, argv);
  initCuda(argc,argv);

  try {

    /// init simple UI (glut clone for both GL and D3D)
    cerr << "Init SimpleUI   ";
    SimpleUI::init(SCREEN_W, SCREEN_H);
    SimpleUI::onRender = mainRender;
    cerr << "[OK]" << endl;

    /// bind imgui
    SimpleUI::bindImGui();
    SimpleUI::initImGui();
    SimpleUI::onReshape(SCREEN_W, SCREEN_H);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    /// help
    printf("[ESC]    - quit\n");

    /// display stuff
    g_Framebuffer = ImageRGBA_Ptr(new ImageRGBA(640,480));
    g_Quad = AutoPtr<GLMesh>(new GLMesh());
    g_Quad->begin(GPUMESH_TRIANGLESTRIP);
    g_Quad->texcoord0_2(0, 0); g_Quad->vertex_2(0, 0);
    g_Quad->texcoord0_2(1, 0); g_Quad->vertex_2(1, 0);
    g_Quad->texcoord0_2(0, 1); g_Quad->vertex_2(0, 1);
    g_Quad->texcoord0_2(1, 1); g_Quad->vertex_2(1, 1);
    g_Quad->end();

    /// GPU shaders init
    g_ShVisu.init();

    /// GPU timer
    g_GPU_timer.init();

    /// load up design
    vector<pair<string,int> > outbits;
    readDesign(netlistFilename, g_luts, outbits, g_ones);

#if 0
    analyze(g_luts, outbits, g_ones, g_step_starts, g_step_ends, g_cpu_depths);
    printf("Level differences before optimization:\n");
    profileInputDifferences(g_luts, g_step_starts, g_step_ends, 9);
#endif

    if (randomOrder){
        optimizeRandomOrder(g_luts, outbits, g_ones);
    }

    if (!dumpEdgesFilename.empty()){
        profileDumpLouvainGraph(g_luts, dumpEdgesFilename, maxFanout);
        exit(0);
    }

    if (!reorderFilename.empty()){
        unordered_map<int,int> id2group;
        int num_groups = optimizeReadReorderFile(reorderFilename.c_str(), id2group);
        optimizeSortByGroup(g_luts, outbits, g_ones, id2group);
    }

    if (cuthillMckee){
        optimizeCuthillMckee(g_luts, outbits, g_ones, maxFanout);
    }

    analyze(g_luts, outbits, g_ones, g_step_starts, g_step_ends, g_cpu_depths);

#if 0
    printf("Level differences after optimization:\n");
    profileInputDifferences(g_luts, g_step_starts, g_step_ends, 9);
#endif

    buildFanout(g_luts, g_cpu_fanout);

#if 0
    profileHistogram(g_luts);
#endif

    int rank = 0;
    for (auto op : outbits) {
      g_OutPorts.insert(make_pair(op.first,v2i(rank++, op.second)));
    }
    g_OutPortsValues.allocate(rank * CYCLE_BUFFER_LEN);

    /// GPU buffers init
    simulInit_cuda(g_luts, g_ones, g_step_starts, g_step_ends);

    // init CPU simulation
    simulInit_cpu(g_luts, g_step_starts, g_step_ends, g_ones, g_cpu_computelists, g_cpu_outputs);

    /// Quick benchmarking at startup
#if 1
    // -> time GPU
    simulBegin_cuda(g_luts,g_step_starts,g_step_ends,g_ones);   
    {
      ForIndex(trials, 3) {
        int n_cycles = 10000;

        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));

        checkCudaErrors(cudaEventRecord(start));

        ForIndex(cycle, n_cycles) {
          simulCycle_cuda(g_luts, g_step_starts, g_step_ends);
          simulReadback_cuda();
        }
        checkCudaErrors(cudaEventRecord(stop));
        float ms = 0;

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        cudaDeviceSynchronize();
        simulPrintOutput_cuda(outbits, 10);

        printf("[GPU] %f msec, ~ %f Hz, cycle time: %f usec\n",
          ms,
          (double)n_cycles / ((double)ms / 1000.0),
          (double)ms * 1000.0 / (double)n_cycles);
      }
    }
    simulEnd_cuda();
    // -> time CPU
    {
      ForIndex(trials, 3) {
        Elapsed el;
        int n_cycles = 1000;
        ForIndex(cy, n_cycles) {
          simulCycle_cpu(g_luts, g_cpu_depths, g_step_starts, g_step_ends, g_cpu_fanout, g_cpu_computelists, g_cpu_outputs);
          simulPosEdge_cpu(g_luts, g_cpu_depths, (int)g_step_starts.size(), g_cpu_fanout, g_cpu_computelists, g_cpu_outputs);
        }
        auto ms = el.elapsed();
        printf("[CPU] %d msec, ~ %f Hz, cycle time: %f usec\n",
          (int)ms,
          (double)n_cycles / ((double)ms / 1000.0),
          (double)ms * 1000.0 / (double)n_cycles);
      }
    }
#endif

    /// shader parameters
    g_ShVisu.begin();
    int n_simul = (int)g_luts.size() - g_step_ends[0];
    int sqsz = (int)sqrt((double)(n_simul)) + 1;
    fprintf(stderr, "simulating %d LUTs+FF (%dx%d pixels)", n_simul, sqsz, sqsz);
    g_ShVisu.sqsz      .set(sqsz);
    g_ShVisu.num       .set((int)(g_luts.size()));
    g_ShVisu.depth0_end.set((int)(g_step_ends[0]));
    g_ShVisu.end();

    /// main loop
    simulBegin_cuda(g_luts, g_step_starts, g_step_ends, g_ones);
    SimpleUI::loop();
    simulEnd_cuda();

    /// clean exit
    simulTerminate_cuda();
    g_ShVisu.terminate();
    g_GPU_timer.terminate();
    g_FramebufferTex = Tex2DRGBA_Ptr();
    g_Quad = AutoPtr<GLMesh>();

    /// shutdown SimpleUI
    SimpleUI::shutdown();

  } catch (Fatal& e) {
    cerr << e.message() << endl;
    return (-1);
  }

  return (0);
}

/* -------------------------------------------------------- */
