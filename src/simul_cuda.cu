// @sylefeb 2022-02-11
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

#include <LibSL/LibSL.h>
#include <LibSL/LibSL_gl4core.h>

#include "read.h"
#include "analyze.h"
#include "simul_cuda.h"

#include "dummy.h"

using namespace std;

// --------------------------------------------------------------

// Ah, some good old globals and externs
// defined in silixel.cc
extern map<string, v2i> g_OutPorts;
extern Array<int>       g_OutPortsValues;
extern int              g_Cycle;

// --------------------------------------------------------------

using namespace std;

// --------------------------------------------------------------

typedef uint16_t    lut_cfg_t;
typedef uint32_t    lut_addr_t;
typedef uint8_t     lut_val_t;

// 1 bit per LUT location, so 16-bits for a 4-input LUT. 1 Cfg value for each LUT in the design.
lut_cfg_t       * g_cuLUTs_Cfg;

// 1 address (or better: index) to a LUT output that serves as the input for this LUT.
// 4 addresses per LUT. 1 set of addresses for each LUT in the design.
lut_addr_t      * g_cuLUTs_Addrs;

// Unregistered (D, stored in bit 0) and registered (Q, stored in bit 1) output value of a LUT.
// (D,Q) set for each LUT in the design.
lut_val_t       * g_cuLUTs_Outputs;

// List with all the D or Q LUT values that are 1 at the start of a simulation.
// The value store is formated as (LUT addr << 1) | (Q ? 1 : 0)
// So the LSB indicates whether the indicates value is for a Q or a D.
lut_addr_t      * g_cuOutInits;

// Location of the output values.
// Uses the same format a g_cuOutInits: value is stored at (LUT addr << 1) | (Q ? 1: 0)
// There are as many items in this array as there are output ports.
lut_addr_t      * g_cuOutPortsLocs;
// Value of the output ports. A shader loops through all the OutPortsLocs, fetches
// the data from g_cuLUTs_Outputs, and stores the value here.
// This array contais as many values as there are output ports times CYCLE_BUFFER_LEN.
lut_val_t       * g_cuOutPortsVals;


#define NR_LUT_INPUTS   4
const int G = 128;

void simulInit_cuda(const vector<t_lut>& luts,const vector<int>& ones)
{
    int n_luts = (int)luts.size();

    // Round up n_luts to be a multiple of 128.
    n_luts += ( (n_luts & (G - 1)) ? (G - (n_luts & (G - 1))) : 0 );

    cudaMallocManaged(&g_cuLUTs_Cfg,      n_luts                 * sizeof(lut_cfg_t));
    cudaMallocManaged(&g_cuLUTs_Addrs,    n_luts * NR_LUT_INPUTS * sizeof(lut_addr_t));
    cudaMallocManaged(&g_cuLUTs_Outputs,  n_luts                 * sizeof(lut_val_t));

    cudaMallocManaged(&g_cuOutPortsLocs,  g_OutPorts.size()      * sizeof(lut_addr_t));
    cudaMallocManaged(&g_cuOutPortsVals,  g_OutPorts.size() * CYCLE_BUFFER_LEN * sizeof(lut_val_t));

    cudaMallocManaged(&g_cuOutInits,      ones.size()            * sizeof(lut_addr_t));

    for(int i=0; i<n_luts; ++i){
        // initialize the static LUT table
        // -> configs
        g_cuLUTs_Cfg[i] = (int)luts[i].cfg; 
  
        // -> addrs
        for(int j=0;i<NR_LUT_INPUTS;++j){
            g_cuLUTs_Addrs[i*NR_LUT_INPUTS + j] = max(0,(int)luts[i].inputs[j]);
        }

        // we initialize all outputs to zero
        g_cuLUTs_Outputs[i] = 0;
    }

    for(auto op : g_OutPorts) {
        g_cuOutPortsLocs[op.second[0]] = op.second[1];
    }

    for(int i=0; i<ones.size();++i){
        g_cuOutInits[i] = ones[i];
    }
}

/* -------------------------------------------------------- */

void simulBegin_cuda(
  const vector<t_lut>& luts,
  const vector<int>&   step_starts,
  const vector<int>&   step_ends,
  const vector<int>&   ones)
{
#if 0
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, g_LUTs_Cfg.glId());
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, g_LUTs_Addrs.glId());
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, g_LUTs_Outputs.glId());
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, g_GPU_OutPortsLocs.glId());
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, g_GPU_OutPortsVals.glId());
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, g_GPU_OutInits.glId());
  // init cells
  g_ShInit.begin();
  g_ShInit.run(v3i((int)ones.size(),1,1));
  g_ShInit.end();
  // resolve constant cells
  ForIndex (c,2) {
    int n = step_ends[0] - step_starts[0] + 1;
    g_ShSimul.begin();
    g_ShSimul.start_lut.set((uint)0);
    g_ShSimul.num.set((uint)n);
    g_ShSimul.run(v3i((n / G) + ((n & (G - 1)) ? 1 : 0), 1, 1));
    g_ShSimul.end();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    g_ShPosEdge.begin();
    n = (int)luts.size();
    g_ShPosEdge.num.set((uint)n);
    g_ShPosEdge.run(v3i((n / G) + ((n & (G - 1)) ? 1 : 0), 1, 1));
    g_ShPosEdge.end();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  }
  // init cells
  // Why a second time? Some of these registers may have been cleared after const resolve
  g_ShInit.begin();
  g_ShInit.run(v3i((int)ones.size(), 1, 1));
  g_ShInit.end();
#endif
}

/* -------------------------------------------------------- */

/*
Simulate one cycle on the GPU
*/
void simulCycle_cuda(
  const vector<t_lut>& luts,
  const vector<int>&   step_starts,
  const vector<int>&   step_ends)
{
#if 0

  g_ShSimul.begin();
  // iterate on depth levels (skipping const depth 0)
  ForRange(depth, 1, (int)step_starts.size()-1) {
    // only update LUTs at this particular level
    int n = step_ends[depth] - step_starts[depth] + 1;
    g_ShSimul.start_lut.set((uint)step_starts[depth]);
    g_ShSimul.num.set((uint)n);
    g_ShSimul.run(v3i((n / G) + ((n & (G - 1)) ? 1 : 0), 1, 1));
    // sync required between iterations
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  }
  g_ShSimul.end();
  // simulate positive clock edge
  g_ShPosEdge.begin();
  int n = (int)luts.size();
  g_ShPosEdge.num.set((uint)n);
  g_ShPosEdge.run(v3i((n / G) + ((n & (G - 1)) ? 1 : 0), 1, 1));
  g_ShPosEdge.end();
  // sync required to ensure further reads see the update
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

  ++g_Cycle;
#endif

}

/* -------------------------------------------------------- */

void simulEnd_cuda()
{
#if 0
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, 0);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, 0);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, 0);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, 0);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
#endif
}

/* -------------------------------------------------------- */

bool simulReadback_cuda()
{
#if 0
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

  // gather outport values
  g_ShOutPorts.begin();
  g_ShOutPorts.offset.set((uint)g_OutPorts.size() * g_RBCycle);
  g_ShOutPorts.run(v3i((int)g_OutPorts.size(), 1, 1)); // TODO: local size >= 32
  g_ShOutPorts.end();

  ++g_RBCycle;

  if (g_RBCycle == CYCLE_BUFFER_LEN) {
    // readback buffer
    glBindBufferARB(GL_SHADER_STORAGE_BUFFER, g_GPU_OutPortsVals.glId());
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, g_OutPortsValues.sizeOfData(), g_OutPortsValues.raw());
    glBindBufferARB(GL_SHADER_STORAGE_BUFFER, 0);
    g_RBCycle = 0;
  }

  return g_RBCycle == 0;
#endif

    return false;
}

/* -------------------------------------------------------- */

void simulPrintOutput_cuda(const vector<pair<string, int> >& outbits)
{
#if 0
  // display result (assumes readback done)
  int val = 0;
  string str;
  for (int b = 0; b < outbits.size(); b++) {
    int vb = g_OutPortsValues[b];
    str = (vb ? "1" : "0") + str;
    val += vb << b;
  }
  fprintf(stderr, "b%s (d%d h%x)\n", str.c_str(), val, val);
#endif
}

// --------------------------------------------------------------

void simulTerminate_cuda()
{
#if 0
  g_LUTs_Addrs.terminate();
  g_LUTs_Cfg.terminate();
  g_LUTs_Outputs.terminate();
  g_GPU_OutPortsLocs.terminate();
  g_GPU_OutPortsVals.terminate();
  g_GPU_OutInits.terminate();
#endif
}

// --------------------------------------------------------------
