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

#include <unordered_map>
#include <vector>

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

typedef uint16_t    t_lut_cfg;
typedef uint32_t    t_lut_addr;
typedef uint8_t     t_lut_val;

// 1 bit per LUT location, so 16-bits for a 4-input LUT. 1 Cfg value for each LUT in the design.
t_lut_cfg       * g_cuLUTs_Cfg;

// 1 address (or better: index) to a LUT output that serves as the input for this LUT.
// 4 addresses per LUT. 1 set of addresses for each LUT in the design.
t_lut_addr      * g_cuLUTs_Addrs;

// Unregistered (D, stored in bit 0) and registered (Q, stored in bit 1) output value of a LUT.
// (D,Q) set for each LUT in the design.
t_lut_val       * g_cuLUTs_Outputs;

// List with all the D or Q LUT values that are 1 at the start of a simulation.
// The value store is formated as (LUT addr << 1) | (Q ? 1 : 0)
// So the LSB indicates whether the indicates value is for a Q or a D.
t_lut_addr      * g_cuOutInits;

// Location of the output values.
// Uses the same format a g_cuOutInits: value is stored at (LUT addr << 1) | (Q ? 1: 0)
// There are as many items in this array as there are output ports.
t_lut_addr      * g_cuOutPortsLocs;
// Value of the output ports. A shader loops through all the OutPortsLocs, fetches
// the data from g_cuLUTs_Outputs, and stores the value here.
// This array contais as many values as there are output ports times CYCLE_BUFFER_LEN.
t_lut_val       * g_cuOutPortsVals;


#define NR_LUT_INPUTS   4

#define DEBUG 0

#if DEBUG == 1
const int g_blockSize   = 1;
const int g_numBlocks   = 1;
#else
const int g_blockSize   = 512;
const int g_numBlocks   = 1024;
#endif

int rounded_n(int n)
{
    // Round up n to be a multiple of G.
    n += ( (n & (g_blockSize - 1)) ? (g_blockSize - (n & (g_blockSize - 1))) : 0 );
    return n;
}

extern CUdevice g_cuDevice;

// Number of bytes!
//const int   shared_mem_bytes_per_block = 8192;
const int   shared_mem_bytes_per_block = 128;     

void mapGlobalToShared(
    const vector<t_lut>& luts,
    int     start,
    int     end
    )
{


    // We only store the value of the input LUTs in shared memory.
    // Loop through all the LUTs of a level, and gradually fill up the following structure:
    // * Address with data that is being fetched (t_lut_val)
    // * For each such address, the LUTs that need it.
    // * Remember the original location in shared memory
    // When the number of addresses fills up the size of shared memory, then 
    // finish the current block, and start the next one.
    // Finishing a block does the following:
    // * Sort all addresses in increasing order.
    // * Creating a data structure that links the input LUT ids to the shared memory access.

    typedef struct s_input_data {
        t_lut_addr                 addr;                   // Global memory address that must be fetched.
        vector<pair<int,bool>>     referencing_luts;       // Luts that are using this data.
    } t_input_data;

    unordered_map<t_lut_addr, t_input_data>  input_datas;

    // All data that's needed for one block with shared memory.
    typedef struct s_block_data {
        int                     start_lut;
        int                     end_lut;
        // Addresses of global memory data that must be fetched 
        vector<t_lut_addr>      global_mem_fetch_list;
        // shared memory address and as_ff
        vector<pair<int, bool>> lut_attributes;
        
    } t_block_data;

    vector<t_block_data> all_blocks;

    int cur_start_lut = start;
    for(int lidx=start; lidx<=end; ++lidx){
        const t_lut&  cur_lut = luts[lidx];

        for(int in_idx=0; in_idx<4; ++in_idx){
            int input_addr   = cur_lut.inputs[in_idx];

            if (input_addr == -1)
                continue;
            bool as_ff = input_addr & 1;
            input_addr >>= 1;

            if (input_datas.find(input_addr) == input_datas.end()){
                t_input_data& new_input_data = input_datas[input_addr];
                new_input_data.addr = input_addr;
                new_input_data.referencing_luts.push_back(make_pair(lidx, as_ff));
            }
            else{
                input_datas[input_addr].referencing_luts.push_back(make_pair(lidx, as_ff));
            }

        }

        // -3 because there are 4 inputs per LUT, and we don't want the next one to cause an overflow.
        if (input_datas.size() * sizeof(t_lut_val) >= shared_mem_bytes_per_block-(3 *sizeof(t_lut_val)) || lidx==end)
        {
            // Sort all data accesses 
            vector<pair<t_lut_addr, t_input_data>> input_datas_sorted(input_datas.begin(), input_datas.end());
            sort(
                input_datas_sorted.begin(), 
                input_datas_sorted.end(), 
                [](const pair<t_lut_addr, t_input_data>& a, const pair<t_lut_addr, t_input_data>& b) -> bool 
                {
                    return a.first < b.first;
                } );

            all_blocks.emplace_back();
            t_block_data& cur_block = all_blocks.back();

            cur_block.start_lut = cur_start_lut;
            cur_block.end_lut   = lidx;
            cur_start_lut       = lidx+1;

            printf("-------\nBlock %zu - %zu shared mem - %d LUTs : %d -> %d\n", all_blocks.size()-1, input_datas_sorted.size(), cur_block.end_lut-cur_block.start_lut+1, cur_block.start_lut, cur_block.end_lut);

            cur_block.lut_attributes.resize(cur_block.end_lut-cur_block.start_lut+1);
            cur_block.global_mem_fetch_list.resize(input_datas_sorted.size());

            int id_idx=0;
            for(auto id: input_datas_sorted){
                cur_block.global_mem_fetch_list[id_idx] = id.second.addr;

                for(auto ref_lut: id.second.referencing_luts){
                    cur_block.lut_attributes[ref_lut.first-cur_block.start_lut] = make_pair(id_idx, ref_lut.second);
                }

                ++id_idx;
            }

            for(auto id: input_datas_sorted){
                printf("data %d:\n", id.second.addr);
                for(auto rl: id.second.referencing_luts){
                    printf("    <- LUT %d, as_ff(%d)\n", rl.first, rl.second);
                }
            }


            input_datas.clear();
        }
    }

    // We have all the block for a particular level. Now we need to create the cuda memory
    // structures.


}

void simulInit_cuda(
    const vector<t_lut>&    luts,
    const vector<int>&      ones,
    const vector<int>&      step_starts,
    const vector<int>&      step_ends
    )
{
    int n_luts = rounded_n((int)luts.size());

    mapGlobalToShared(luts, step_starts[2], step_ends[2]);

    for(int i=1; i<step_starts.size(); ++i){
        printf("============================ step %d (%d -> %d)\n", i, step_starts[i], step_ends[i]);
        mapGlobalToShared(luts, step_starts[i], step_ends[i]);
    }
    
    int cfg_size            = n_luts                 * sizeof(t_lut_cfg);
    int addrs_size          = n_luts * NR_LUT_INPUTS * sizeof(t_lut_addr);
    int outputs_size        = n_luts                 * sizeof(t_lut_val);
    int out_ports_locs_size = g_OutPorts.size()      * sizeof(t_lut_addr);
    int out_ports_vals_size = g_OutPorts.size() * CYCLE_BUFFER_LEN * sizeof(t_lut_val);
    int out_inits_size      = ones.size()            * sizeof(t_lut_addr);


    checkCudaErrors(cudaMallocManaged(&g_cuLUTs_Cfg,      cfg_size));
    checkCudaErrors(cudaMallocManaged(&g_cuLUTs_Addrs,    addrs_size));
    checkCudaErrors(cudaMallocManaged(&g_cuLUTs_Outputs,  outputs_size));

    checkCudaErrors(cudaMallocManaged(&g_cuOutPortsLocs,  out_ports_locs_size));
    checkCudaErrors(cudaMallocManaged(&g_cuOutPortsVals,  out_ports_vals_size));

    checkCudaErrors(cudaMallocManaged(&g_cuOutInits,      out_inits_size));

    for(int i=0; i<n_luts; ++i){
        // initialize the static LUT table
        // -> configs
        g_cuLUTs_Cfg[i] = (int)luts[i].cfg; 
  
        // -> addrs
        for(int j=0;j<NR_LUT_INPUTS;++j){
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

    // Move everything to GPU memory in one go instead of relying on page
    // faults. This makes the kernels run at max speed even when they are called
    // the first time. That makes it easier for profiling.
    checkCudaErrors(cudaMemPrefetchAsync(g_cuLUTs_Cfg,      cfg_size, g_cuDevice));
    checkCudaErrors(cudaMemPrefetchAsync(g_cuLUTs_Addrs,    addrs_size, g_cuDevice));
    checkCudaErrors(cudaMemPrefetchAsync(g_cuLUTs_Outputs,  outputs_size, g_cuDevice));
    checkCudaErrors(cudaMemPrefetchAsync(g_cuOutPortsLocs,  out_ports_locs_size, g_cuDevice));
    checkCudaErrors(cudaMemPrefetchAsync(g_cuOutPortsVals,  out_ports_vals_size, g_cuDevice));

    if (out_inits_size > 0)
        checkCudaErrors(cudaMemPrefetchAsync(g_cuOutInits,      out_inits_size, g_cuDevice));
}


/* -------------------------------------------------------- */

__global__ void simInit_cuda(
    const t_lut_addr    * ones,
    t_lut_val           * outputs,
    const int           N
)
{
    // Use Grid-Stride loops: https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for(int i=blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x){
        t_lut_addr addr = ones[i];
        t_lut_addr real_addr = addr >> 1;

        t_lut_val   old_val = outputs[real_addr];
        t_lut_val   new_val = old_val | (1 << (addr & 1));

        if (old_val != new_val){
            outputs[real_addr] = new_val;
        }
    }
}

__device__ t_lut_val get_output(t_lut_val * outputs, t_lut_addr a)
{
    return (outputs[a >> 1] >> (a & 1)) & 1;
}

__global__ void simSimul_cuda(
    const t_lut_cfg     * cfg,
    const t_lut_addr    * addrs,
    t_lut_val           * outputs, 
    const int           start_lut,
    const int           N
)
{
#if 0
    printf("blockIdx.x: %d, threadIdx.x: %d\n", blockIdx.x, threadIdx.x);
#endif

    for(int i=blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x){
        int lut_id = start_lut + i;

        t_lut_cfg C     = cfg[lut_id];
        t_lut_val i0    = get_output(outputs, addrs[lut_id*4]);
        t_lut_val i1    = get_output(outputs, addrs[lut_id*4 + 1]);
        t_lut_val i2    = get_output(outputs, addrs[lut_id*4 + 2]);
        t_lut_val i3    = get_output(outputs, addrs[lut_id*4 + 3]);
        int sh = i3 | (i2 << 1) | (i1 << 2) | (i0 << 3);

        t_lut_val outv = outputs[lut_id];
        t_lut_val old_d = outv & 1u;
        t_lut_val new_d = (C >> sh) & 1u;

#if DEBUG==1
        printf("start_lut: %d, i: %d, id: %d, N: %d, new_d: %d <- i0(%d), i1(%d), i2(%d), i3(%d), a0(%d), a1(%d), a2(%d), a3(%d)\n", 
                start_lut, i, lut_id, N, new_d, i0, i1, i2, i3, addrs[lut_id*4], addrs[lut_id*4+1], addrs[lut_id*4+2], addrs[lut_id*4+3]);
#endif

        if (old_d != new_d) {
            if (new_d == 1){
                outputs[lut_id] = outv | 1;
            }
            else{
                outputs[lut_id] = outv & 0xfffffffe;
            }
        }
    }
}

#if 0
__global__ void simSimulShared_cuda(
    const t_block_info  * block_info,
    const t_lut_info    * lut_info,
    const t_lut_addr    * global_mem_fetch_list,
    t_lut_val           * lut_vals,
    int                 N
)
{
    __shared__ t_lut_val    shared_mem_lut_vals[shared_mem_bytes_per_block/sizeof(t_lut_val)];

    t_block_info &   bi = block_info[blockIdx.x];
    int nr_shared_mem_values = bi.nr_shared_mem_values;
    int nr_luts              = bi.nr_luts;

    // Fetch all input LUT values from global memory and stored in shared memory.
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < nr_shared_mem_values; i += blockDim.x * gridDim.x){
        t_lut_val   lut_val = lut_vals[global_mem_fetch_list[i]];
        shared_mem_lut_vals[i] = lut_val;
    }

    __syncthreads();

    // Calculate all values...
    for(int lut_id = blockDim.x * blockIdx.x + threadIdx.x; lut_id < nr_shared_mem_values; lut_id += blockDim.x * gridDim.x){
        t_lut_info & li = lut_info[lut_id];
        t_lut_cfg   = lut_info.cfg;

        t_lut_val i0    = (shared_mem_lut_vals[li.shared_mem_addrs[0]] >> li.is_ff[0]) & 1;
        t_lut_val i1    = (shared_mem_lut_vals[li.shared_mem_addrs[1]] >> li.is_ff[1]) & 1;
        t_lut_val i2    = (shared_mem_lut_vals[li.shared_mem_addrs[2]] >> li.is_ff[2]) & 1;
        t_lut_val i3    = (shared_mem_lut_vals[li.shared_mem_addrs[3]] >> li.is_ff[3]) & 1;
        int sh = i3 | (i2 << 1) | (i1 << 2) | (i0 << 3);

        t_lut_val outv = lut_vals[lut_id];
        t_lut_val old_d = outv & 1u;
        t_lut_val new_d = (C >> sh) & 1u;

#if DEBUG==1
        printf("start_lut: %d, lut_id: %d, N: %d, new_d: %d <- i0(%d), i1(%d), i2(%d), i3(%d), a0(%d), a1(%d), a2(%d), a3(%d)\n", 
                start_lut, lut_id, lut_id, N, new_d, i0, i1, i2, i3, addrs[lut_id*4], addrs[lut_id*4+1], addrs[lut_id*4+2], addrs[lut_id*4+3]);
#endif

        if (old_d != new_d) {
            if (new_d == 1){
                outputs[lut_id] = outv | 1;
            }
            else{
                outputs[lut_id] = outv & 0xfffffffe;
            }
        }
    }

}
#endif


__global__ void simPosEdge_cuda(
    t_lut_val           * outputs, 
    const int           N
)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<N){
        int lut_id = i;

        t_lut_val outv = outputs[lut_id];

        if ((outv & 1) != ((outv>>1)&1)){
            if ((outv & 1) == 1) {
                outputs[lut_id] = 3;
            } else {
                outputs[lut_id] = 0;
            }
        }
    }
}

__global__ void simOutPorts_cuda(
    const t_lut_val     * outputs, 
    const t_lut_addr    * portlocs, 
    t_lut_val           * portvals,
    int offset,
    int N
)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<N){
        int id = i;
        t_lut_addr  o = portlocs[id];
        t_lut_val ov = (outputs[o>>1] >> (o & 1)) & 1;
        portvals[offset + id] = ov;

#if DEBUG==1
        printf("output port %d (a:%d) = %d, offset = %d\n", i, o, ov, offset);
#endif

    }
}


/* -------------------------------------------------------- */

void simulBegin_cuda(
  const vector<t_lut>& luts,
  const vector<int>&   step_starts,
  const vector<int>&   step_ends,
  const vector<int>&   ones)
{
    int n;
    int blockSize;
    int numBlocks;

    // init cells
    n = ones.size();

    //blockSize = G;
    //numBlocks = (n+blockSize-1)/blockSize;

    blockSize = g_blockSize;
    numBlocks = min(g_numBlocks, (n+blockSize-1)/blockSize);

    simInit_cuda<<<numBlocks,blockSize>>>(g_cuOutInits, g_cuLUTs_Outputs, n);

    // resolve constant cells
    for(int c=0;c<2;++c){
        n = step_ends[0]-step_starts[0]+1;
        blockSize = g_blockSize;
        numBlocks = (n+blockSize-1)/blockSize;

        simSimul_cuda<<<numBlocks,blockSize>>>(g_cuLUTs_Cfg, g_cuLUTs_Addrs, g_cuLUTs_Outputs, 0, n);

        n = luts.size();
        blockSize = g_blockSize;
        numBlocks = (n+blockSize-1)/blockSize;

        simPosEdge_cuda<<<numBlocks,blockSize>>>(g_cuLUTs_Outputs, n);
    }

    // init cells
    // Why a second time? Some of these registers may have been cleared after const resolve
    n = ones.size();
    blockSize = g_blockSize;
    numBlocks = (n+blockSize-1)/blockSize;

    simInit_cuda<<<numBlocks,blockSize>>>(g_cuOutInits, g_cuLUTs_Outputs, n);
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
    int n;
    int blockSize;
    int numBlocks;

    for(int depth=1; depth < step_starts.size(); ++depth){
        n = step_ends[depth]-step_starts[depth]+1;

        //blockSize = g_blockSize;
        //numBlocks = (n+blockSize-1)/blockSize;

        blockSize = g_blockSize;
        numBlocks = min(g_numBlocks, (n+blockSize-1)/blockSize);

        simSimul_cuda<<<numBlocks,blockSize>>>(g_cuLUTs_Cfg, g_cuLUTs_Addrs, g_cuLUTs_Outputs, step_starts[depth], n);
    }

    n = luts.size();
    blockSize = g_blockSize;
    numBlocks = (n+blockSize-1)/blockSize;

    simPosEdge_cuda<<<numBlocks,blockSize>>>(g_cuLUTs_Outputs, n);

    ++g_Cycle;
}

/* -------------------------------------------------------- */

void simulEnd_cuda()
{
}

/* -------------------------------------------------------- */

static uint g_RBCycle = 0;

bool simulReadback_cuda()
{
    int n;
    int blockSize;
    int numBlocks;

    n = g_OutPorts.size();
    blockSize = g_blockSize;
    numBlocks = (n+blockSize-1)/blockSize;

    simOutPorts_cuda<<<numBlocks,blockSize>>>(g_cuLUTs_Outputs, g_cuOutPortsLocs, g_cuOutPortsVals, n * g_RBCycle, n);

    if (g_RBCycle == CYCLE_BUFFER_LEN-1) {
        cudaDeviceSynchronize();
        int out_ports_vals_size = g_OutPorts.size() * CYCLE_BUFFER_LEN * sizeof(t_lut_val);
        checkCudaErrors(cudaMemPrefetchAsync(g_cuOutPortsVals, out_ports_vals_size, cudaCpuDeviceId));
        for(int i=0; i<g_OutPortsValues.size();++i){
            g_OutPortsValues[i] = g_cuOutPortsVals[i];
        }
        g_RBCycle = 0;
    }
    else
        ++g_RBCycle;

    return g_RBCycle == 0;
}

/* -------------------------------------------------------- */

void simulPrintOutput_cuda(const vector<pair<string, int> >& outbits, int nr)
{
  // display result (assumes readback done)
  for(int i=0; i<nr; ++i){
    int val = 0;
    string str;

    for (int b = 0; b < outbits.size(); b++) {
      int vb = g_OutPortsValues[b + i *outbits.size() ];
      str = (vb ? "1" : "0") + str;
      val += vb << b;

#if 0
      fprintf(stderr, "output %d: %s = %d\n", b, outbits[b].first.c_str(), vb);
#endif
    }
    fprintf(stderr, "b%s (d%d h%x)\n", str.c_str(), val, val); 
  }
}

// --------------------------------------------------------------

void simulTerminate_cuda()
{
    cudaFree(g_cuLUTs_Cfg);
    cudaFree(g_cuLUTs_Addrs);
    cudaFree(g_cuLUTs_Outputs);
    cudaFree(g_cuOutPortsLocs);
    cudaFree(g_cuOutPortsVals);
    cudaFree(g_cuOutInits);
}

// --------------------------------------------------------------
