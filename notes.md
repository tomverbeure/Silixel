
# Louvain

* Generate graph.txt with all edges
* Convert to .bin file: 

    ../louvain-generic/convert -i louvain_input.vga_demo.txt -o louvain_input.vga_demo.bin

* Run Louvain and output the full graph (by using `-l -1`). This can take a long time a netlist with
  a signals that have a huge fanout.

    ../louvain-generic/louvain louvain_input.vga_demo.bin -v -l -1 > graph.tree

* Run hierarchy to see how each original node is grouped into the same community for a given level:

    ../louvain-generic/hierarchy graph.tree -l 2

* Choose one of these hierarchies and create node to community file:

    ../louvain-generic/hierarchy graph.tree -l 3 > louvain.vga_demo.l3.group

# Louvain notes

* The algorithm gets much slower when there are nodes with a huge fanout. Need to do experiments where
  nodes with large fanout don't have a fanout (only an edge to itself?)

* execution time: 
    * blaze netlist without high fanout nets removed:     1569s
    * blaze netlist with high fanout nets removed (>500): 406s
    * blaze netlist with high fanout nets removed (>255): 45s

# Blaze with double memory:

* Louvain >255 fanout removed: 75s

* Standard: 309us
* Random:   350us
* Louvain, prune 256, l2: 312us
* Louvain, prune 256, l3: 311us
* Louvain, prune 256, l4: 310us

# SM notes

* Pascal GP10x:
    * Per SM:
        * 128 warps in parallel (4 x 32)
        * 64KB register file
        * max 255 registers per thread
        * max 32 thread blocks per SM
        * 48KB L1 cache
        * 96KB shared memory
        * Max 48KB shared memory per thread block.
        * Recommened to use at most 32KB per thread block to allow for 3 blocks per SM.
        * Unified L1/Texture cache
    * BU default, global loads are only cached in L2, not L1, except when using the LDG read-only
      data cache mechanism. However, global loads can be stored in L1/Tex cache with -Xptxas -dlcm=ca
      nvcc compiler flags.
    * Data access granularity is 32 bytes.
    * Native shared memory atomic operations for 32-bit integer.
    * [Pascal Tuning Guide](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html#pascal-tuning)

* GTX 1060
    * 10 SMs

* Ampere GA10x:
    * Per SM:
        * 64 warps in parallel (2 x32)
        * 64KB register file
        * max 255 registers per thread
        * max 16 thread blocks per SM
        * 100KB shared memory (max 99KB per thread block)
    * asynchronous copy from global memory to shared memory
    * warp 
    * [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#ampere-tuning)

* RTX 3070
    * 46 SMs

* Copy and compute pattern: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#copy_and_compute_pattern

# Exploration tool (or just use Yosys?)

* Read netlist
* Write as binary for speed?
* Start as 1 community
* Split into communities (Louvain)
* Exclude high fanout nets as an optimization
* Run Cuthill-McKee on each of community individually
* Statistics/histogram about fetch distance within a warp




