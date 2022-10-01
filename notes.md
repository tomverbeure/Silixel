
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
