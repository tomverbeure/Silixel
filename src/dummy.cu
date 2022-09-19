
#include <stdio.h>

#include <cuda.h>

extern "C" __global__ void CudaDummy_kernel(
        const float *A, 
        const float *B,
        float *C, 
        int N) 
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N) C[i] = A[i] + B[i];
}

extern CUdevice g_cuDevice;

extern "C" void CudaDummy() 
{
    int N   = 5 * 64 * 1024 * 1024;

    float *A;
    float *B;
    float *C; 

    printf("Elements: %ld, memory: %ld\n", N, N*3*4);

    cudaMallocManaged(&A, N*sizeof(float));
    cudaMallocManaged(&B, N*sizeof(float));
    cudaMallocManaged(&C, N*sizeof(float));

    for(int i=0;i<N;++i){
        A[i] = (float)i;
        B[i] = (float)i;
    }

    cudaMemPrefetchAsync(A, N*sizeof(float), g_cuDevice);
    cudaMemPrefetchAsync(B, N*sizeof(float), g_cuDevice);

    int blockSize = 256;
    int numBlocks = (N+blockSize-1)/blockSize;

    CudaDummy_kernel<<<numBlocks,blockSize>>>(A, B, C, N);
    CudaDummy_kernel<<<numBlocks,blockSize>>>(A, B, C, N);
    CudaDummy_kernel<<<numBlocks,blockSize>>>(A, B, C, N);
    CudaDummy_kernel<<<numBlocks,blockSize>>>(A, B, C, N);
    CudaDummy_kernel<<<numBlocks,blockSize>>>(A, B, C, N);
    cudaDeviceSynchronize();

    for(int i=0;i<10;++i){
        printf("%i: %f + %f = %f\n", i, A[i], B[i], C[i]);
    }

    for(int i=N-10;i<N;++i){
        printf("%i: %f + %f = %f\n", i, A[i], B[i], C[i]);
    }
}
