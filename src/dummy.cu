
#include <stdio.h>
#include <stdint.h>

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

    printf("Elements: %ld, memory: %ld\n", (long int)N, (long int)N*3*4);

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

#define int_type uint8_t

extern "C" __global__ void CudaBWTest_kernel(
        const int_type *A, 
        const int_type *B,
        int_type *C, 
        int N) 
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N) C[i] = A[i] + B[i];
}

extern "C" void CudaBWTest() 
{
    int N   = 64 * 1024 * 1024;

    int_type *A;
    int_type *B;
    int_type *C; 

    printf("Elements: %ld, memory: %ld\n", (long int)N, (long int)N*3*sizeof(int8_t));

    cudaMallocManaged(&A, N*sizeof(int_type));
    cudaMallocManaged(&B, N*sizeof(int_type));
    cudaMallocManaged(&C, N*sizeof(int_type));

    for(int i=0;i<N;++i){
        A[i] = (int_type)i;
        B[i] = (int_type)i;
    }

    cudaMemPrefetchAsync(A, N*sizeof(int_type), g_cuDevice);
    cudaMemPrefetchAsync(B, N*sizeof(int_type), g_cuDevice);

    int blockSize = 256;
    int numBlocks = (N+blockSize-1)/blockSize;

    CudaBWTest_kernel<<<numBlocks,blockSize>>>(A, B, C, N);
    CudaBWTest_kernel<<<numBlocks,blockSize>>>(A, B, C, N);
    CudaBWTest_kernel<<<numBlocks,blockSize>>>(A, B, C, N);
    CudaBWTest_kernel<<<numBlocks,blockSize>>>(A, B, C, N);
    CudaBWTest_kernel<<<numBlocks,blockSize>>>(A, B, C, N);
    cudaDeviceSynchronize();

    for(int i=0;i<10;++i){
        printf("%i: %d + %d = %d\n", i, A[i], B[i], C[i]);
    }

    for(int i=N-10;i<N;++i){
        printf("%i: %d + %d = %d\n", i, A[i], B[i], C[i]);
    }
}


