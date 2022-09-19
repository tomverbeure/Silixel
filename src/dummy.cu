
#include <stdio.h>


extern "C" __global__ void CudaDummy_kernel(
        const float *A, 
        const float *B,
        float *C, 
        int N) 
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N) C[i] = A[i] + B[i];
}

extern "C" void CudaDummy() 
{
    int N   = 100000000;

    float *A;
    float *B;
    float *C; 

    cudaMallocManaged(&A, N*sizeof(float));
    cudaMallocManaged(&B, N*sizeof(float));
    cudaMallocManaged(&C, N*sizeof(float));


    for(int i=0;i<N;++i){
        A[i] = (float)i;
        B[i] = (float)i;
    }

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
