#include "gemm.h"

template <typename Tin, typename Tw, typename Tacc, typename Tout>
__global__ void gemm_gpu_kernel(int M, int N, int K, const void *a, const void *b, const void *bias, void *c)
{
    const Tw *A = static_cast<const Tw *>(a);
    const Tin *B = static_cast<const Tin *>(b);
    const Tacc *Bias = static_cast<const Tacc *>(bias);
    Tout *C = static_cast<Tout *>(c);

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (iy >= M || ix >= N) return;

    Tacc acc = 0;
    for (int k = 0; k < K; ++k)
    {
        acc += A[iy * K + k] * B[k * N + ix];
    }
    if (Bias)
    {
        acc += Bias[iy];
    }
    C[iy * N + ix] = static_cast<Tout>(acc);
}

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void gemm_gpu(int M, int N, int K, const void *a, const void *b, const void *bias, void *c)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);
    gemm_gpu_kernel<Tin, Tw, Tacc, Tout><<<dimGrid, dimBlock>>>(M, N, K, a, b, bias, c);
}