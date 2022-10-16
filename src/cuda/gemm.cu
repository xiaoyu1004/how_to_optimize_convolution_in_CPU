#include "gemm.h"

template <typename Tin, typename Tw, typename Tacc, typename Tout>
__global__ void gemm_gpu_kernel(int M, int N, int K, const Tw *a, const Tin *b, const Tacc *bias, Tout *c)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (iy >= M || ix >= N) return;

    Tacc acc = 0;
    for (int k = 0; k < K; ++k)
    {
        acc += a[iy * K + k] * b[k * N + ix];
    }
    if (bias)
    {
        acc += bias[iy];
    }
    c[iy * N + ix] = static_cast<Tout>(acc);
}

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void gemm_gpu(int M, int N, int K, const Tw *a, const Tin *b, const Tacc *bias, Tout *c)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);
    gemm_gpu_kernel<Tin, Tw, Tacc, Tout><<<dimGrid, dimBlock>>>(M, N, K, a, b, bias, c);
}

template void gemm_gpu<float, float, float, float>(int M, int N, int K, const float *a, const float *b, const float *bias, float *c);