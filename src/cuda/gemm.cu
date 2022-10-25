#include "gemm.h"

template <constexpr int BLOCK_SIZE_M, constexpr int BLOCK_SIZE_N, constexpr int BLOCK_SIZE_K,
          typename Tin, typename Tw, typename Tacc, typename Tout>
__global__ void gemm_gpu_kernel(int M, int N, int K, const Tw *A, const Tin *B, const Tacc *Bias, Tout *C)
{
    __shared__ float SLB_A[BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float SLB_B[BLOCK_SIZE_K][BLOCK_SIZE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int g_idx = blockIdx.x * BLOCK_SIZE_N + tx;
    int g_idy = blockIdx.y * BLOCK_SIZE_M + ty;

    Tacc Csub = static_cast<Tacc>(0);

    for (int idx_k = 0; idx_k < K; idx_k += BLOCK_SIZE_K)
    {
        Tin *A_gmem_ptr = A + blockIdx.y * BLOCK_SIZE_M * N + idx_k;
        Tin *B_gmem_ptr = B + blockIdx.x * BLOCK_SIZE_N + idx_k * N;

        // load gmem from A to smem
        if ((idx_k + tx) < K && g_idy < M)
        {
            SLB_A[ty][tx] = A_gmem_ptr[ty * K + tx];
        }
        // load gmem from B to smem
        if ((idx_k + ty) < K && g_idx < N)
        {
            SLB_B[ty][tx] = B_gmem_ptr[ty * N + tx];
        }

        __syncthreads();

        // calculate
        for (int p = 0; p < BLOCK_SIZE_K; ++p)
        {
            if ((idx_k + tx) < K && g_idy < M && g_idx < N)
            {
                Csub += SLB_A[ty][p] * SLB_B[p][tx];
            }
        }
    }

    // store to c & bias
    if (g_idy < M && g_idx < N)
    {
        if (Bias)
        {
            Csub += Bias[g_idy];
        }
        C[g_idy * N + g_idx] = Csub;
    }
}

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void gemm_gpu(int M, int N, int K, const Tw *a, const Tin *b, const Tacc *bias, Tout *c)
{
    constexpr int BLOCK_SIZE_M = 32;
    constexpr int BLOCK_SIZE_N = 32;
    constexpr int BLOCK_SIZE_K = 32;

    dim3 dimBlock(BLOCK_SIZE_N, BLOCK_SIZE_M);
    dim3 dimGrid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    gemm_gpu_kernel<Tin, Tw, Tacc, Tout><<<dimGrid, dimBlock>>>(M, N, K, a, b, bias, c);
}

template void gemm_gpu<float, float, float, float>(int M, int N, int K, const float *a, const float *b, const float *bias, float *c);