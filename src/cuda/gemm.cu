#include "gemm.h"

template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K,
          typename Tin, typename Tw, typename Tacc, typename Tout>
__global__ void gemm_gpu_kernel(int M, int N, int K, const Tw *A, const Tin *B, const Tacc *Bias, Tout *C)
{
    __shared__ float SLB_A[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float SLB_B[BLOCK_SIZE_K][BLOCK_SIZE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int g_idx = bx * BLOCK_SIZE_N + tx;
    int g_idy = by * BLOCK_SIZE_M + ty;

    Tacc Csub = static_cast<Tacc>(0);

    const Tin *A_gmem_ptr = A + blockIdx.y * BLOCK_SIZE_M * K;
    const Tin *B_gmem_ptr = B + blockIdx.x * BLOCK_SIZE_N;

    for (int idx_k = 0; idx_k < K; idx_k += BLOCK_SIZE_K)
    {
        Tin val_a = static_cast<Tin>(0);
        // load gmem from A to smem
        if ((by * BLOCK_SIZE_M + ty) < M && (idx_k + tx) < K)
        {
            val_a = A_gmem_ptr[ty * K + tx + idx_k];
        }
        SLB_A[ty][tx] = val_a;

        Tin val_b = static_cast<Tin>(0);
        // load gmem from B to smem
        if ((idx_k + ty) < K && (bx * BLOCK_SIZE_N + tx) < N)
        {
            val_b = B_gmem_ptr[(ty + idx_k) * N + tx];
        }
        SLB_B[ty][tx] = val_b;

        __syncthreads();

        // calculate
        for (int p = 0; p < BLOCK_SIZE_K; ++p)
        {
            Csub += SLB_A[ty][p] * SLB_B[p][tx];
        }
        __syncthreads();
    }

    // store to c & bias
    if (Bias)
    {
        if ((by * BLOCK_SIZE_M + ty) < M && (bx * BLOCK_SIZE_N + tx) < N)
        {
            Csub += Bias[g_idy];
        }
    }
    if ((by * BLOCK_SIZE_M + ty) < M && (bx * BLOCK_SIZE_N + tx) < N)
    {
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
    gemm_gpu_kernel<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, Tin, Tw, Tacc, Tout><<<dimGrid, dimBlock>>>(M, N, K, a, b, bias, c);
}

template void gemm_gpu<float, float, float, float>(int M, int N, int K, const float *a, const float *b, const float *bias, float *c);