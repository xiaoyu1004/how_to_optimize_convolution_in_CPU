#ifndef GEMM_H
#define GEMM_H

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void gemm_cpu(int M, int N, int K, const void *a, const void *b, const void *bias, void *c);

#ifdef ENABLE_CUDA
    template <typename Tin, typename Tw, typename Tacc, typename Tout>
    void gemm_gpu(int M, int N, int K, const void *a, const void *b, const void *bias, void *c);
#endif

#endif