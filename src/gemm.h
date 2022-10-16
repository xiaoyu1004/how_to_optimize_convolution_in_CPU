#ifndef GEMM_H
#define GEMM_H

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void gemm_cpu(int m, int n, int k, const Tw *a, const Tin *b, const Tacc *bias, Tout *c);

#ifdef ENABLE_CUDA
template <typename Tin, typename Tw, typename Tacc, typename Tout>
void gemm_gpu(int m, int n, int k, const Tw *a, const Tin *b, const Tacc *bias, Tout *c);
#endif

#endif