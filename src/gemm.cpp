#include "gemm.h"

#include <type_traits>

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void gemm_cpu(int m, int n, int k, const Tw *a, const Tin *b, const Tacc *bias, Tout *c)
{
    int lda = k;
    int ldb = n;
    int ldc = n;

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            Tacc acc = 0;
            for (int l = 0; l < k; ++l)
            {
                acc += (a[i * k + l] * b[l * n + j]);
            }
            if (bias)
            {
                acc += bias[i];
            }

            if (std::is_same<Tout, half>::value)
            {
                c[i * n + j] = __float2half(acc);
            }
            else
            {
                c[i * n + j] = acc;
            }
        }
    }
}

template void gemm_cpu<float, float, float, float>(int m, int n, int k, const float *a, const float *b, const float *bias, float *c);
template void gemm_cpu<half, half, float, half>(int m, int n, int k, const half *a, const half *b, const float *bias, half *c);