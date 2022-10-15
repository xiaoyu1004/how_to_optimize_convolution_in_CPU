#include "gemm.h"

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void gemm_cpu(int m, int n, int k, const void *a, const void *b, const void *bias, void *c)
{
    const Tw *A = static_cast<const Tw *>(a);
    const Tin *B = static_cast<const Tin *>(b);
    const Tacc *Bias = static_cast<const Tacc *>(bias);
    Tout *C = static_cast<Tout *>(c);

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
                acc += (A[i * k + l] * B[l * n + j]);
            }
            if (Bias)
            {
                acc += Bias[i];
            }
            C[i * n + j] = acc;
        }
    }
}

template void gemm_cpu<float, float, float, float>(int m, int n, int k, const void *a, const void *b, const void *bias, void *c);