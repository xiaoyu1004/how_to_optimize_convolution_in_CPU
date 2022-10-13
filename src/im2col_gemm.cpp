#include "im2col_gemm.h"

#include <cstring>
#include <intrin.h>

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define UP_DIV(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y))

template <typename Tin>
void im2col(int input_n, int input_c, int input_h, int input_w,
            int output_c, int kernel_h, int kernel_w,
            int stride_h, int stride_w,
            int pad_h, int pad_w,
            int dilation_h, int dilation_w,
            int group_count,
            const void *x, void *y)
{
    const Tin *src = static_cast<const Tin *>(x);
    Tin *dst = static_cast<Tin *>(y);

    int khd = (kernel_h - 1) * dilation_h + 1;
    int kwd = (kernel_w - 1) * dilation_w + 1;
    int output_h = (input_h - khd + 2 * pad_h) / stride_h + 1;
    int output_w = (input_w - kwd + 2 * pad_w) / stride_w + 1;

    int channels_col = input_c * kernel_h * kernel_w;

    // im2col
    for (int c = 0; c < channels_col; c++)
    {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / kernel_h / kernel_w;

        int h_base = h_offset * dilation_h - pad_h;
        int w_base = w_offset * dilation_w - pad_w;

        int h_base_start = MAX(0, (UP_DIV(-h_base, stride_h)));
        int h_base_end = MIN(output_h, UP_DIV(input_h - h_base, stride_h));
        int w_base_start = MAX(0, (UP_DIV(-w_base, stride_w)));
        int w_base_end = MIN(output_w, UP_DIV(input_w - w_base, stride_w));

        auto src_c = src + c_im * input_h * input_w;
        auto dst_c = dst + c * output_h * output_w;

        memset(dst_c, 0, h_base_start * output_w * sizeof(Tin));
        for (int h = h_base_start; h < h_base_end; h++)
        {
            int h_pad = h_base + h * stride_h;

            auto src_h = src_c + h_pad * input_w;
            auto dst_h = dst_c + h * output_w;

            for (int w = 0; w < w_base_start; w++)
            {
                dst_h[w] = 0;
            }
            for (int w = w_base_start; w < w_base_end; w++)
            {
                int w_pad = w_base + w * stride_w;
                dst_h[w] = src_h[w_pad];
            }
            for (int w = w_base_end; w < output_w; w++)
            {
                dst_h[w] = 0;
            }
        }
        memset(dst_c + h_base_end * output_w, 0, (output_h - h_base_end) * output_w * sizeof(Tin));
    }
}

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void sgemm(int m, int n, int k, const void *a, const void *b, const void *bias, void *c)
{
    const Tw *A = static_cast<const Tw *>(a);
    const Tin *B = static_cast<const Tin *>(b);
    const Tacc *Bias = static_cast<const Tacc *>(bias);
    Tout *C = static_cast<Tout *>(c);

    int lda = k;
    int ldb = n;
    int ldc = n;

    /*
    int i = 0;
    int partM = m - m % 4;
    for (; i < partM; i += 4)
    {
        int j = 0;
        int partN = n - n % 4;
        for (; j < partN; j += 4)
        {
            __m128 c_0 = _mm_load_ps(C + i * ldc + j);
            __m128 c_1 = _mm_load_ps(C + (i + 1) * ldc + j);
            __m128 c_2 = _mm_load_ps(C + (i + 2) * ldc + j);
            __m128 c_3 = _mm_load_ps(C + (i + 3) * ldc + j);

            for (int p = 0; p < k; ++p)
            {
                const __m128 a_0 = _mm_load_ps1(A + i * lda + p);
                const __m128 a_1 = _mm_load_ps1(A + (i + 1) * lda + p);
                const __m128 a_2 = _mm_load_ps1(A + (i + 2) * lda + p);
                const __m128 a_3 = _mm_load_ps1(A + (i + 3) * lda + p);

                const __m128 b_v = _mm_load_ps(B + p * ldb + j);

                c_0 = _mm_fmadd_ps(a_0, b_v, c_0);
                c_1 = _mm_fmadd_ps(a_1, b_v, c_1);
                c_2 = _mm_fmadd_ps(a_2, b_v, c_2);
                c_3 = _mm_fmadd_ps(a_3, b_v, c_3);
            }

            _mm_store_ps(C + i * ldc + j, c_0);
            _mm_store_ps(C + (i + 1) * ldc + j, c_1);
            _mm_store_ps(C + (i + 2) * ldc + j, c_2);
            _mm_store_ps(C + (i + 3) * ldc + j, c_3);
        }
        for (; j < n; ++j)
        {
            float v0 = 0.f;
            float v1 = 0.f;
            float v2 = 0.f;
            float v3 = 0.f;

            for (int p = 0; p < k; ++p)
            {
                float b_v = B[p * ldb + j];

                v0 += A[i * lda + p] * b_v;
                v1 += A[(i + 1) * lda + p] * b_v;
                v2 += A[(i + 2) * lda + p] * b_v;
                v3 += A[(i + 3) * lda + p] * b_v;
            }

            C[i * ldc + j] += v0;
            C[(i + 1) * ldc + j] += v1;
            C[(i + 2) * ldc + j] += v2;
            C[(i + 3) * ldc + j] += v3;
        }
    }

    for (; i < m; ++i)
    {
        int j = 0;
        int partN = n - n % 4;

        for (; j < partN; j += 4)
        {
            float v0 = 0.f;
            float v1 = 0.f;
            float v2 = 0.f;
            float v3 = 0.f;

            for (int p = 0; p < k; ++p)
            {
                float a_v = A[i * lda + p];

                v0 += a_v * B[p * ldb + j];
                v1 += a_v * B[p * ldb + j + 1];
                v2 += a_v * B[p * ldb + j + 2];
                v3 += a_v * B[p * ldb + j + 3];
            }

            C[i * ldc + j] += v0;
            C[i * ldc + j + 1] += v1;
            C[i * ldc + j + 2] += v2;
            C[i * ldc + j + 3] += v3;
        }
        for (; j < n; ++j)
        {
            for (int p = 0; p < k; ++p)
            {
                C[i * ldc + j] += A[i * lda + p] * B[p * ldb + j];
            }
        }
    }
    */

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
            // TODO int8
            C[i * n + j] = acc;
        }
    }
}

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void con2d_gemm(int input_n, int input_c, int input_h, int input_w,
                 int output_c, int kernel_h, int kernel_w,
                 int stride_h, int stride_w,
                 int pad_h, int pad_w,
                 int dilation_h, int dilation_w,
                 int group_count,
                 const void *x, const void *w, const void *bias, void *y)
{
    const Tin *input_data = static_cast<const Tin *>(x);
    const Tw *weight_data = static_cast<const Tw *>(w);
    const Tacc *bias_data = static_cast<const Tacc *>(bias);
    Tout *output_data = static_cast<Tout *>(y);

    int khd = (kernel_h - 1) * dilation_h + 1;
    int kwd = (kernel_w - 1) * dilation_w + 1;
    int output_h = (input_h - khd + 2 * pad_h) / stride_h + 1;
    int output_w = (input_w - kwd + 2 * pad_w) / stride_w + 1;
    int channels_col = input_c * kernel_h * kernel_w;
    size_t im2col_size = channels_col * output_h * output_w;
    Tin *im2col_workspace = new Tin[im2col_size];
    // im2col
    im2col<Tin>(input_n, input_c, input_h, input_w,
                output_c, kernel_h, kernel_w,
                stride_h, stride_w,
                pad_h, pad_w,
                dilation_h, dilation_w,
                group_count,
                input_data, im2col_workspace);

    // gemm
    sgemm<Tin, Tw, Tacc, Tout>(output_c, output_h * output_w, channels_col,
                               weight_data, im2col_workspace, bias_data, output_data);
    delete[] im2col_workspace;
}

template void con2d_gemm<float, float, float, float>(int input_n, int input_c, int input_h, int input_w,
                                                      int output_c, int kernel_h, int kernel_w,
                                                      int stride_h, int stride_w,
                                                      int pad_h, int pad_w,
                                                      int dilation_h, int dilation_w,
                                                      int group_count,
                                                      const void *x, const void *w, const void *bias, void *y);