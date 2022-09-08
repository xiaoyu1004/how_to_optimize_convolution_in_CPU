#include "im2col_gemm_conv.h"

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
void sgemm(int M, int N, int K, const void *a, const void *b, const void *bias, float c)
{
    const Tw *A = static_cast<const Tw *>(a);
    const Tin *B = static_cast<const Tin *>(b);
    const Tacc *Bias = static_cast<const Tacc *>(bias);
    Tout *C = static_cast<Tout *>(c);

    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            Tacc acc = 0;
            for (int k = 0; k < K; ++k)
            {
                acc += (A[m * M + k] * B[k * N + n]);
            }
            if (Bias)
            {
                acc += Bias[m];
            }
            // TODO int8 
            C[m * N + n] = acc;
        }
    }
}

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void implict_precomp_sgemm(int input_n, int input_c, int input_h, int input_w,
                           int output_c, int kernel_h, int kernel_w,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dilation_h, int dilation_w,
                           int group_count,
                           const void *x, const void *w, const void *bias, void *y)
{
    int khd = (kernel_h - 1) * dilation_h + 1;
    int kwd = (kernel_w - 1) * dilation_w + 1;
    int output_h = (input_h - khd + 2 * pad_h) / stride_h + 1;
    int output_w = (input_w - kwd + 2 * pad_w) / stride_w + 1;

    int channels_col = input_c * kernel_h * kernel_w;
}