#include "common.h"

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void implicit_conv2d_cpu(int input_n, int input_c, int input_h, int input_w,
                         int output_c, int kernel_h, int kernel_w,
                         int stride_h, int stride_w,
                         int pad_h, int pad_w,
                         int dilation_h, int dilation_w,
                         int group_count,
                         const Tin *x, const Tw *w, const Tacc *bias, Tout *y)
{
    int khd = (kernel_h - 1) * dilation_h + 1;
    int kwd = (kernel_w - 1) * dilation_w + 1;
    int output_h = (input_h - khd + 2 * pad_h) / stride_h + 1;
    int output_w = (input_w - kwd + 2 * pad_w) / stride_w + 1;

    int M = output_c;
    int N = input_n * output_h * output_w;
    int K = input_c * kernel_h * kernel_w;

    for (int i = 0; i < M; ++i)
    {
        int oc = i;
        for (int j = 0; j < N; ++j)
        {
            int n = j / (output_h * output_w);
            int oh = j % (output_h * output_w) / output_w;
            int ow = j % (output_h * output_w) % output_w;

            Tacc result = static_cast<Tacc>(0);
            for (int k = 0; k < K; ++k)
            {
                int kh = k % (kernel_h * kernel_w) / kernel_w;
                int kw = k % (kernel_h * kernel_w) % kernel_w;
                int ic = k / (kernel_h * kernel_w);
                int ih = oh * stride_h - pad_h + kh;
                int iw = ow * stride_w - pad_w + kw;

                if (ih < 0 || ih >= input_h || iw < 0 || iw >= input_w)
                {
                    continue;
                }

                int kernel_idx = oc * K + k;
                int input_idx = n * input_c * input_h * input_w +
                                ic * input_h * input_w +
                                ih * input_w +
                                iw;
                result += w[kernel_idx] * x[input_idx];
            }
            if (bias)
            {
                result += bias[oc];
            }

            int output_idx = n * output_c * output_h * output_w +
                             oc * output_h * output_w +
                             oh * output_w +
                             ow;
            y[output_idx] = static_cast<Tout>(result);
        }
    }
}

template void implicit_conv2d_cpu<float, float, float, float>(int input_n, int input_c, int input_h, int input_w,
                                                              int output_c, int kernel_h, int kernel_w,
                                                              int stride_h, int stride_w,
                                                              int pad_h, int pad_w,
                                                              int dilation_h, int dilation_w,
                                                              int group_cnt,
                                                              const float *x, const float *w, const float *bias, float *y);