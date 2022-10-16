#include "naive_conv.h"

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void naive_conv_cpu(int input_n, int input_c, int input_h, int input_w,
                    int output_c, int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dilation_h, int dilation_w,
                    int group_cnt,
                    const Tin *x, const Tw *w, const Tacc *bias, Tout *y)
{
    int khd = (kernel_h - 1) * dilation_h + 1;
    int kwd = (kernel_w - 1) * dilation_w + 1;
    int output_h = (input_h - khd + 2 * pad_h) / stride_h + 1;
    int output_w = (input_w - kwd + 2 * pad_w) / stride_w + 1;

    for (int n = 0; n < input_n; ++n)
    {
        for (int oc = 0; oc < output_c; ++oc)
        {
            for (int oh = 0; oh < output_h; ++oh)
            {
                int ih = oh * stride_h - pad_h;
                for (int ow = 0; ow < output_w; ++ow)
                {
                    Tacc result = static_cast<Tacc>(0.0f);
                    for (int ic = 0; ic < input_c; ++ic)
                    {
                        for (int kh = 0; kh < kernel_h; ++kh)
                        {
                            int ih = oh * stride_h + kh - pad_h;
                            if (ih < 0 || ih >= input_h)
                            {
                                continue;
                            }
                            for (int kw = 0; kw < kernel_w; ++kw)
                            {
                                int iw = ow * stride_w + kw - pad_w;
                                if (iw < 0 || iw >= input_w)
                                {
                                    continue;
                                }
                                int input_idx = ((n * input_c + ic) * input_h + ih) * input_w + iw;
                                int weight_idx = ((oc * input_c + ic) * kernel_h + kh) * kernel_w + kw;
                                result += x[input_idx] * w[weight_idx];
                            }
                        }
                    }

                    int output_idx = ((n * output_c + oc) * output_h + oh) * output_w + ow;
                    if (bias)
                    {
                        result += bias[oc];
                    }

                    y[output_idx] = result;
                }
            }
        }
    }
}

template void naive_conv_cpu<float, float, float, float>(int input_n, int input_c, int input_h, int input_w,
                                                         int output_c, int kernel_h, int kernel_w,
                                                         int stride_h, int stride_w,
                                                         int pad_h, int pad_w,
                                                         int dilation_h, int dilation_w,
                                                         int group_cnt,
                                                         const float *x, const float *w, const float *bias, float *y);