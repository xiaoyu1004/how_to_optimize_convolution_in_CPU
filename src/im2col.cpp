#include "im2col.h"

template <typename Tin>
void im2col_cpu(int input_n, int input_c, int input_h, int input_w,
                int output_c, int kernel_h, int kernel_w,
                int stride_h, int stride_w,
                int pad_h, int pad_w,
                int dilation_h, int dilation_w,
                int group_count,
                const Tin *x, Tin *y)
{
    int khd = (kernel_h - 1) * dilation_h + 1;
    int kwd = (kernel_w - 1) * dilation_w + 1;
    int output_h = (input_h - khd + 2 * pad_h) / stride_h + 1;
    int output_w = (input_w - kwd + 2 * pad_w) / stride_w + 1;

    int channels_col = input_c * kernel_h * kernel_w;

    for (int c = 0; c < channels_col; c++)
    {
        int ic = c / (kernel_h * kernel_w);
        int kh = c % (kernel_h * kernel_w) / kernel_w;
        int kw = c % (kernel_h * kernel_w) % kernel_w;

        for (int oh = 0; oh < output_h; ++oh)
        {
            int ih = oh * stride_h + kh - pad_h;
            for (int ow = 0; ow < output_w; ++ow)
            {
                int iw = ow * stride_w + kw - pad_w;
                Tin val = 0;
                if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w)
                {
                    int input_idx = ic * input_h * input_w + 
                                    ih * input_w + 
                                    iw;
                    val = x[input_idx];
                }
                int output_idx = c * output_h * output_w + 
                                 oh * output_w + 
                                 ow;
                y[output_idx] = val;
            }
        }
    }
}

template void im2col_cpu<float>(int input_n, int input_c, int input_h, int input_w,
                                int output_c, int kernel_h, int kernel_w,
                                int stride_h, int stride_w,
                                int pad_h, int pad_w,
                                int dilation_h, int dilation_w,
                                int group_count,
                                const float *x, float *y);