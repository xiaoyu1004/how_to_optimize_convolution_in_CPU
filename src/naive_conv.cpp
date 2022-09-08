#include "naive_conv.h"

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void naive_conv(int input_n, int input_c, int input_h, int input_w,
                int output_c, int kernel_h, int kernel_w,
                int stride_h, int stride_w,
                int pad_h, int pad_w,
                int dilation_h, int dilation_w,
                int group_cnt,
                const void *input_ptr, const void *weight_ptr, const void *bias, void *output_ptr)
{
    const Tin *input_data = static_cast<const Tin *>(input_ptr);
    const Tw *weight_data = static_cast<const Tw *>(weight_ptr);
    Tout *output_data = static_cast<Tout *>(output_ptr);
    const Tacc *bias_data = static_cast<const Tacc *>(bias);

    int khd = (kernel_h - 1) * dilation_h + 1;
    int kwd = (kernel_w - 1) * dilation_w + 1;
    int output_h = (input_h - khd + 2 * pad_h) / stride_h + 1;
    int output_w = (input_w - kwd + 2 * pad_w) / stride_w + 1;

    int output_channels_per_group = output_c / group_cnt;
    int input_channels_per_group = input_c / group_cnt;

    for (int n = 0; n < input_n; ++n)
    {
        for (int g = 0; g < group_cnt; ++g)
        {
            int output_c_start = g * output_channels_per_group;
            int output_c_end = (g + 1) * output_channels_per_group;
            int input_c_start = g * input_channels_per_group;
            int input_c_end = (g + 1) * input_channels_per_group;
            int weights_start = g * output_channels_per_group * input_channels_per_group * kernel_w * kernel_h;
            for (int output_c = output_c_start; output_c < output_c_end; ++output_c)
            {
                for (int h = 0; h < output_h; ++h)
                {
                    int input_h_start = h * stride_h - pad_h;
                    for (int w = 0; w < output_w; ++w)
                    {
                        int input_w_start = w * stride_w - pad_w;
                        Tacc result = static_cast<Tacc>(0.0f);
                        for (int kh = 0; kh < kernel_h; ++kh)
                        {
                            int ih = input_h_start + kh * dilation_h;
                            if (ih < 0 || ih >= input_h)
                            {
                                continue;
                            }
                            for (int kw = 0; kw < kernel_w; ++kw)
                            {
                                int iw = input_w_start + kw * dilation_w;
                                if (iw < 0 || iw >= input_w)
                                {
                                    continue;
                                }
                                for (int input_c = input_c_start; input_c < input_c_end; ++input_c)
                                {
                                    int input_position = ((n * input_c + input_c) * input_h + ih) * input_w + iw;
                                    int weight_position = weights_start + (((output_c - output_c_start) * input_channels_per_group + input_c - input_c_start) * kernel_h + kh) * kernel_w + kw;
                                    result += input_data[input_position] * weight_data[weight_position];
                                }
                            }
                        }

                        int output_position = ((n * output_c + output_c) * output_h + h) * output_w + w;
                        if (bias_data)
                        {
                            result += bias_data[output_c];
                        }

                        output_data[output_position] = result;
                    }
                }
            }
        }
    }
}

template void naive_conv<float, float, float, float>(int input_n, int input_c, int input_h, int input_w,
                                                     int output_c, int kernel_h, int kernel_w,
                                                     int stride_h, int stride_w,
                                                     int pad_h, int pad_w,
                                                     int dilation_h, int dilation_w,
                                                     int group_cnt,
                                                     const void *input_ptr, const void *weight_ptr, const void *bias, void *output_ptr);