#include "im2col_conv2d.h"
#include "im2col.h"
#include "gemm.h"

#include <cstring>
#include <intrin.h>

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void im2col_conv2d_cpu(int input_n, int input_c, int input_h, int input_w,
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
    im2col_cpu<Tin>(input_n, input_c, input_h, input_w,
                    output_c, kernel_h, kernel_w,
                    stride_h, stride_w,
                    pad_h, pad_w,
                    dilation_h, dilation_w,
                    group_count,
                    input_data, im2col_workspace);

    // gemm
    gemm_cpu<Tin, Tw, Tacc, Tout>(output_c, output_h * output_w, channels_col,
                                  weight_data, im2col_workspace, bias_data, output_data);
    delete[] im2col_workspace;
}

template void im2col_conv2d_cpu<float, float, float, float>(int input_n, int input_c, int input_h, int input_w,
                                                            int output_c, int kernel_h, int kernel_w,
                                                            int stride_h, int stride_w,
                                                            int pad_h, int pad_w,
                                                            int dilation_h, int dilation_w,
                                                            int group_count,
                                                            const void *x, const void *w, const void *bias, void *y);