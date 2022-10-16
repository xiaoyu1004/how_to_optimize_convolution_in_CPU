#include "common.h"

int get_convolution_workspace_size(ConvolutionFwdAlgo_t algo,
                                   int input_n, int input_c, int input_h, int input_w,
                                   int output_c, int kernel_h, int kernel_w,
                                   int stride_h, int stride_w,
                                   int pad_h, int pad_w,
                                   int dilation_h, int dilation_w,
                                   int group_count)
{
    int khd = (kernel_h - 1) * dilation_h + 1;
    int kwd = (kernel_w - 1) * dilation_w + 1;
    int output_h = (input_h - khd + 2 * pad_h) / stride_h + 1;
    int output_w = (input_w - kwd + 2 * pad_w) / stride_w + 1;

    if (algo == CONVOLUTION_FWD_ALGO_GEMM)
    {
        return input_c * kernel_h * kernel_w * output_h * output_w;
    }

    return 0;
}