#ifndef CONV_2D_H
#define CONV_2D_H

#include "common.h"
#include "naive_conv.h"

#include <iostream>

void conv2d(unsigned input_n, unsigned input_c, unsigned input_h, unsigned input_w,
            unsigned output_c, unsigned kernel_h, unsigned kernel_w,
            unsigned stride_h, unsigned stride_w,
            unsigned pad_h, unsigned pad_w,
            unsigned dialation_h, unsigned dialation_w,
            unsigned group_count,
            ConvolutionFwdAlgo_t algo,
            const float *x, const float *w, const float *bias, float *y)
{
    if (algo == CONVOLUTION_FWD_ALGO_DIRECT)
    {
        naive_conv(input_n, input_c, input_h, input_w,
                   output_c, kernel_h, kernel_w,
                   stride_h, stride_w,
                   pad_h, pad_w,
                   dialation_h, dialation_w,
                   group_count,
                   x, w, bias, y);
    }
    else if (algo == CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
    {

    }
    else
    {
        std::cout << "ERROR: unsurported FWD\n";
        std::terminate();
    }
}

#endif