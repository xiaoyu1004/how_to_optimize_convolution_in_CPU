#ifndef CONV_2D_H
#define CONV_2D_H

#include "common.h"
#include "naive_conv.h"
#include "im2col_conv2d.h"
#include "implicit_conv2d.h"

#include <iostream>

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void Conv2dCPU(int input_n, int input_c, int input_h, int input_w,
               int output_c, int kernel_h, int kernel_w,
               int stride_h, int stride_w,
               int pad_h, int pad_w,
               int dilation_h, int dilation_w,
               int group_count,
               ConvolutionFwdAlgo_t algo,
               const void *x, const void *w, const void *bias, void *y)
{
    if (algo == CONVOLUTION_FWD_ALGO_DIRECT)
    {
        naive_conv_cpu<Tin, Tw, Tacc, Tout>(input_n, input_c, input_h, input_w,
                                            output_c, kernel_h, kernel_w,
                                            stride_h, stride_w,
                                            pad_h, pad_w,
                                            dilation_h, dilation_w,
                                            group_count,
                                            x, w, bias, y);
    }
    else if (algo == CONVOLUTION_FWD_ALGO_GEMM)
    {
        im2col_conv2d_cpu<Tin, Tw, Tacc, Tout>(input_n, input_c, input_h, input_w,
                                               output_c, kernel_h, kernel_w,
                                               stride_h, stride_w,
                                               pad_h, pad_w,
                                               dilation_h, dilation_w,
                                               group_count,
                                               x, w, bias, y);
    }
    else if (algo == CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
    {
        implicit_conv2d_cpu<Tin, Tw, Tacc, Tout>(input_n, input_c, input_h, input_w,
                                                 output_c, kernel_h, kernel_w,
                                                 stride_h, stride_w,
                                                 pad_h, pad_w,
                                                 dilation_h, dilation_w,
                                                 group_count,
                                                 x, w, bias, y);
    }
    else
    {
        std::cout << "ERROR: unsurported FWD\n";
        std::terminate();
    }
}

#ifdef ENABLE_CUDA
template <typename Tin, typename Tw, typename Tacc, typename Tout>
void Conv2dGPU(int input_n, int input_c, int input_h, int input_w,
               int output_c, int kernel_h, int kernel_w,
               int stride_h, int stride_w,
               int pad_h, int pad_w,
               int dilation_h, int dilation_w,
               int group_count,
               ConvolutionFwdAlgo_t algo,
               const void *x, const void *w, const void *bias, void *y)
{
    if (algo == CONVOLUTION_FWD_ALGO_DIRECT)
    {
        naive_conv_gpu<Tin, Tw, Tacc, Tout>(input_n, input_c, input_h, input_w,
                                            output_c, kernel_h, kernel_w,
                                            stride_h, stride_w,
                                            pad_h, pad_w,
                                            dilation_h, dilation_w,
                                            group_count,
                                            x, w, bias, y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}
#endif

#endif