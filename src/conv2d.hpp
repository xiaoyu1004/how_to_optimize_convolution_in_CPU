#ifndef CONV_2D_HPP
#define CONV_2D_HPP

#include "common.h"
#include "naive_conv.h"
#include "im2col_conv2d.h"
#include "implicit_conv2d.h"

#include <iostream>

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void Conv2dCPU(ConvolutionFwdAlgo_t algo,
               int input_n, int input_c, int input_h, int input_w,
               int output_c, int kernel_h, int kernel_w,
               int stride_h, int stride_w,
               int pad_h, int pad_w,
               int dilation_h, int dilation_w,
               int group_count,
               const Tin *x, const Tw *w, const Tacc *bias, Tout *y)
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
void Conv2dGPU(ConvolutionFwdAlgo_t algo,
               int input_n, int input_c, int input_h, int input_w,
               int output_c, int kernel_h, int kernel_w,
               int stride_h, int stride_w,
               int pad_h, int pad_w,
               int dilation_h, int dilation_w,
               int group_count,
               Tin *workspace,
               const Tin *x, const Tw *w, const Tacc *bias, Tout *y)
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
    else if (algo == CONVOLUTION_FWD_ALGO_GEMM)
    {
        im2col_conv2d_gpu<Tin, Tw, Tacc, Tout>(input_n, input_c, input_h, input_w,
                                               output_c, kernel_h, kernel_w,
                                               stride_h, stride_w,
                                               pad_h, pad_w,
                                               dilation_h, dilation_w,
                                               group_count,
                                               workspace,
                                               x, w, bias, y);
    }
    else if (algo == CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
    {
        implicit_conv2d_gpu<Tin, Tw, Tacc, Tout>(input_n, input_c, input_h, input_w,
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
    CUDA_CHECK(cudaDeviceSynchronize());
}
#endif // ENABLE_CUDA

#endif