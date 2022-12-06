#ifndef NAIVE_CONV_H
#define NAIVE_CONV_H

#include "common.h"

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void naive_conv_cpu(int input_n, int input_c, int input_h, int input_w,
                    int output_c, int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dilation_h, int dilation_w,
                    int group_count,
                    const Tin *x, const Tw *w, const Tacc *bias, Tout *y);

#ifdef ENABLE_CUDA
template <typename Tin, typename Tw, typename Tacc, typename Tout>
void naive_conv_gpu(int input_n, int input_c, int input_h, int input_w,
                    int output_c, int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dilation_h, int dilation_w,
                    int group_count,
                    const Tin *x, const Tw *w, const Tacc *bias, Tout *y);
#endif

#endif