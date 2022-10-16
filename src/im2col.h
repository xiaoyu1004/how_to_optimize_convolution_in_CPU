#ifndef IM2COL_H
#define IM2COL_H

template <typename Tin>
void im2col_cpu(int input_n, int input_c, int input_h, int input_w,
                int output_c, int kernel_h, int kernel_w,
                int stride_h, int stride_w,
                int pad_h, int pad_w,
                int dilation_h, int dilation_w,
                int group_count,
                const Tin *x, Tin *y);

#ifdef ENABLE_CUDA
template <typename Tin>
void im2col_gpu(int input_n, int input_c, int input_h, int input_w,
                int output_c, int kernel_h, int kernel_w,
                int stride_h, int stride_w,
                int pad_h, int pad_w,
                int dilation_h, int dilation_w,
                int group_count,
                const Tin *x, Tin *y);
#endif

#endif