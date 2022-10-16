#include "im2col.h"

template <typename Tin>
__global__ void im2col_kernel(int num_threads,
                              int input_n, int input_c, int input_h, int input_w,
                              int output_c, int kernel_h, int kernel_w,
                              int output_h, int output_w,
                              int stride_h, int stride_w,
                              int pad_h, int pad_w,
                              int dilation_h, int dilation_w,
                              int group_count,
                              const Tin *x, Tin *y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) return;

    int channels_col = input_c * kernel_h * kernel_w;
    int width_col = output_h * output_w;
    int c = tid / width_col;
    int oh = tid % width_col / output_w;
    int ow = tid % width_col % output_w;

    int ic = c / (kernel_h * kernel_w);
    int kh = c % (kernel_h * kernel_w) / kernel_w;
    int kw = c % (kernel_h * kernel_w) % kernel_w;

    int ih = oh * stride_h + kh - pad_h;
    int iw = ow * stride_w + kw - pad_w;

    Tin val = 0;
    if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w)
    {
        int input_idx = ic * input_h * input_w + 
                        ih * input_w + 
                        iw;
        val = x[input_idx];
    }
    y[tid] = val;
}

template <typename Tin>
void im2col_gpu(int input_n, int input_c, int input_h, int input_w,
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
    int num_threads = channels_col * output_h * output_w;

    dim3 dimBlock(1024);
    dim3 dimGrid((num_threads + dimBlock.x - 1) / dimBlock.x);
    im2col_kernel<Tin><<<dimGrid, dimBlock>>>(num_threads,
                                             input_n, input_c, input_h, input_w,
                                             output_c, kernel_h, kernel_w,
                                             output_h, output_w,
                                             stride_h, stride_w,
                                             pad_h, pad_w,
                                             dilation_h, dilation_w,
                                             group_count,
                                             x, y);
}

template void im2col_gpu<float>(int input_n, int input_c, int input_h, int input_w,
                                int output_c, int kernel_h, int kernel_w,
                                int stride_h, int stride_w,
                                int pad_h, int pad_w,
                                int dilation_h, int dilation_w,
                                int group_count,
                                const float *x, float *y);