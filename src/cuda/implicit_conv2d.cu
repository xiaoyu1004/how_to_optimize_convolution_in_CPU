#include "implicit_conv2d.h"

template <typename Tin, typename Tw, typename Tacc, typename Tout>
__global__ void implicit_conv2d_kernel(int input_n, int input_c, int input_h, int input_w,
                                       int output_c, int kernel_h, int kernel_w,
                                       int output_h, int output_w,
                                       int stride_h, int stride_w,
                                       int pad_h, int pad_w,
                                       int dilation_h, int dilation_w,
                                       int group_count,
                                       const Tin *x, const Tw *w, const Tacc *bias, Tout *y)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= input_n * output_h * output_w || iy >= output_c) return;

    int oc = iy;
    int n = ix / (output_h * output_w);
    int oh = ix % (output_h * output_w) / output_w;
    int ow = ix % (output_h * output_w) % output_w;

    Tacc result = static_cast<Tacc>(0);
    int k = input_c * kernel_h * kernel_w;
    for (int i = 0; i < k; ++i)
    {
        // calc index
        int ic = i / (kernel_h * kernel_w);
        int kh = i % (kernel_h * kernel_w) / kernel_w;
        int kw = i % (kernel_h * kernel_w) % kernel_w;
        
        int ih = oh * stride_h + kh - pad_h;
        int iw = ow * stride_w + kw - pad_w;
        if (ih < 0 || ih >= input_h || iw < 0 || iw >= input_w) continue;

        int x_idx = ((n * input_c + ic) * input_h + ih) * input_w + iw;
        int w_idx = ((oc * input_c + ic) * kernel_h + kh) * kernel_w + kw;
        result += x[x_idx] * w[w_idx];
    }
    if (bias)
    {
        result += bias[oc];
    }
    int y_idx = ((n * output_c + oc) * output_h + oh) * output_w + ow;
    y[y_idx] = result;
}

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void implicit_conv2d_gpu(int input_n, int input_c, int input_h, int input_w,
                         int output_c, int kernel_h, int kernel_w,
                         int stride_h, int stride_w,
                         int pad_h, int pad_w,
                         int dilation_h, int dilation_w,
                         int group_count,
                         const Tin *x, const Tw *w, const Tacc *bias, Tout *y)
{
    int khd = (kernel_h - 1) * dilation_h + 1;
    int kwd = (kernel_w - 1) * dilation_w + 1;
    int output_h = (input_h - khd + 2 * pad_h) / stride_h + 1;
    int output_w = (input_w - kwd + 2 * pad_w) / stride_w + 1;

    int m = output_c;
    int n = input_n * output_h * output_w;

    dim3 dimBlock(32, 32);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);
    implicit_conv2d_kernel<Tin, Tw, Tacc, Tout><<<dimGrid, dimBlock>>>(input_n, input_c, input_h, input_w,
                                                                       output_c, kernel_h, kernel_w,
                                                                       output_h, output_w,
                                                                       stride_h, stride_w,
                                                                       pad_h, pad_w,
                                                                       dilation_h, dilation_w,
                                                                       group_count,
                                                                       x, w, bias, y);
}

template void implicit_conv2d_gpu<float, float, float, float>(int input_n, int input_c, int input_h, int input_w,
                                                              int output_c, int kernel_h, int kernel_w,
                                                              int stride_h, int stride_w,
                                                              int pad_h, int pad_w,
                                                              int dilation_h, int dilation_w,
                                                              int group_count,
                                                              const float *x, const float *w, const float *bias, float *y);