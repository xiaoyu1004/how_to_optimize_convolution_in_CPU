#include <cuda.h>

template <typename Tin, typename Tw, typename Tacc, typename Tout>
__global__ void Conv2DNaiveKernel(int input_n, int input_c, int input_h, int input_w,
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

    int output_size = input_n * output_c * output_h * output_w;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;

    int n = tid / (output_c * output_h * output_w);
    int oc = tid % (output_c * output_h * output_w) / (output_h * output_w);
    int oh = tid % (output_c * output_h * output_w) % (output_h * output_w) / output_w;
    int ow = tid % (output_c * output_h * output_w) % (output_h * output_w) % output_w;

    Tacc result = static_cast<Tacc>(0.0f);
    for (int ic = 0; ic < input_c; ++ic)
    {
        for (int kh = 0; kh < kernel_h; ++kh)
        {
            int ih = oh * stride_h + kh - pad_h;
            if (ih < 0 || ih >= input_h) continue;
            for (int kw = 0; kw < kernel_w; ++kw)
            {
                int iw = ow * stride_w + kw - pad_w;
                if (iw < 0 || iw >= input_w) continue;

                int input_idx = n * input_c * input_h * input_w + 
                                ic * input_h * input_w + 
                                ih * input_w + 
                                iw;
                int kernel_idx = oc * input_c * kernel_h * kernel_w + 
                                 ic * kernel_h * kernel_w + 
                                 kh * kernel_w + 
                                 kw;
                result += x[input_idx] * w[kernel_idx];
            }
        }
    }
    if (bias)
    {
        result += bias[oc];
    }
    int output_idx = n * output_c * output_h * output_w + 
                     oc * output_h * output_w + 
                     oh * output_w + 
                     ow;
    y[output_idx] = static_cast<Tout>(result);
}

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void naive_conv_gpu(int input_n, int input_c, int input_h, int input_w,
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

    int output_size = input_n * output_c * output_h * output_w;
    dim3 dimBlock(1024);
    dim3 dimGrid((output_size + dimBlock.x - 1) / dimBlock.x);
    Conv2DNaiveKernel<Tin, Tw, Tacc, Tout><<<dimGrid, dimBlock>>>(input_n, input_c, input_h, input_w,
                                                                  output_c, kernel_h, kernel_w,
                                                                  stride_h, stride_w,
                                                                  pad_h, pad_w,
                                                                  dilation_h, dilation_w,
                                                                  group_count,
                                                                  x, w, bias, y);
}

template void naive_conv_gpu<float, float, float, float>(int input_n, int input_c, int input_h, int input_w,
                                                         int output_c, int kernel_h, int kernel_w,
                                                         int stride_h, int stride_w,
                                                         int pad_h, int pad_w,
                                                         int dilation_h, int dilation_w,
                                                         int group_cnt,
                                                         const float *x, const float *w, const float *bias, float *y);