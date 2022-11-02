#include "implicit_conv2d.h"

template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K,
          typename Tin, typename Tw, typename Tacc, typename Tout>
__global__ void implicit_conv2d_kernel_v1(int input_n, int input_c, int input_h, int input_w,
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
    if (ix >= input_n * output_h * output_w || iy >= output_c)
        return;

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
        if (ih < 0 || ih >= input_h || iw < 0 || iw >= input_w)
            continue;

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

template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K,
          typename Tin, typename Tw, typename Tacc, typename Tout>
__global__ void implicit_conv2d_kernel_v2(int input_n, int input_c, int input_h, int input_w,
                                          int output_c, int kernel_h, int kernel_w,
                                          int output_h, int output_w,
                                          int stride_h, int stride_w,
                                          int pad_h, int pad_w,
                                          int dilation_h, int dilation_w,
                                          int group_count,
                                          const Tin *x, const Tw *w, const Tacc *bias, Tout *y)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int M = output_c;
    int N = input_n * output_h * output_w;
    int K = input_c * kernel_h * kernel_w;

    __shared__ Tw SLB_W[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ Tin SLB_X[BLOCK_SIZE_K][BLOCK_SIZE_N];
    Tacc sum = static_cast<Tacc>(0);

    int n = (bx * BLOCK_SIZE_N + tx) / (output_h * output_w);
    int x_rest_offset = (bx * BLOCK_SIZE_N + tx) % (output_h * output_w);
    int oc = by * BLOCK_SIZE_M + ty;
    int oh = x_rest_offset / output_w;
    int ow = x_rest_offset % output_w;

    for (int idx = 0; idx < K; idx += BLOCK_SIZE_K)
    {
        // load kernel from gmem to smem
        Tw val_w = static_cast<Tw>(0);
        if ((by * BLOCK_SIZE_M + ty) < M && (idx + tx) < K)
        {
            int filter_rest_offset = (idx + tx) % (kernel_h * kernel_w);
            int ic = (idx + tx) / (kernel_h * kernel_w);
            int w_idx = (oc * input_c + ic) * kernel_h * kernel_w + filter_rest_offset;
            val_w = w[w_idx];
        }
        SLB_W[ty][tx] = val_w;

        // load x from gmem to smem
        Tin val_x = static_cast<Tin>(0);
        if ((idx + ty) < K && (bx * BLOCK_SIZE_N + tx) < N)
        {
            int k_rest_offset = (idx + ty) % (kernel_h * kernel_w);
            int kh = k_rest_offset / kernel_w;
            int kw = k_rest_offset % kernel_w;

            int ic = (idx + ty) / (kernel_h * kernel_w);
            int ih = oh * stride_h + kh - pad_h;
            int iw = ow * stride_w + kw - pad_w;
            if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w)
            {
                int x_idx = ((n * input_c + ic) * input_h + ih) * input_w + iw;
                val_x = x[x_idx];
            }
        }
        SLB_X[ty][tx] = val_x;

        __syncthreads();

        // compute
        for (int p = 0; p < BLOCK_SIZE_K; ++p)
        {
            sum += SLB_W[ty][p] * SLB_X[p][tx];
        }
        __syncthreads();
    }

    // store to c
    if ((by * BLOCK_SIZE_M + ty) < M && (bx * BLOCK_SIZE_N + tx) < N)
    {
        if (bias)
        {
            sum += bias[by * BLOCK_SIZE_M + ty];
        }
        int y_idx = ((n * output_c + oc) * output_h + oh) * output_w + ow;
        y[y_idx] = sum;
    }
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

    constexpr int BLOCK_SIZE_M = 16;
    constexpr int BLOCK_SIZE_N = 16;
    constexpr int BLOCK_SIZE_K = 16;

    dim3 dimBlock(BLOCK_SIZE_N, BLOCK_SIZE_M);
    dim3 dimGrid((n + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (m + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    implicit_conv2d_kernel_v2<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, Tin, Tw, Tacc, Tout><<<dimGrid, dimBlock>>>(input_n, input_c, input_h, input_w,
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