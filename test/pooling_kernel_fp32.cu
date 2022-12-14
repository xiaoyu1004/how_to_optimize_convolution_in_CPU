#include "common.h"
#include "utils.h"

#include <iostream>
#include <vector>
#include <random>
#include <ctime>

#define ENABLE_COMPARE 1
static constexpr int kNbThreadsPerBlockReduceAllDim = 1024;

typedef unsigned v4u32 __attribute__((ext_vector_type(4)));

template <typename value_t>
__host__ __device__ value_t clz(value_t x)
{
    for (int i = 31; i >= 0; --i)
    {
        if ((1 << i) & x)
            return 31 - i;
    }
    return 32;
}

__host__ __device__ inline int find_log2(int x)
{
    int a = 31 - clz(x);
    a += (x & (x - 1)) != 0; // Round up, add 1 if not a power of 2.
    return a;
}

/**
 * Find divisor, using find_log2
 */
__host__ __device__ inline void find_divisor(int &mul, int &shr, int denom)
{
    if (denom == 1)
    {
        mul = 0;
        shr = 0;
    }
    else
    {
        int p = 31 + find_log2(denom);
        int m = ((1ull << p) + denom - 1) / denom;

        mul = m;
        shr = p - 32;
    }
}

/**
 * Find quotient and remainder using device-side intrinsics
 */
__device__ inline void fast_divmod(int &quo, int &rem, int src, int div,
                                   unsigned int mul, unsigned int shr)
{
    quo = __umulhi(src, mul) >> shr;
    quo = (div == 1) ? src : quo;
    rem = src - (quo * div);
}

template <typename T>
__device__ void ReductionSum(int tid, T *sdata, int len)
{
    auto pow2 = len;
    if (pow2 & (pow2 - 1))
    {
        while (pow2 & (pow2 - 1))
        {
            pow2 &= (pow2 - 1);
        }
        if (tid >= pow2)
        {
            sdata[tid - pow2] = sdata[tid - pow2] + sdata[tid];
        }
        __syncthreads();
    }

    if (pow2 == 4096)
    {
        if (tid < 2048)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 2048];
        }
        __syncthreads();
    }

    if (pow2 >= 2048)
    {
        if (tid < 1024)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 1024];
        }
        __syncthreads();
    }

    if (pow2 >= 1024)
    {
        if (tid < 512)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 512];
        }
        __syncthreads();
    }

    if (pow2 >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 256];
        }
        __syncthreads();
    }

    if (pow2 >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 128];
        }
        __syncthreads();
    }

#ifndef __BI__
    if (pow2 >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] = sdata[tid] + sdata[tid + 64];
        }
        __syncthreads();
    }
#endif

#if __BI__
    if (tid < 64)
    {
        volatile T *vsdata = sdata;
        if (pow2 >= 128 && tid < 64)
        {
            vsdata[tid] = vsdata[tid] + vsdata[tid + 64];
        }
#else
    if (tid < 32)
    {
        volatile T *vsdata = sdata;
#endif
        if (pow2 >= 64 && tid < 32)
        {
            vsdata[tid] = vsdata[tid] + vsdata[tid + 32];
        }

        if (pow2 >= 32 && tid < 16)
        {
            vsdata[tid] = vsdata[tid] + vsdata[tid + 16];
        }

        if (pow2 >= 16 && tid < 8)
        {
            vsdata[tid] = vsdata[tid] + vsdata[tid + 8];
        }

        if (pow2 >= 8 && tid < 4)
        {
            vsdata[tid] = vsdata[tid] + vsdata[tid + 4];
        }

        if (pow2 >= 4 && tid < 2)
        {
            vsdata[tid] = vsdata[tid] + vsdata[tid + 2];
        }

        if (pow2 >= 2 && tid < 1)
        {
            vsdata[tid] = vsdata[tid] + vsdata[tid + 1];
        }
    }
}

template <typename T>
__global__ void AveragePool2DForwardNCHWCUDAKernel(
    const int X_H,
    const int X_W,
    const int Y_H,
    const int Y_W,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    const bool count_include_pad,
    int mul,
    int shr,
    const T *X,
    T *Y)
{
    const int X_HxW = X_H * X_W;
    const int Y_HxW = Y_H * Y_W;

    int nc, yh;
    fast_divmod(nc, yh, blockIdx.x, Y_H, mul, shr);

    const T *X_ptr = X + nc * X_HxW;
    T *Y_ptr = Y + nc * Y_HxW;
    const int xh = yh * stride_h - pad_t;
    const int t = max(xh, 0);
    const int b = min(xh + kernel_h, X_H);
    for (int yw = threadIdx.x; yw < Y_W; yw += blockDim.x)
    {
        const int xw = yw * stride_w - pad_l;
        const int l = max(xw, 0);
        const int r = min(xw + kernel_w, X_W);
        const T scale = T(1) / static_cast<T>(count_include_pad ? kernel_h * kernel_w : (b - t) * (r - l));
        T sum = 0;
        for (int i = t; i < b; ++i)
        {
            for (int j = l; j < r; ++j)
            {
                sum += X_ptr[i * X_W + j];
            }
        }
        Y_ptr[yh * Y_W + yw] = sum * scale;
    }
}

template <typename T>
__global__ void GlobalAveragePool2DForwardNCHWCUDAKernel(
    const int X_HxW,
    const T *X,
    T *Y)
{
    const T *X_ptr = X + blockIdx.x * X_HxW;

    __shared__ float sdata[kNbThreadsPerBlockReduceAllDim];
    sdata[threadIdx.x] = 0.0f; // important
    __syncthreads();

    int spatial_idx = threadIdx.x;
    const int tid = threadIdx.x;
    while (spatial_idx < X_HxW)
    {
        // sdata[tid] += UpCast<T, float>(X_ptr[spatial_idx]);
        sdata[tid] += X_ptr[spatial_idx];
        spatial_idx += blockDim.x;
    }
    const T scale = T(1) / static_cast<T>(X_HxW);
    __syncthreads();
    ReductionSum<float>(threadIdx.x, sdata, blockDim.x);
    __syncthreads();

    if (threadIdx.x == 0)
    {
        // Y[blockIdx.x] = DownCast<float, T>(sdata[0] * scale);
        Y[blockIdx.x] = sdata[0] * scale;
    }
}

template <typename Tin>
void TestPooling(int input_n, int input_c, int input_h, int input_w,
                 int kernel_h, int kernel_w,
                 int stride_h, int stride_w,
                 int pad_h, int pad_w)
{
    // std::default_random_engine e(static_cast<unsigned>(time(NULL)));
    std::default_random_engine e(static_cast<unsigned>(1000));
    std::uniform_real_distribution<float> dist(-9, 14);

    int input_size = input_n * input_c * input_h * input_w;
    Tin *h_x = new Tin[input_size]{};
    for (int i = 0; i < input_size; ++i)
    {
        if (std::is_same<Tin, half>::value)
        {
            h_x[i] = __float2half(dist(e));
        }
        else
        {
            h_x[i] = static_cast<Tin>(dist(e));
        }
        // h_x[i] = static_cast<Tin>(i);
    }

    Tin *d_x;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), input_size * sizeof(Tin)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, input_size * sizeof(Tin), cudaMemcpyHostToDevice));

    int output_h = (input_h - kernel_h + 2 * pad_h) / stride_h + 1;
    int output_w = (input_w - kernel_w + 2 * pad_w) / stride_w + 1;

    int output_size = input_n * input_c * output_h * output_w;
    Tin *h_ref_y = new Tin[output_size]{};

    Tin *h_y = new Tin[output_size];
    Tin *d_y;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y), output_size * sizeof(Tin)));
    CUDA_CHECK(cudaMemset(d_y, 0, output_size * sizeof(Tin)));

#if ENABLE_COMPARE
#ifdef ENABLE_ILUVATAR
    ReadDataFromFile("../../data/pooling_dnn_fp32.bin", reinterpret_cast<char *>(h_ref_y), output_size * sizeof(Tin));
#endif
#endif

    timer t;
    int warm_cnt = 3;
    int loop_cnt = 10;
    float avg_t = 0;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float alpha = 1.f;
    float beta = 0.f;
    const int num_blocks = input_n * input_c * output_h;
    constexpr int CUDA_NUM_THREADS = 128;

    int mul, shr;
    find_divisor(mul, shr, output_h);

// 7.Start pooling calculation
#define CUDA_AVG_POOLING_FWD                                                                                                                                                                             \
    {                                                                                                                                                                                                    \
        const int num_blocks = input_n * input_c * output_h;                                                                                                                                             \
        AveragePool2DForwardNCHWCUDAKernel<Tin><<<num_blocks, CUDA_NUM_THREADS>>>(input_h, input_w, output_h, output_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, true, mul, shr, d_x, d_y); \
    }

#define CUDA_GLOBAL_AVG_POOLING_FWD                                                                                     \
    {                                                                                                                   \
        const int num_blocks = input_n * input_c;                                                                       \
        const int X_HxW = input_h * input_w;                                                                            \
        GlobalAveragePool2DForwardNCHWCUDAKernel<Tin><<<num_blocks, kNbThreadsPerBlockReduceAllDim>>>(X_HxW, d_x, d_y); \
    }

    // warm
    for (int i = 0; i < warm_cnt; ++i)
    {
        // CUDA_AVG_POOLING_FWD;
        CUDA_GLOBAL_AVG_POOLING_FWD
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < loop_cnt; ++i)
    {
        // CUDA_AVG_POOLING_FWD;
        CUDA_GLOBAL_AVG_POOLING_FWD
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&avg_t, start, stop));
    avg_t /= loop_cnt; // std::cout << "input_n\tinput_c\tinput_h\tinput_w\tkernel_h\tkernel_w\tstride_h\tstride_w\tpad_h\tpad_w" << std::endl;
    std::cout << input_n << "\t"
              << input_c << "\t"
              << input_h << "\t"
              << kernel_h << "\t"
              << kernel_w << "\t"
              << stride_h << "\t"
              << stride_w << "\t"
              << pad_h << "\t"
              << pad_w << "\t"
              << avg_t
              << " ms" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_y, d_y, output_size * sizeof(Tin), cudaMemcpyDeviceToHost));

#ifdef ENABLE_NVIDIA
    WriteDataToFile("../../data/pooling_dnn_fp32.bin", reinterpret_cast<char *>(h_y), output_size * sizeof(Tin));
#endif

#if ENABLE_COMPARE
#ifdef ENABLE_ILUVATAR
    for (int i = 0; i < output_size; ++i)
    {
        Tin diff1 = std::abs(h_ref_y[i] - h_y[i]);
        if (diff1 > 1e-3f)
        {
            std::cout << "ERROR: h_ref_y[" << i << "] = " << h_ref_y[i]
                      << " vs h_y[" << i << "] = " << h_y[i]
                      << "\tdiff1: " << diff1 << std::endl;
            std::cout << "compare failed!" << std::endl;
            std::terminate();
        }
    }
    std::cout << "compare pass!" << std::endl;
#endif // ENABLE_ILUVATAR
#endif // ENABLE_COMPARE

    delete[] h_x;
    delete[] h_ref_y;

#ifdef ENABLE_CUDA
    delete[] h_y;
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
#endif // ENABLE_CUDA
}

int main()
{
    std::cout << "input_n\tinput_c\tinput_h\tinput_w\tkernel_h\tkernel_w\tstride_h\tstride_w\tpad_h\tpad_w" << std::endl;
    std::vector<std::vector<int>> test_cases = {
        // n  c  h  w  kh  kw  sh  sw  ph  pw

        // {4, 32, 64, 64, 3, 3, 1, 1, 0, 0},

        // {1, 2048, 128, 256, 23, 46, 21, 42, 0, 0},

        // {32, 64, 320, 320, 3, 3, 1, 1, 1, 1}

        // {32, 64, 320, 320, 3, 3, 1, 1, 1, 1},
        // {32, 16, 320, 320, 3, 3, 1, 1, 1, 1},
        // {32, 64, 128, 128, 3, 3, 1, 1, 1, 1},
        // {16, 64, 256, 256, 3, 3, 1, 1, 1, 1},
        // {64, 64, 64, 64, 3, 3, 1, 1, 1, 1},

        // {32, 64, 320, 320, 6, 6, 1, 1, 1, 1},
        // {32, 16, 320, 320, 6, 6, 1, 1, 1, 1},
        // {32, 64, 128, 128, 6, 6, 1, 1, 1, 1},
        // {16, 64, 256, 256, 6, 6, 1, 1, 1, 1},
        // {64, 64, 64, 64, 6, 6, 1, 1, 1, 1},

        ///////////////////////////////////////////////////

        {32, 64, 320, 320, 320, 320, 1, 1, 0, 0},
        // {32, 16, 320, 320, 320, 320, 1, 1, 1, 1},
        // {32, 64, 128, 128, 128, 128, 1, 1, 1, 1},
        // {16, 64, 256, 256, 256, 256, 1, 1, 1, 1},
        // {64, 64, 64, 64, 64, 64, 1, 1, 1, 1},
    };

    // using Tin = half;
    using Tin = float;

    for (int i = 0; i < test_cases.size(); ++i)
    {
        TestPooling<Tin>(test_cases[i][0], test_cases[i][1], test_cases[i][2], test_cases[i][3],
                         test_cases[i][4], test_cases[i][5],
                         test_cases[i][6], test_cases[i][7],
                         test_cases[i][8], test_cases[i][9]);
    }
}