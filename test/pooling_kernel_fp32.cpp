#include "common.h"

#include <iostream>
#include <vector>
#include <random>
#include <ctime>

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
    const T *X,
    T *Y)
{
    const int X_HxW = X_H * X_W;
    const int Y_HxW = Y_H * Y_W;
    const int nc = blockIdx.x / Y_H;
    const int yh = blockIdx.x % Y_H;
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
        const T scale = T(1) /
                        static_cast<T>(count_include_pad ? kernel_h * kernel_w
                                                         : (b - t) * (r - l));
        T sum = 0;
        for (int i = t; i < b; ++i)
        {
            for (int j = l; j < r; ++j)
            {
#if __CUDA_ARCH__ >= 350
                sum += __ldg(X_ptr + i * X_W + j);
#else
                sum += X_ptr[i * X_W + j];
#endif
            }
        }
        Y_ptr[yh * Y_W + yw] = sum * scale;
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
    std::normal_distribution<float> dist;

    int input_size = input_n * input_c * input_h * input_w;
    Tin *h_x = new Tin[input_size]{};
    for (int i = 0; i < input_size; ++i)
    {
        if (std::is_same<Tin, half>::value)
        {
            float tmp = dist(e);
            h_x[i] = __float2half(tmp);
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

#ifdef ENABLE_ILUVATAR
    ReadDataFromFile("../../data/pooling_dnn_fp32.bin", reinterpret_cast<char *>(h_ref_y), output_size * sizeof(Tin));
#endif

    timer t;
    int warm_cnt = 3;
    int loop_cnt = 10;
    float avg_t = 0;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 7.Start pooling calculation
#define CUDNN_POOLING_FWD                        \
    {                                            \
        float alpha = 1.f;                       \
        float beta = 0.f;                        \
        const int num_blocks = N * C * output_h; \
        constexpr int CUDA_NUM_THREADS = 128;    \
        AveragePool2DForwardNCHWCUDAKernel<Tin>  \
            <<<num_blocks, CUDA_NUM_THREADS>>>(  \
                input_h,                         \
                input_w,                         \
                output_h,                        \
                output_w,                        \
                kernel_h,                        \
                kernel_w,                        \
                stride_h,                        \
                stride_w,                        \
                pad_h,                           \
                pad_w,                           \
                true,                            \
                d_x,                             \
                d_y);                            \
    }

    // warm
    for (int i = 0; i < warm_cnt; ++i)
    {
        CUDNN_POOLING_FWD;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < loop_cnt; ++i)
    {
        CUDNN_POOLING_FWD;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&avg_t, start, stop));
    avg_t /= loop_cnt;
    std::cout << "device: GPU: " << avg_t << " ms" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_y, d_y, output_size * sizeof(Tin), cudaMemcpyDeviceToHost));

#ifdef ENABLE_NVIDIA
    WriteDataToFile("../../data/pooling_dnn_fp32.bin", reinterpret_cast<char *>(h_y), output_size * sizeof(Tin));
#endif

#ifdef ENABLE_LOG
#ifdef ENABLE_ILUVATAR
    std::cout << "cpu:" << std::endl;
    for (int i = 0; i < output_h; ++i)
    {
        for (int j = 0; j < input_c; ++j)
        {
            for (int k = 0; k < output_w; ++k)
            {
                std::cout << static_cast<Tin *>(h_ref_y)[j * output_h * output_w + i * output_w + k] << "\t";
            }
            std::cout << "\t\t";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    std::cout << "gpu(cudnn):" << std::endl;
    for (int i = 0; i < output_h; ++i)
    {
        for (int j = 0; j < input_c; ++j)
        {
            for (int k = 0; k < output_w; ++k)
            {
                std::cout << static_cast<Tin *>(h_y)[j * output_h * output_w + i * output_w + k] << "\t";
            }
            std::cout << "\t\t";
        }
        std::cout << std::endl;
    }
#endif // ENABLE_ILUVATAR
#endif // ENABLE_LOG

#ifdef ENABLE_ILUVATAR
    for (int i = 0; i < output_size; ++i)
    {
        Tin diff1 = std::abs(h_ref_y[i] - h_y[i]);
        if (diff1 > 1e-1f)
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
    std::vector<std::vector<int>> test_cases = {
        // n  c  h  w  kh  kw  sh  sw  ph  pw
        // {1, 1, 4, 4, 1, 3, 3, 1, 1, 0, 0, 1, 1, 1},
        // {6, 3, 6, 6, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1},
        {4, 32, 64, 64, 3, 3, 1, 1, 0, 0},

        // {1, 2048, 128, 256, 23, 46, 21, 42, 0, 0},

        // {128, 32, 416, 672, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1},
        // {32, 64, 320, 320, 3, 3, 1, 1, 1, 1}
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