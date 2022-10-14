#include "common.h"
#include "conv2d.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <ctime>

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void TestConv(int input_n, int input_c, int input_h, int input_w,
              int output_c, int kernel_h, int kernel_w,
              int stride_h, int stride_w,
              int pad_h, int pad_w,
              int dilation_h, int dilation_w,
              int group_count,
              ConvolutionFwdAlgo_t algo)
{
    std::default_random_engine e(static_cast<unsigned>(time(NULL)));
    std::normal_distribution<Tin> dist(-1, 1);

    int input_size = input_n * input_c * input_h * input_w;
    int weight_size = output_c * input_c * kernel_h * kernel_w;
    Tin *h_x = new Tin[input_size];
    for (int i = 0; i < input_size; ++i)
    {
        // x[i] = dist(e);
        h_x[i] = Tin(i % 10);
    }
#ifdef ENABLE_CUDA
    Tin *d_x;
    CUDA_CHECK(cudaMalloc(&d_x, input_size * sizeof(Tin)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, input_size * sizeof(Tin), cudaMemcpyHostToDevice));
#endif

    Tw *h_w = new Tw[weight_size];
    for (int i = 0; i < weight_size; ++i)
    {
        // w[i] = dist(e);
        h_w[i] = Tw(i % 10);
    }
#ifdef ENABLE_CUDA
    Tw *d_w;
    CUDA_CHECK(cudaMalloc(&d_w, weight_size * sizeof(Tw)));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, weight_size * sizeof(Tw), cudaMemcpyHostToDevice));
#endif

    Tacc *h_bias = new Tacc[output_c];
    for (int i = 0; i < output_c; ++i)
    {
        // bias[i] = dist(e);
        h_bias[i] = (Tacc)0;
    }
#ifdef ENABLE_CUDA
    Tacc *d_bias;
    CUDA_CHECK(cudaMalloc(&d_bias, output_c * sizeof(Tacc)));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, output_c * sizeof(Tacc), cudaMemcpyHostToDevice));
#endif

    int khd = (kernel_h - 1) * dilation_h + 1;
    int kwd = (kernel_w - 1) * dilation_w + 1;
    int output_h = (input_h - khd + 2 * pad_h) / stride_h + 1;
    int output_w = (input_w - kwd + 2 * pad_w) / stride_w + 1;
    int output_size = input_n * output_c * output_h * output_w;
    Tout *h_ref_y = new Tout[output_size];
    for (int i = 0; i < output_size; ++i)
    {
        h_ref_y[i] = (Tout)0;
    }
#ifdef ENABLE_CUDA
    Tout *h_y = new Tout[output_size];
    Tout *d_y;
    CUDA_CHECK(cudaMalloc(&d_y, output_size * sizeof(Tout)));
#endif

    timer t;
    int warm_cnt = 5;
    int loop_cnt = 20;
    double avg_t = 0;

    // warm
    for (int i = 0; i < warm_cnt; ++i)
    {
        Conv2dCPU<Tin, Tw, Tacc, Tout>(input_n, input_c, input_h, input_w,
                                       output_c, kernel_h, kernel_w,
                                       stride_h, stride_w,
                                       pad_h, pad_w,
                                       dilation_h, dilation_w,
                                       group_count,
                                       algo,
                                       h_x, h_w, h_bias, h_ref_y);
    }

    for (int i = 0; i < loop_cnt; ++i)
    {
        t.start();
        Conv2dCPU<Tin, Tw, Tacc, Tout>(input_n, input_c, input_h, input_w,
                                       output_c, kernel_h, kernel_w,
                                       stride_h, stride_w,
                                       pad_h, pad_w,
                                       dilation_h, dilation_w,
                                       group_count,
                                       algo,
                                       h_x, h_w, h_bias, h_ref_y);
        t.stop();
        avg_t += t.get_elapsed_milli_seconds();
    }
    avg_t /= loop_cnt;
    std::cout << "device: CPU  algo: " << get_convolution_fwd_str(algo) << "\ttime(ms)" << avg_t << " ms" << std::endl;

#ifdef ENABLE_CUDA
    avg_t = 0;
    // warm
    for (int i = 0; i < warm_cnt; ++i)
    {
        Conv2dGPU<Tin, Tw, Tacc, Tout>(input_n, input_c, input_h, input_w,
                                       output_c, kernel_h, kernel_w,
                                       stride_h, stride_w,
                                       pad_h, pad_w,
                                       dilation_h, dilation_w,
                                       group_count,
                                       algo,
                                       d_x, d_w, d_bias, d_y);
    }

    for (int i = 0; i < loop_cnt; ++i)
    {
        t.start();
        Conv2dGPU<Tin, Tw, Tacc, Tout>(input_n, input_c, input_h, input_w,
                                       output_c, kernel_h, kernel_w,
                                       stride_h, stride_w,
                                       pad_h, pad_w,
                                       dilation_h, dilation_w,
                                       group_count,
                                       algo,
                                       d_x, d_w, d_bias, d_y);
        t.stop();
        avg_t += t.get_elapsed_milli_seconds();
    }
    avg_t /= loop_cnt;
    std::cout << "device: GPU  algo: " << get_convolution_fwd_str(algo) << "\ttime(ms)" << avg_t << " ms" << std::endl;
    CUDA_CHECK(cudaMemcpy(h_y, d_y, output_size * sizeof(Tout), cudaMemcpyDeviceToHost));
#endif

#ifdef ENABLE_LOG
    std::cout << "cpu:" << std::endl;
    for (int i = 0; i < output_h; ++i)
    {
        for (int j = 0; j < output_c; ++j)
        {
            for (int k = 0; k < output_w; ++k)
            {
                std::cout << static_cast<Tout *>(h_ref_y)[j * output_h * output_w + i * output_w + k] << "\t";
            }
            std::cout << "\t\t";
        }
        std::cout << std::endl;
    }

#ifdef ENABLE_CUDA
    std::cout << "gpu:" << std::endl;
    for (int i = 0; i < output_h; ++i)
    {
        for (int j = 0; j < output_c; ++j)
        {
            for (int k = 0; k < output_w; ++k)
            {
                std::cout << static_cast<Tout *>(h_y)[j * output_h * output_w + i * output_w + k] << "\t";
            }
            std::cout << "\t\t";
        }
        std::cout << std::endl;
    }
#endif
#endif

#ifdef ENABLE_CUDA
    for (int i = 0; i < output_size; ++i)
    {
        if (std::abs(h_y[i] - h_ref_y[i]) > 1e-4f)
        {
            std::cout << "ERROR: h_y[" << i << "] = " << h_y[i] << " vs h_ref_y[" << i << "] = " << h_ref_y[i] << std::endl;
            std::cout << "compare failed!" << std::endl;
        }
    }
    std::cout << "compare pass!" << std::endl;
#endif

    delete[] h_x;
    delete[] h_w;
    delete[] h_bias;
    delete[] h_ref_y;

#ifdef ENABLE_CUDA
    delete[] h_y;
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_y));
#endif
}

int main()
{
    std::vector<std::vector<int>> test_cases = {
        // n c h w oc kh kw sh sw ph pw dh dw g
        // {1, 3, 8, 8, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1},
        // {1, 3, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1},
        // {1, 2, 3, 3, 2, 2, 2, 1, 1, 0, 0, 1, 1, 1},
        // {1, 3, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1},
        {1, 16, 64, 64, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1}
    };
    std::vector<ConvolutionFwdAlgo_t> algos = {CONVOLUTION_FWD_ALGO_DIRECT, CONVOLUTION_FWD_ALGO_GEMM, CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM};
    // std::vector<ConvolutionFwdAlgo_t> algos = {CONVOLUTION_FWD_ALGO_GEMM};
    // std::vector<ConvolutionFwdAlgo_t> algos = {CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM};

    using Tin = float;
    using Tw = float;
    using Tacc = float;
    using Tout = float;

    for (int i = 0; i < test_cases.size(); ++i)
    {
        for (ConvolutionFwdAlgo_t algo : algos)
        {
            TestConv<Tin, Tw, Tacc, Tout>(test_cases[i][0], test_cases[i][1], test_cases[i][2], test_cases[i][3],
                                          test_cases[i][4], test_cases[i][5], test_cases[i][6],
                                          test_cases[i][7], test_cases[i][8],
                                          test_cases[i][9], test_cases[i][10],
                                          test_cases[i][11], test_cases[i][12],
                                          test_cases[i][13],
                                          algo);
        }
    }
}