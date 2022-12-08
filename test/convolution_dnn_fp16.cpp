#include "common.h"
#include "conv2d.hpp"
#include "utils.h"

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
              int group_count)
{
    // std::default_random_engine e(static_cast<unsigned>(time(NULL)));
    std::default_random_engine e(static_cast<unsigned>(1000));
    std::normal_distribution<float> dist;

    int input_size = input_n * input_c * input_h * input_w;
    int weight_size = output_c * input_c * kernel_h * kernel_w;
    Tin *h_x = new Tin[input_size]{};
    for (int i = 0; i < input_size; ++i)
    {
        if (std::is_same<Tout, half>::value)
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
#ifdef ENABLE_CUDA
    Tin *d_x;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), input_size * sizeof(Tin)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, input_size * sizeof(Tin), cudaMemcpyHostToDevice));
#endif

    Tw *h_w = new Tw[weight_size];
    for (int i = 0; i < weight_size; ++i)
    {
        if (std::is_same<Tout, half>::value)
        {
            h_w[i] = __float2half(dist(e));
        }
        else
        {
            h_w[i] = static_cast<Tw>(dist(e));
        }
        // h_w[i] = static_cast<Tw>(i);
    }
#ifdef ENABLE_CUDA
    Tw *d_w;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_w), weight_size * sizeof(Tw)));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, weight_size * sizeof(Tw), cudaMemcpyHostToDevice));
#endif

    int khd = (kernel_h - 1) * dilation_h + 1;
    int kwd = (kernel_w - 1) * dilation_w + 1;
    int output_h = (input_h - khd + 2 * pad_h) / stride_h + 1;
    int output_w = (input_w - kwd + 2 * pad_w) / stride_w + 1;
    int output_size = input_n * output_c * output_h * output_w;
    Tout *h_ref_y = new Tout[output_size]{};
#ifdef ENABLE_CUDA
    Tout *h_y = new Tout[output_size];
    Tout *d_y;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y), output_size * sizeof(Tout)));
    CUDA_CHECK(cudaMemset(d_y, 0, output_size * sizeof(Tout)));
#endif

#ifdef ENABLE_ILUVATAR
    ReadDataFromFile(reinterpret_cast<char *>(h_ref_y), output_size * sizeof(Tout));
#endif

#ifdef ENABLE_CUDNN
    // 1.init cudnn handle
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    // 2.tensor descriptor
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, input_n, input_c, input_h, input_w));

    cudnnFilterDescriptor_t filter_desc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, output_c, input_c, kernel_h, kernel_w));

    // 3.Describes operations and sets related parameters
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    cudnnTensorDescriptor_t output_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, input_n, output_c, output_h, output_w));

    // 4.Selection algorithm
    cudnnConvolutionFwdAlgo_t cudnnAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    // CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(handle, input_desc, filter_desc, conv_desc, output_desc, 1, 0, &cudnnAlgo));

    // 5.Applying for a Workspace
    size_t dnn_workspace_size = 0;
    // CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc, output_desc, cudnnAlgo.algo, &dnn_workspace_size));
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc, output_desc, cudnnAlgo, &dnn_workspace_size));
    void *d_workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_workspace, dnn_workspace_size));

    // 6.Transfer the data that needs to be computed to the gpu

    // 7.Start convolution calculation
    float alpha = 1.f;
    float beta = 0.f;
    CUDNN_CHECK(cudnnConvolutionForward(handle,
                                        &alpha,
                                        input_desc,
                                        d_x,
                                        filter_desc,
                                        d_w,
                                        conv_desc,
                                        cudnnAlgo,
                                        d_workspace,
                                        dnn_workspace_size,
                                        &beta,
                                        output_desc,
                                        d_y));
    CUDA_CHECK(cudaMemcpy(h_y, d_y, output_size * sizeof(Tout), cudaMemcpyDeviceToHost));

#ifdef ENABLE_NVIDIA
    WriteDataToFile(reinterpret_cast<char *>(h_y), output_size * sizeof(Tout));
#endif
#endif

#ifdef ENABLE_LOG
#ifdef ENABLE_ILUVATAR
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


    std::cout << std::endl;

    std::cout << "gpu(cudnn):" << std::endl;
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
#endif // ENABLE_ILUVATAR
#endif // ENABLE_LOG

#ifdef ENABLE_ILUVATAR
    for (int i = 0; i < output_size; ++i)
    {
        Tout diff1 = std::abs(h_ref_y[i] - h_y[i]);
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
    delete[] h_w;
    delete[] h_ref_y;

#ifdef ENABLE_CUDA
    delete[] h_y;
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_y));

#ifdef ENABLE_CUDNN
    CUDA_CHECK(cudaFree(d_workspace));
#endif // ENABLE_CUDNN
#endif // ENABLE_CUDA
}

int main()
{
    std::vector<std::vector<int>> test_cases = {
        // n c h w oc kh kw sh sw ph pw dh dw g
        // {1, 1, 4, 4, 1, 3, 3, 1, 1, 0, 0, 1, 1, 1},
        // {6, 3, 6, 6, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1},
        // {1, 2, 3, 3, 2, 2, 2, 1, 1, 0, 0, 1, 1, 1},
        // {1, 3, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1},
        // {4, 32, 64, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1},
        // {128, 32, 416, 672, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1},
        // {120, 32, 416, 672, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1},
        // {121, 32, 416, 672, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1}
        {128, 32, 320, 320, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1},
        // {128, 32, 64, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1}
    };

    using Tin = half;
    using Tw = half;
    using Tacc = float;
    using Tout = half;

    // using Tin = float;
    // using Tw = float;
    // using Tacc = float;
    // using Tout = float;

    for (int i = 0; i < test_cases.size(); ++i)
    {
        TestConv<Tin, Tw, Tacc, Tout>(test_cases[i][0], test_cases[i][1], test_cases[i][2], test_cases[i][3],
                                      test_cases[i][4], test_cases[i][5], test_cases[i][6],
                                      test_cases[i][7], test_cases[i][8],
                                      test_cases[i][9], test_cases[i][10],
                                      test_cases[i][11], test_cases[i][12],
                                      test_cases[i][13]);
    }
}