#include "common.h"
#include "utils.h"

#include <iostream>
#include <vector>
#include <random>
#include <ctime>

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
#ifdef ENABLE_CUDA
    Tin *d_x;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), input_size * sizeof(Tin)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, input_size * sizeof(Tin), cudaMemcpyHostToDevice));
#endif

    int output_h = (input_h - kernel_h + 2 * pad_h) / stride_h + 1;
    int output_w = (input_w - kernel_w + 2 * pad_w) / stride_w + 1;

    int output_size = input_n * input_c * output_h * output_w;
    Tin *h_ref_y = new Tin[output_size]{};
#ifdef ENABLE_CUDA
    Tin *h_y = new Tin[output_size];
    Tin *d_y;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y), output_size * sizeof(Tin)));
    CUDA_CHECK(cudaMemset(d_y, 0, output_size * sizeof(Tin)));
#endif

#ifdef ENABLE_ILUVATAR
    ReadDataFromFile(reinterpret_cast<char *>(h_ref_y), output_size * sizeof(Tin));
#endif

#ifdef ENABLE_CUDNN
    // 1.init cudnn handle
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    // 2.tensor descriptor
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w));

    // 3.Describes operations and sets related parameters
    cudnnPoolingMode_t mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    cudnnPoolingDescriptor_t pooling_desc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pooling_desc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(pooling_desc, mode, CUDNN_NOT_PROPAGATE_NAN, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));

    cudnnTensorDescriptor_t output_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, output_h, output_w));

    // 7.Start pooling calculation
    float alpha = 1.f;
    float beta = 0.f;
    CUDNN_CHECK(cudnnPoolingForward(handle, pooling_desc, &alpha, input_desc, d_x, &beta, output_desc, d_y));
    CUDA_CHECK(cudaMemcpy(h_y, d_y, output_size * sizeof(Tin), cudaMemcpyDeviceToHost));

#ifdef ENABLE_NVIDIA
    WriteDataToFile(reinterpret_cast<char *>(h_y), output_size * sizeof(Tin));
#endif
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
        // n c h w oc kh kw sh sw ph pw dh dw g
        // {1, 1, 4, 4, 1, 3, 3, 1, 1, 0, 0, 1, 1, 1},
        // {6, 3, 6, 6, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1},
        // {4, 32, 64, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1},
        // {128, 32, 416, 672, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1},
        {128, 32, 320, 320, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1}};

    using Tin = half;
    // using Tin = float;

    for (int i = 0; i < test_cases.size(); ++i)
    {
        TestConv<Tin>(test_cases[i][0], test_cases[i][1], test_cases[i][2], test_cases[i][3],
                      test_cases[i][4], test_cases[i][5], test_cases[i][6],
                      test_cases[i][7], test_cases[i][8],
                      test_cases[i][9], test_cases[i][10],
                      test_cases[i][11], test_cases[i][12],
                      test_cases[i][13]);
    }
}