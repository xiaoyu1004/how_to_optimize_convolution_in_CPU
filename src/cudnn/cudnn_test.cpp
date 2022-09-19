#include <cuda_runtime.h>
#include <cudnn.h>

#include <iostream>
#include <sstream>

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define CUDNN_CHECK(status) do {                                       \
    cudnnStatus_t err = (status);                                      \
    std::stringstream _error;                                          \
    if (err != CUDNN_STATUS_SUCCESS) {                                 \
      _error << "CUDNN failure: " << cudnnGetErrorString(err);         \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

int main()
{
    // define size
    int input_n = 1;
    int input_c = 3;
    int input_h = 256;
    int input_w = 256;

    int output_c = 2;
    int kernel_h = 3;
    int kernel_w = 3;

    int stride_h = 1;
    int stride_w = 1;
    int pad_h = 1;
    int pad_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    int group = 1;

    int input_num = input_n * input_c * input_h * input_w;
    int input_bytes = input_num * sizeof(float);

    int filter_num = output_c * input_c * kernel_h * kernel_w;
    int filter_bytes = filter_num * sizeof(float);

    // calculate output size
    int khd = (kernel_h - 1) * dilation_h + 1;
    int kwd = (kernel_w - 1) * dilation_w + 1;
    int output_h = (input_h - khd + 2 * pad_h) / stride_h + 1;
    int output_w = (input_w - kwd + 2 * pad_w) / stride_w + 1;

    int output_num = input_n * output_c * output_h * output_w;
    int output_bytes = output_num * sizeof(float);

    // malloc host memory
    float *h_input_ptr = new float[input_num]{0};
    for (int i = 0; i < input_num; ++i)
    {
        h_input_ptr[i] = (i % 5) * 0.5f;
    }

    float *h_filter_ptr = new float[filter_num]{0};
    for (int i = 0; i < filter_num; ++i)
    {
        h_filter_ptr[i] = (i % 5) * 0.5f;
    }

    float *h_output_ptr = new float[output_num]{0};

    // 1.init cudnn handle
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    // 2.tensor descriptor
    cudnnTensorDescriptor_t input_descriptor;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w));

    cudnnTensorDescriptor_t output_descriptor;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, output_c, output_h, output_w));

    cudnnFilterDescriptor_t filter_descriptor;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_descriptor));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, output_c, input_c, kernel_h, kernel_w));

    // 3.描述操作并设置相关参数
    cudnnConvolutionDescriptor_t conv_descriptor;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_descriptor, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CONVOLUTION , CUDNN_DATA_FLOAT));

    // 4.选择算法
    cudnnConvolutionFwdAlgoPerf_t algo;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(handle, input_descriptor, filter_descriptor, conv_descriptor, output_descriptor, 1, 0, &algo));

    // 5.申请工作空间
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle, input_descriptor, filter_descriptor, conv_descriptor, output_descriptor, algo.algo, &workspace_size));
    void *d_workspace = nullptr;
    cudaMalloc(&d_workspace, workspace_size);

    // 6.将需要计算的数据传输到gpu
    // malloc device memory
    float *d_input_ptr;
    float *d_filter_ptr;
    float *d_output_ptr;

    cudaMalloc((void**)&d_input_ptr, input_bytes);
    cudaMemcpy(d_input_ptr, h_input_ptr, input_bytes, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_filter_ptr, filter_bytes);
    cudaMemcpy(d_filter_ptr, h_filter_ptr, filter_bytes, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_output_ptr, output_bytes);

    // 7.开始计算
    float alpha = 1.f;
    float beta = 0.f;
    CUDNN_CHECK(cudnnConvolutionForward(handle,
                                        &alpha,
                                        input_descriptor,
                                        d_input_ptr,
                                        filter_descriptor,
                                        d_filter_ptr,
                                        conv_descriptor,
                                        algo.algo,
                                        d_workspace,
                                        workspace_size,
                                        &beta,
                                        output_descriptor,
                                        d_output_ptr));

    // 8.将计算结果传回cpu内存
    cudaMemcpy(h_output_ptr, d_output_ptr, output_bytes, cudaMemcpyDeviceToHost);

    // free memory
    delete []h_input_ptr;
    delete []h_filter_ptr;
    delete []h_output_ptr;

    cudaFree(d_input_ptr);
    cudaFree(d_filter_ptr);
    cudaFree(d_output_ptr);
    cudaFree(d_workspace);

    std::cout << "call cudnnConvolutionForward finish!" << std::endl;

    return 0;
}