#include "common.h"
#include "conv2d.h"

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
    std::normal_distribution<float> dist(-1.f, 1.f);

    int input_size = input_n * input_c * input_h * input_w;
    int weight_size = output_c * input_c * kernel_h * kernel_w;
    float *x = new float[input_size];
    for (int i = 0; i < input_size; ++i)
    {
        x[i] = dist(e);
        // x[i] = Tin(i + 1);
    }

    float *w = new float[weight_size];
    for (int i = 0; i < weight_size; ++i)
    {
        w[i] = dist(e);
        // w[i] = Tw(i + 1);
    }

    float *bias = new float[output_c];
    for (int i = 0; i < output_c; ++i)
    {
        // bias[i] = dist(e);
        bias[i] = (Tacc)0;
    }

    int khd = (kernel_h - 1) * dilation_h + 1;
    int kwd = (kernel_w - 1) * dilation_w + 1;
    int output_h = (input_h - khd + 2 * pad_h) / stride_h + 1;
    int output_w = (input_w - kwd + 2 * pad_w) / stride_w + 1;
    int output_size = input_n * output_c * output_h * output_w;
    float *y = new float[output_size];
    for (int i = 0; i < output_size; ++i)
    {
        y[i] = (Tout)0;
    }

    timer t;
    int warm_cnt = 5;
    int loop_cnt = 20;
    double avg_t = 0;

    // warm
    for (int i = 0; i < warm_cnt; ++i)
    {
        conv2d<Tin, Tw, Tacc, Tout>(input_n, input_c, input_h, input_w,
                                    output_c, kernel_h, kernel_w,
                                    stride_h, stride_w,
                                    pad_h, pad_w,
                                    dilation_h, dilation_w,
                                    group_count,
                                    algo,
                                    x, w, bias, y);
    }

    for (int i = 0; i < loop_cnt; ++i)
    {
        t.start();
        conv2d<Tin, Tw, Tacc, Tout>(input_n, input_c, input_h, input_w,
                                    output_c, kernel_h, kernel_w,
                                    stride_h, stride_w,
                                    pad_h, pad_w,
                                    dilation_h, dilation_w,
                                    group_count,
                                    algo,
                                    x, w, bias, y);
        t.stop();
        avg_t += t.get_elapsed_milli_seconds();
    }
    avg_t /= loop_cnt;
    std::cout << get_convolution_fwd_str(algo) << ":\t" << avg_t << " ms" << std::endl;

    // for (int i = 0; i < output_h; ++i)
    // {
    //     for (int j = 0; j < output_c; ++j)
    //     {
    //         for (int k = 0; k < output_w; ++k)
    //         {
    //             std::cout << static_cast<Tout *>(y)[j * output_h * output_w + i * output_w + k] << "\t";
    //         }
    //         std::cout << "\t\t";
    //     }
    //     std::cout << std::endl;
    // }

    delete[] x;
    delete[] w;
    delete[] bias;
    delete[] y;
}

int main()
{
    // int a = 3000;
    // int b = (int)(char)a;
    
    std::vector<std::vector<int>> test_cases = {
        // n c h w cout kh kw sh sw ph pw dh dw g
        // {1, 2, 3, 3, 2, 2, 2, 1, 1, 0, 0, 1, 1, 1},
        // {1, 3, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1},
        {1, 32, 640, 640, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1}};
    std::vector<ConvolutionFwdAlgo_t> algos = {CONVOLUTION_FWD_ALGO_DIRECT, CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM};

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