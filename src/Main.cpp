#include "common.h"
#include "conv2d.h"

#include <iostream>
#include <vector>

void TestConv(unsigned input_n, unsigned input_c, unsigned input_h, unsigned input_w,
              unsigned output_c, unsigned kernel_h, unsigned kernel_w,
              unsigned stride_h, unsigned stride_w,
              unsigned pad_h, unsigned pad_w,
              unsigned dialation_h, unsigned dialation_w,
              unsigned group_count,
              ConvolutionFwdAlgo_t algo)
{
    unsigned input_size = input_n * input_c * input_h * input_w;
    unsigned weight_size = output_c * input_c * kernel_h * kernel_w;
    float *x = new float[input_size];
    for (unsigned i = 0; i < input_size; ++i)
    {
        x[i] = 1.f;
    }

    float *w = new float[weight_size];
    for (unsigned i = 0; i < input_size; ++i)
    {
        w[i] = 1.f;
    }

    float *bias = new float[output_c];
    for (unsigned i = 0; i < input_size; ++i)
    {
        bias[i] = 0.f;
    }

    unsigned khd = (kernel_h - 1) * dialation_h + 1;
    unsigned kwd = (kernel_w - 1) * dialation_w + 1;
    unsigned output_h = (input_h - khd + 2 * pad_h) / stride_h + 1;
    unsigned output_w = (input_w - kwd + 2 * pad_w) / stride_w + 1;
    unsigned output_size = output_c * input_c * output_h * output_w;
    float *y = new float[output_size];
    for (unsigned i = 0; i < input_size; ++i)
    {
        y[i] = -1.f;
    }

    conv2d(input_n, input_c, input_h, input_w,
           output_c, kernel_h, kernel_w,
           stride_h, stride_w,
           pad_h, pad_w,
           dialation_h, dialation_w,
           group_count,
           algo,
           x, w, bias, y);

    free(x);
    free(w);
    free(bias);
    free(y);
}

int main()
{
    std::vector<std::vector<unsigned>> test_cases = {
        {1, 3, 4, 4, 1, 3, 3, 1, 1, 0, 0, 1, 1, 1}};
    std::vector<ConvolutionFwdAlgo_t> algos = {CONVOLUTION_FWD_ALGO_DIRECT};

    for (int i = 0; i < test_cases.size(); ++i)
    {
        for (ConvolutionFwdAlgo_t algo : algos)
        {
            TestConv(test_cases[i][0], test_cases[i][1], test_cases[i][2], test_cases[i][3],
                     test_cases[i][4], test_cases[i][5], test_cases[i][6],
                     test_cases[i][7], test_cases[i][8],
                     test_cases[i][9], test_cases[i][10],
                     test_cases[i][11], test_cases[i][12],
                     test_cases[i][13],
                     algo);
        }
    }
}