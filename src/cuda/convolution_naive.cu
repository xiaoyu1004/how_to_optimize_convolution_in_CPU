#include <cuda.h>

__global__ void Conv2DNaiveKernel(int input_n, int input_c, int input_h, int input_w, const float *input_ptr,
                                  int output_c, int kernel_h, int kernel_w, const float *weight_ptr,
                                  int stride_h, int stride_w,
                                  int pad_h, int pad_w,
                                  int group,
                                  float *output_ptr)
{

}

void Conv2DGPU(int input_n, int input_c, int input_h, int input_w, const float *input_ptr,
               int output_c, int kernel_h, int kernel_w, const float *weight_ptr,
               int stride_h, int stride_w,
               int pad_h, int pad_w,
               int group,
               float *output_ptr)
{

}