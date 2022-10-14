#include "im2col_conv2d.h"
#include "im2col.h"
#include "gemm.h"

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void im2col_conv2d_gpu(int input_n, int input_c, int input_h, int input_w,
                       int output_c, int kernel_h, int kernel_w,
                       int stride_h, int stride_w,
                       int pad_h, int pad_w,
                       int dilation_h, int dilation_w,
                       int group_count,
                       const void *x, const void *w, const void *bias, void *y)
{

}