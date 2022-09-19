#ifndef IM2COL_GEMM_CONV_H
#define IM2COL_GEMM_CONV_H

template <typename Tin>
void im2col(int input_n, int input_c, int input_h, int input_w,
            int output_c, int kernel_h, int kernel_w,
            int stride_h, int stride_w,
            int pad_h, int pad_w,
            int dilation_h, int dilation_w,
            int group_count,
            const void *x, void *y);

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void sgemm(int M, int N, int K, const void *a, const void *b, const void *bias, void *c);

template <typename Tin, typename Tw, typename Tacc, typename Tout>
void implict_precomp_sgemm(int input_n, int input_c, int input_h, int input_w,
                           int output_c, int kernel_h, int kernel_w,
                           int stride_h, int stride_w,
                           int pad_h, int pad_w,
                           int dilation_h, int dilation_w,
                           int group_count,
                           const void *x, const void *w, const void *bias, void *y);

#endif