#ifndef IM2COL_GEMM_CONV_H
#define IM2COL_GEMM_CONV_H

void im2col(unsigned input_n, unsigned input_c, unsigned input_h, unsigned input_w,
            unsigned output_c, unsigned kernel_h, unsigned kernel_w,
            unsigned stride_h, unsigned stride_w,
            unsigned pad_h, unsigned pad_w,
            unsigned dialation_h, unsigned dialation_w,
            unsigned group_count,
            const float *x, float *y);

void sgemm(unsigned m, unsigned n, unsigned k, const float *a, const float *b, float c);

void implict_precomp_sgemm(unsigned input_n, unsigned input_c, unsigned input_h, unsigned input_w,
                           unsigned output_c, unsigned kernel_h, unsigned kernel_w,
                           unsigned stride_h, unsigned stride_w,
                           unsigned pad_h, unsigned pad_w,
                           unsigned dialation_h, unsigned dialation_w,
                           unsigned group_count,
                           const float *x, const float *w, const float *bias, float *y);

#endif