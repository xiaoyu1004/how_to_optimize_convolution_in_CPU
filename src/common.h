#ifndef COMMON_H
#define COMMON_H

#ifdef ENABLE_CUDA
#include <cuda.h>
#endif

#include <cstdint>
#include <chrono>
#include <string>

#ifdef ENABLE_CUDA
#define CUDA_CHECK(func)                                                           \
    {                                                                              \
        cudaError_t e = (func);                                                    \
        if (e != cudaSuccess)                                                      \
            printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }
#endif

class timer
{
public:
    timer() : start_(), end_()
    {
    }

    void start()
    {
        start_ = std::chrono::system_clock::now();
    }

    void stop()
    {
        end_ = std::chrono::system_clock::now();
    }

    double get_elapsed_seconds() const
    {
        return (double)std::chrono::duration_cast<std::chrono::seconds>(end_ - start_).count();
    }

    double get_elapsed_milli_seconds() const
    {
        return (double)std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count();
    }

    double get_elapsed_micro_seconds() const
    {
        return (double)std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count();
    }

    double get_elapsed_nano_seconds() const
    {
        return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end_ - start_).count();
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start_;
    std::chrono::time_point<std::chrono::system_clock> end_;
};

typedef enum
{
    CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
    CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CONVOLUTION_FWD_ALGO_GEMM = 2,
    CONVOLUTION_FWD_ALGO_DIRECT = 3,
    CONVOLUTION_FWD_ALGO_FFT = 4,
    CONVOLUTION_FWD_ALGO_FFT_TILING = 5,
    CONVOLUTION_FWD_ALGO_WINOGRAD = 6,
    CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7,
    CONVOLUTION_FWD_ALGO_COUNT = 8,
    CONVOLUTION_FWD_ALGO_UNKNOW
} ConvolutionFwdAlgo_t;

inline std::string get_convolution_fwd_str(ConvolutionFwdAlgo_t algo)
{
    if (algo == CONVOLUTION_FWD_ALGO_DIRECT)
    {
        return "CONVOLUTION_FWD_ALGO_DIRECT";
    }
    else if (algo == CONVOLUTION_FWD_ALGO_GEMM)
    {
        return "CONVOLUTION_FWD_ALGO_GEMM";
    }
    else if (algo == CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
    {
        return "CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM";
    }
    else
    {
        return "CONVOLUTION_FWD_ALGO_UNKNOW";
    }
}

#endif