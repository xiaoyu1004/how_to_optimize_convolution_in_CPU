#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <chrono>
#include <string>

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

std::string get_convolution_fwd_str(ConvolutionFwdAlgo_t algo)
{
    if (algo == CONVOLUTION_FWD_ALGO_DIRECT)
    {
        return "CONVOLUTION_FWD_ALGO_DIRECT";
    }
    else if (algo == CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
    {
        return "CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM";
    }
    else
    {
        return "CONVOLUTION_FWD_ALGO_UNKNOW";
    }
}

#endif