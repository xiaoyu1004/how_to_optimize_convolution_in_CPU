#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>

#define CUDA_CHECK(func)                                                           \
    {                                                                              \
        cudaError_t e = (func);                                                    \
        if (e != cudaSuccess)                                                      \
            printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }

constexpr int MALLOC_ALIGN = 64;

void reformatPack4Host(unsigned size, char *src_ptr, char *dst_ptr)
{
    unsigned align_size = (unsigned)((size + MALLOC_ALIGN - 1) & -MALLOC_ALIGN);
    for (unsigned i = 0; i < align_size; ++i)
    {
        int block_index = i / 64;
        int ht = i % 64 / 4;
        int wt = i % 64 % 4;

        int src_index = block_index * 64 + wt * 16 + ht;
        dst_ptr[i] = src_ptr[src_index];
    }
}

__global__ void reformatPack4_naive(unsigned size, const char *src_ptr,
                                    unsigned align_size, char *dst_ptr)
{
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned tid = threadIdx.x;
    extern __shared__ char SLB[];

    ((unsigned *)SLB)[tid] = ((unsigned *)src_ptr)[gid];

    char v[4];
    v[0] = (SLB + tid / 16 * 64)[tid % 16];
    v[1] = (SLB + tid / 16 * 64)[tid % 16 + 16];
    v[2] = (SLB + tid / 16 * 64)[tid % 16 + 32];
    v[3] = (SLB + tid / 16 * 64)[tid % 16 + 48];

    // write
    ((unsigned *)dst_ptr)[gid] = *(unsigned *)v;
}

__device__ void byte_transform(int &a, unsigned lane_id)
{
    unsigned part_ex = lane_id >> 4;
    unsigned tail_ex = lane_id & 15;
    unsigned part_in = tail_ex >> 2;
    unsigned tail_in = tail_ex & 3;

    unsigned lane_0 = (part_ex << 4) + part_in + 0;
    unsigned lane_1 = (part_ex << 4) + part_in + 4;
    unsigned lane_2 = (part_ex << 4) + part_in + 8;
    unsigned lane_3 = (part_ex << 4) + part_in + 12;

    int a_0 = __shfl_sync(0xffffffff, a, lane_0);
    int a_1 = __shfl_sync(0xffffffff, a, lane_1);
    int a_2 = __shfl_sync(0xffffffff, a, lane_2);
    int a_3 = __shfl_sync(0xffffffff, a, lane_3);

    int hi;
    int lo;

    int pos = tail_in & 1;

    if (tail_in < 2)
    {
        hi = 0x5410;
    }
    else
    {
        hi = 0x7632;
    }

    if (pos == 0)
    {
        lo = 0x6420;
    }
    else
    {
        lo = 0x7531;
    }

    int tmp_0;
    int tmp_1;

    asm volatile("prmt.b32 %0, %1, %2, %3;"
                 : "=r"(tmp_0)
                 : "r"(a_0), "r"(a_1), "r"(hi));
    asm volatile("prmt.b32 %0, %1, %2, %3;"
                 : "=r"(tmp_1)
                 : "r"(a_2), "r"(a_3), "r"(hi));
    asm volatile("prmt.b32 %0, %1, %2, %3;"
                 : "=r"(a)
                 : "r"(tmp_0), "r"(tmp_1), "r"(lo));
}

__global__ void byte_t(const char *val, char *dst)
{
    unsigned thread_id = threadIdx.x;
    unsigned laneId = thread_id & 63;
    unsigned data_id = blockIdx.x * blockDim.x + thread_id;

    const int *p_val = reinterpret_cast<const int *>(val);
    int *p_dst = reinterpret_cast<int *>(dst);
    int a = p_val[data_id];
    byte_transform(a, laneId);
    p_dst[data_id] = a;
}

void cuinferReformatPack4(unsigned size, const char *src_ptr, char *dst_ptr,
                          cudaStream_t stream = 0)
{
    unsigned align_size = (unsigned)((size + MALLOC_ALIGN - 1) & -MALLOC_ALIGN);
    dim3 dimBlock(1024, 1, 1);
    dim3 dimGrid((align_size + 4095) / 4096, 1, 1);

    // byte_t<<<dimGrid, dimBlock, 0, stream>>>(src_ptr, dst_ptr);
    reformatPack4_naive<<<dimGrid, dimBlock, 1024 * sizeof(unsigned), stream>>>(size, src_ptr, align_size, dst_ptr);
}

void TestReformatPack4(unsigned input_n, unsigned input_h, unsigned input_w,
                       unsigned input_c)
{
    unsigned size = input_n * input_h * input_w * input_c;
    unsigned align_size = (unsigned)((size + MALLOC_ALIGN - 1) & -MALLOC_ALIGN);

    srand(1000);

    // malloc input
    char *d_src_ptr;
    CUDA_CHECK(cudaMalloc((void **)&d_src_ptr, size));
    char *h_src_ptr = new char[size];
    for (unsigned i = 0; i < size; i++)
    {
        // h_src_ptr[i] = rand() % 256;
        h_src_ptr[i] = i % 256;
        // printf("h_src_ptr[i]: %d\n", h_src_ptr[i]);
    }
    CUDA_CHECK(cudaMemcpy(d_src_ptr, h_src_ptr, size, cudaMemcpyHostToDevice));

    // malloc output
    char *d_dst_ptr;
    CUDA_CHECK(cudaMalloc((void **)&d_dst_ptr, align_size));
    CUDA_CHECK(cudaMemset(d_dst_ptr, 0, align_size));
    char *h_dst_ptr = new char[align_size]{0};
    char *h_dst_ref_ptr = new char[align_size]{0};

    // host
    reformatPack4Host(size, h_src_ptr, h_dst_ptr);

    // device
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    unsigned nIter = 20;
    unsigned warmCnt = 5;

    // warm up
    for (unsigned i = 0; i < warmCnt; ++i)
    {
        cuinferReformatPack4(size, d_src_ptr, d_dst_ptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // convert layout
    CUDA_CHECK(cudaEventRecord(start));
    for (unsigned i = 0; i < nIter; ++i)
    {
        cuinferReformatPack4(size, d_src_ptr, d_dst_ptr);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float msecTotal = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop));
    double msecPer = (double)msecTotal / nIter;
    double bandWidth = (double)(size + align_size) / (msecPer * 1000 * 1000);

    std::cout << size << "\t" << align_size << "\t" << msecPer << "\t"
              << bandWidth << "\t" << std::endl;

    CUDA_CHECK(
        cudaMemcpy(h_dst_ref_ptr, d_dst_ptr, align_size, cudaMemcpyDeviceToHost));

    // compare convert layout
    for (unsigned i = 0; i < align_size; ++i)
    {
        int err = std::abs(h_dst_ptr[i] - h_dst_ref_ptr[i]);
        if (err > 0) {
          std::cout << "convert FAIL! h_dst_ptr[" << i
                    << "]=" << static_cast<unsigned>(h_dst_ptr[i])
                    << " vs h_dst_ref_ptr[" << i
                    << "]=" << static_cast<unsigned>(h_dst_ref_ptr[i]) <<
                    std::endl;
          std::terminate();
        }
        // std::cout << "h_dst_ptr[" << i << "]=" <<
        // static_cast<int>(h_dst_ptr[i])
        //               << "\tvs\th_dst_ref_ptr[" << i << "]=" <<
        //               static_cast<int>(h_dst_ref_ptr[i])
        //               << std::endl;
    }

    delete[] h_src_ptr;
    delete[] h_dst_ptr;
    delete[] h_dst_ref_ptr;
    CUDA_CHECK(cudaFree(d_src_ptr));
    CUDA_CHECK(cudaFree(d_dst_ptr));
}

int main(int argc, const char **argv)
{
    std::cout << "size\talign_Size\tmsecPer\tbandWidth(GB/s)" << std::endl;

    // TestReformatPack4(1, 4, 4, 64);

    // TestReformatPack4(32, 64, 64, 256);
    // TestReformatPack4(32, 216, 216, 64);
    // TestReformatPack4(32, 216, 216, 64);
    TestReformatPack4(32, 216, 216, 64);
    TestReformatPack4(32, 216, 216, 64);

    std::cout << std::endl;
    return 0;
}