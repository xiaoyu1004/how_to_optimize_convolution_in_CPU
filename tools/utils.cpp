#include "common.h"
#include "utils.h"

void WriteDataToFile(char *ptr, size_t memSize)
{
    std::ofstream fs("convolution_fp16.bin", std::ios::out | std::ios::binary);
    fs.write(ptr, memSize);
    fs.close();
}

void ReadDataFromFile(char *ptr, size_t memSize)
{
    std::ifstream fs("convolution_fp16.bin", std::ios::in | std::ios::binary);
    if (!fs.is_open())
    {
        std::cout << "Error! unable open data file!" << std::endl;
        std::terminate();
    }
    fs.read(ptr, memSize);
    fs.close();
}