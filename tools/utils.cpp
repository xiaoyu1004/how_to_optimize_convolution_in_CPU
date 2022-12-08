#include "common.h"
#include "utils.h"

void WriteDataToFile(std::string filepath, char *ptr, size_t memSize)
{
    std::ofstream fs(filepath, std::ios::out | std::ios::binary);
    fs.write(ptr, memSize);
    fs.close();
    std::cout << "write data to file finish!" << std::endl;
}

void ReadDataFromFile(std::string filepath, char *ptr, size_t memSize)
{
    std::ifstream fs(filepath, std::ios::in | std::ios::binary);
    if (!fs.is_open())
    {
        std::cout << "Error! unable open data file!" << std::endl;
        std::terminate();
    }
    fs.read(ptr, memSize);
    fs.close();
}