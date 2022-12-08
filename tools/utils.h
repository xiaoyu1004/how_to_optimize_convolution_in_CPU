#ifndef UTILS_H
#define UTILS_H

#include "common.h"

#include <fstream>

void WriteDataToFile(std::string filepath, char *ptr, size_t memSize);

void ReadDataFromFile(std::string filepath, char *ptr, size_t memSize);

#endif