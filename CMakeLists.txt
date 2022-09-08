cmake_minimum_required(VERSION 3.10)
project(how_to_optimize_conv_in_cpu)

set(srcs src/Main.cpp
         src/im2col_gemm_conv.cpp
         src/naive_conv.cpp)
add_executable(${PROJECT_NAME} ${srcs})