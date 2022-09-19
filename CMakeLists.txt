cmake_minimum_required(VERSION 3.10)
project(how_to_optimize_conv_in_cpu LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)

set(srcs src/Main.cpp
         src/im2col_gemm_conv.cpp
         src/naive_conv.cpp)
add_executable(${PROJECT_NAME} ${srcs})

add_executable(cudnn_test src/cudnn/cudnn_test.cpp)
target_link_libraries(cudnn_test PRIVATE cudart cudnn)
# 设置MT/MTd
set_property(TARGET cudnn_test PROPERTY
  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")