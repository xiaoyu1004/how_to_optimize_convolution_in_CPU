cmake_minimum_required(VERSION 3.10)
project(how_to_optimize_conv LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)

option(ENABLE_CUDA "enable cuda" ON)
option(ENABLE_LOG "enable cuda" ON)
if(ENABLE_CUDA)
    add_definitions(-DENABLE_CUDA)
endif()
if(ENABLE_LOG)
    add_definitions(-DENABLE_LOG)
endif()

set(srcs src/Main.cpp
         src/im2col_gemm.cpp
         src/implicit_gemm.cpp
         src/naive_conv.cpp)
add_executable(${PROJECT_NAME} ${srcs})

# add_executable(shfl_test src/cuda/shfl_test.cu)
# add_executable(pack4 src/cuda/pack4.cu)

# add_executable(cudnn_test src/cudnn/cudnn_test.cpp)
# target_link_libraries(cudnn_test PRIVATE cudart cudnn)
# 设置MT/MTd
# set_property(TARGET cudnn_test PROPERTY
#   MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")