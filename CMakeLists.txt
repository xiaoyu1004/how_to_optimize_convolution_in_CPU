cmake_minimum_required(VERSION 3.10)
project(how_to_optimize_conv LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /NODEFAULTLIB:library")

option(ENABLE_CUDA "enable cuda" ON)
option(ENABLE_CUDNN "enable cuda" ON)
option(ENABLE_LOG "enable cuda" ON)

if(ENABLE_CUDA)
    add_definitions(-DENABLE_CUDA)
endif()

if(ENABLE_CUDNN)
    add_definitions(-DENABLE_CUDNN)
endif()

if(ENABLE_LOG)
    add_definitions(-DENABLE_LOG)
endif()

set(srcs src/Main.cpp
         src/gemm.cpp
         src/im2col.cpp
         src/im2col_conv2d.cpp
         src/implicit_conv2d.cpp
         src/naive_conv.cpp)

if(ENABLE_CUDA)
    list(APPEND srcs src/cuda/gemm.cu
                     src/cuda/im2col_conv2d.cu
                     src/cuda/im2col.cu
                     src/cuda/naive_conv.cu)
endif()
add_executable(${PROJECT_NAME} ${srcs})
target_include_directories(${PROJECT_NAME} PRIVATE src)
target_link_libraries(${PROJECT_NAME} PRIVATE cudnn)