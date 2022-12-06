cmake_minimum_required(VERSION 3.10)
project(how_to_optimize_conv LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 70)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /NODEFAULTLIB:library")

option(ENABLE_CPU "enable cuda" ON)
option(ENABLE_CUDA "enable cuda" ON)
option(ENABLE_CUDNN "enable cuda" ON)
option(ENABLE_LOG "enable cuda" ON)

option(ENABLE_NVIDIA "enable nvidia" ON)
option(ENABLE_ILUVATAR "enable iluvatar" OFF)

if(ENABLE_CPU)
    add_definitions(-DENABLE_CPU)
endif()

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
         src/naive_conv.cpp
         src/common.cpp)

if(ENABLE_CUDA)
    list(APPEND srcs src/cuda/gemm.cu
                     src/cuda/im2col.cu
                     src/cuda/im2col_conv2d.cu
                     src/cuda/implicit_conv2d.cu
                     src/cuda/naive_conv.cu)
endif()

if (ENABLE_NVIDIA)
    add_executable(${PROJECT_NAME} ${srcs})
    target_include_directories(${PROJECT_NAME} PRIVATE src /usr/local/cuda/include)
endif()

if (ENABLE_ILUVATAR)
    include(${PROJECT_SOURCE_DIR}/cmake/CUDA.cmake)
    set(ROOT_PATH "/usr/local/corex/" CACHE PATH "cuda installation root dir")
    set(CUDA_TOOLKIT_ROOT_DIR ${ROOT_PATH} CACHE PATH "root path to cuda/lib64 cuda/bin etc")

    set(COMPILER_PATH ${ROOT_PATH}/bin CACHE PATH "compiler binary path")
    set(CMAKE_CXX_COMPILER ${COMPILER_PATH}/clang++ CACHE PATH "CXX compiler binary path")
    set(CMAKE_C_COMPILER ${COMPILER_PATH}/clang CACHE PATH "C compiler binary path")
    set(CUDA_ARCH "ivcore11" CACHE STRING "CUDA architecture.")

    cuda_add_executable(${PROJECT_NAME} ${srcs})
    target_include_directories(${PROJECT_NAME} PRIVATE src /usr/local/corex/include)
endif()

target_link_libraries(${PROJECT_NAME} cudnn cudart)

add_subdirectory(test)