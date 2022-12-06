set(MODULE_NAME convolution_fp16)
set(srcs ${PROJECT_SOURCE_DIR}/test/convolution_fp16.cpp
         ${PROJECT_SOURCE_DIR}/src/gemm.cpp
         ${PROJECT_SOURCE_DIR}/src/im2col.cpp
         ${PROJECT_SOURCE_DIR}/src/im2col_conv2d.cpp
         ${PROJECT_SOURCE_DIR}/src/implicit_conv2d.cpp
         ${PROJECT_SOURCE_DIR}/src/naive_conv.cpp
         ${PROJECT_SOURCE_DIR}/src/common.cpp)

if (ENABLE_NVIDIA)
    add_executable(${MODULE_NAME} ${srcs})
    target_include_directories(${MODULE_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src /usr/local/cuda/include)
endif()

if (ENABLE_ILUVATAR)
    cuda_add_executable(${MODULE_NAME} ${srcs})
    target_include_directories(${MODULE_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src ${CUDA_TOOLKIT_ROOT_DIR}/include)
endif()

target_link_libraries(${MODULE_NAME} cudnn cudart)