set(TEST_CASE_1 convolution_fp16)
set(srcs ${PROJECT_SOURCE_DIR}/test/convolution_dnn_fp16.cpp
         ${PROJECT_SOURCE_DIR}/src/gemm.cpp
         ${PROJECT_SOURCE_DIR}/src/im2col.cpp
         ${PROJECT_SOURCE_DIR}/src/im2col_conv2d.cpp
         ${PROJECT_SOURCE_DIR}/src/implicit_conv2d.cpp
         ${PROJECT_SOURCE_DIR}/src/naive_conv.cpp
         ${PROJECT_SOURCE_DIR}/src/common.cpp
         ${PROJECT_SOURCE_DIR}/tools/utils.cpp)

if (ENABLE_NVIDIA)
    add_executable(${TEST_CASE_1} ${srcs})
    target_include_directories(${TEST_CASE_1} PRIVATE /usr/local/cuda/include)
endif()

if (ENABLE_ILUVATAR)
    cuda_add_executable(${TEST_CASE_1} ${srcs})
    target_include_directories(${TEST_CASE_1} PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include)
endif()

target_include_directories(${TEST_CASE_1} PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tools)
target_link_libraries(${TEST_CASE_1} cudnn cudart)

################################################################################

set(TEST_CASE_2 pooling_fp32)
set(srcs ${PROJECT_SOURCE_DIR}/test/pooling_dnn_fp32.cpp
         ${PROJECT_SOURCE_DIR}/src/common.cpp
         ${PROJECT_SOURCE_DIR}/tools/utils.cpp)

if (ENABLE_NVIDIA)
    add_executable(${TEST_CASE_2} ${srcs})
    target_include_directories(${TEST_CASE_2} PRIVATE /usr/local/cuda/include)
endif()

if (ENABLE_ILUVATAR)
    cuda_add_executable(${TEST_CASE_2} ${srcs})
    target_include_directories(${TEST_CASE_2} PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include)
endif()

target_include_directories(${TEST_CASE_2} PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tools)
target_link_libraries(${TEST_CASE_2} cudnn cudart)