cmake_minimum_required(VERSION 3.18)
project(SAMPLE_APP LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Requires cmake 3.18
set(CMAKE_CUDA_ARCHITECTURES 70 75 80)

# See https://stackoverflow.com/questions/27018340/cmake-does-not-properly-find-cuda-library
FIND_PACKAGE(CUDAToolkit 11.4 REQUIRED)
FIND_PACKAGE(Boost 1.71 COMPONENTS program_options REQUIRED)

set(CPP_SOURCES
        ${CMAKE_SOURCE_DIR}/src/main.cpp
        ${CMAKE_SOURCE_DIR}/src/row_distribution.cpp
        ${CMAKE_SOURCE_DIR}/src/kernel_args.cpp
)
# Route everything through nvcc
set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE CUDA)


set(CUDA_SOURCES
        ${CMAKE_SOURCE_DIR}/src/naive_shuffle_one_to_many_multirow_both_multimat_right.cu
)

add_executable(sample_app ${CPP_SOURCES} ${CUDA_SOURCES})


target_include_directories(sample_app PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(sample_app PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(sample_app PUBLIC ${Boost_LIBRARIES})
target_link_libraries(sample_app PUBLIC CUDA::cudart)
target_link_libraries(sample_app PUBLIC CUDA::cufft)

target_compile_options(sample_app PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
target_compile_options(sample_app PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra>)

target_compile_options(sample_app PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-lineinfo>)

target_compile_options(sample_app PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-g -rdynamic>)
target_compile_options(sample_app PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G -Xcompiler -rdynamic>)

target_compile_definitions(sample_app PRIVATE SHUFFLE_ONE_TO_MANY_MULTIROW_BOTH_MULTIMAT_RIGHT_SHIFTS_PER_RIGHT_MATRIX_LIMIT=2)
target_compile_definitions(sample_app PRIVATE SHUFFLE_ONE_TO_MANY_MULTIROW_BOTH_MULTIMAT_RIGHT_RIGHT_MATRICES_PER_THREAD_LIMIT=2)
target_compile_definitions(sample_app PRIVATE SHUFFLE_ONE_TO_MANY_MULTIROW_BOTH_MULTIMAT_RIGHT_LEFT_ROWS_PER_ITERATION_LIMIT=2)
