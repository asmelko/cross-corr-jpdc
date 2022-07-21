# Using the implementations in your program

As this thesis is mainly aimed at comparing different implementations by profiling and benchmarking, it is not designed to provide easy use of the cross-correlation implementations in external programs.

In this directory, we showcase a Sample application using one of the implementations provided by this thesis.
Dependencies of any application using the cross-correlation implementation:
- CUDA 11.4 or newer

Sample app specific dependencies:
- CMake 3.18 or newer
- Boost 1.71 or newer

Pick one of the algorithms from the list of implemented algorithms described in the main repository [src](/src) directory.
Copy the listed .cu file. For sample application, we choose the `Warp shuffle with multirow_both and multimat_right`, copying the [<repository>/src/naive_shuffle_one_to_many_multirow_both_multimat_right.cu](/src/naive_shuffle_one_to_many_multirow_both_multimat_right.cu) file.

You will need to copy several header and cpp files listed below from the [src](/src) directory in the root of this repository. All the listed files are symlinked into the [src](./src) directory of this application.

The following header files need to be copied:
- [types.cuh](/src/types.cuh)
- [cuda_helpers.cuh](/src/cuda_helpers.cuh)
- [bound_checked_loads.cuh](/src/bound_checked_loads.cuh)
- [row_distribution.cuh](/src/row_distribution.cuh)
- [warp_size.hpp](/src/warp_size.hpp)
- [kernel_args.hpp](/src/kernel_args.hpp)
- [shared_mem.cuh](/src/shared_mem.cuh)
- [kernels.cuh](/src/kernels.cuh)

The following cpp files need to be copied:
- [row_distribution.cpp](/src/row_distribution.cpp)
- [kernel_args.cpp](/src/kernel_args.cpp)

To allow the sample application to parse csv files, we also require the Boost library and the following headers:
- [matrix.hpp](/src/matrix.hpp)
- [host_helpers.hpp](/src/host_helpers.hpp)

During compilation, you will need to define values for the options defined for your chosen implementation, as listed in the Algorithm options section of the README in repository [src](/src) directory.
In the sample application, this is achieved using CMake for the following variables and values:

| Option name | Value |
|-------------|-------|
| SHUFFLE_ONE_TO_MANY_MULTIROW_BOTH_MULTIMAT_RIGHT_SHIFTS_PER_RIGHT_MATRIX_LIMIT | 2 |
| SHUFFLE_ONE_TO_MANY_MULTIROW_BOTH_MULTIMAT_RIGHT_RIGHT_MATRICES_PER_THREAD_LIMIT | 2 |
| SHUFFLE_ONE_TO_MANY_MULTIROW_BOTH_MULTIMAT_RIGHT_LEFT_ROWS_PER_ITERATION_LIMIT | 2 |

The sample application is compiled using the following commands:
```
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE:STRING=Release ..
cmake --build . --config Release
```

The executable `sample_app` can then be used to compute *one-to-many* cross-correlation such as this:
```
./build/sample_app ../data/ina_64_64_1_1.csv ../data/ina_64_64_2_1.csv > ./results.csv
```

The results can be validated using:
```
../build/cross validate ../data/valid_one_to_many_64_64_1_2_1_1.csv ./results.csv
```