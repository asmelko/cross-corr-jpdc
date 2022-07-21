# Accelerating cross-correlation with GPUs

This repository contains the code accompanying the Master thesis "Accelerating cross-correlation with GPUs". It includes the CUDA C++ implementation of cross-correlation using both definition-based and FFT-based approach, together with the tools for benchmarking and validation of these implementations.

## Repository structure

The top level directories:

- [src](./src/) - The CUDA C++ program implementing cross-correlation using definition-based and FFT-based algorithms together with result validation. See the README for list of implemented algorithms and their compile options.
- [benchmarking](./benchmarking/) - Python tool for benchmarking and validation of the CUDA C++ program and its cross-correlation implementations against each other or against real-world cross-correlation implementations in SciPy and Matlab.
- existing -  Cross-correlation implementations used in the real-world. This repository utilizes them for result validation and comparison against our CUDA C++ implementations.
- [visualization](./visualization/) - Jupyter notebooks providing visualization of the results measured by the benchmarking tool. Includes all diagrams used in the text of the thesis and much more.
- gpulab - Scripts for building, measuring and validating the CUDA C++ program on KSI Cluster gpulab. Also usable locally. Must be executed with the gpulab directory as working directory.
- profiling - Results of profiling the CUDA C++ implementation using Nsight Compute profiler.
- [sample_app](./sample_app/) - Sample application utilizing one of the cross-correlation implementations provided by this thesis.

## Building the C++ CUDA implementations

The CUDA C++ program has following dependencies:
- CMake 3.18 or newer,
- CUDA 11.4 or newer,
- (Optional) Boost 1.71 or newer,
- (Optional) nlohmann/json 3.7.3 or newer

Optional dependencies can either be provided externally or downloaded automatically using a CMake superbuild.

To build the program using externally provided optional dependencies, follow these steps:
```bash
$ cd <repository_root>
$ mkdir build && cd build
$ cmake -D CMAKE_BUILD_TYPE:STRING=Release ..
$ cmake --build . --parallel --config Release
```
The build was tested on Ubuntu 20.04 and Rocky Linux 8.5. With the default configuration provided in the CMakeLists.txt file, the build may take over 8 hours due to the number of generated kernels and other function instantiations. You can use `cmake-gui` or the `-D` option to the `cmake ..` command to change the algorithm options, reducing compile times but limiting the available argument values accepted by the given algorithm argument.

To build the program without externally provided optional dependencies, run the following commands:
```
$ cd <repository_root>/gpulab
$ bash build.sh
```

To build the program on the KSI Cluster gpulab, run the following commands:
```
$ cd <repository_root>/gpulab
$ sbatch build.sh
```
To change any of the algorithm options, add the `-D` option with the correct value to the command on line 18 in the [gpulab/build.sh](./gpulab/build.sh) file.

All builds described above will create `cross` executable in the `build` directory.

To test that the build was successful, run the following command which should list all the algorithms implemented by the executable:
```
$ ./build/cross list
cpu_n_to_m
cpu_n_to_mn
cpu_one_to_many
cpu_one_to_one
fft_better_n_to_m
...
```

To test that the CUDA framework is working properly, run a simple computation:
```
$ ./build/cross run -o output.csv nai_shuffle_one_to_one ./data/ina_128_128_1_1.csv ./data/ina_128_128_1_2.csv
Loading inputs
Allocating
Transfering data
Running test alg
Copying output data to host
Free resources
Storing results
No validation
```

## Running the C++ CUDA implementations

The C++ CUDA program provides a CLI application implementing several commands for running cross-correlation computation, measuring execution times of this computation and its parts or validating results.

Small dataset is included in this repository in the [data](./data) directory. When using the benchmarking tool described following section, random input data (and optionally validation data for this input) is generated by the tool itself.

### Examples

List available algorithms:
```
$ ./build/cross list
```

Compute cross-correlation of two matrices using the Warp shuffle algorithm:
```
$ ./build/cross run -o output.csv nai_shuffle_one_to_one ./data/ina_128_128_1_1.csv ./data/ina_128_128_1_2.csv
```

Run computation and validate the results:
```
$ ./build/cross run -v ./data/valid_n_to_mn_128_128_2_4_1_1.csv nai_shuffle_multirow_both_multimat_right_n_to_mn ./data/ina_128_128_2_1.csv ./data/ina_128_128_4_1.csv
```
Run Warp shuffle base implementation and measure the kernel runtime 5 times, each time adaptively increasing the number of kernel iterations measured until it runs at least 1 second, writing the results to the `kernel_measurement.csv` file. This also will not store the results, as we are not interested in those.

```
./build/cross run -b "Algorithm" -l 5 -m 1 -t "kernel_measurements.csv" "nai_shuffle_one_to_one" ./data/ina_128_128_1_1.csv ./data/ina_128_128_1_2.csv
```

### CLI commands and their options

- **help**: Prints help message describing the CLI interface.
- **run**: Runs a cross-correlation computation using the specified algorithm, optionally measuring execution times or validating output.
    - Positional arguments:
        1. `<alg>`: The algorithm to use for the computation.
        2. `<ref_path>`: Path to the left input matrix/matrices.
        3. `<target_path>`: Path to the right input matrix/matrices.
    - Options and switches
        - `-d|--data_type <"single"|"double">`: Data type to use for computation, choice between single and double floating point numbers, defaults to single.
        - `-o|--out <path>`: Path to the output file for the result of the computation. The result is only stored if this option is provided.
        - `-t|--times <path>`: Path to the file in which to store the measured execution times, defaults to `measurements.csv`.
        - `-v|--validate [path]`: Option with an optional value which either enables validation against valid results computed using simple CPU implementation, or additionally provides a path to valid results to use for comparison. Validation results are written to the standard output.
        - `-n|--normalize`: If results of FFT-based algorithm should be normalized before being written to the output file. Ignored if not writing output or if algorithm is not FFT-based.
        - `-a|--append`: Append times to the file instead of overwriting it. Also changes validation results format printed to the standard output.
		- `-p|--no_progress` Do not print computation progress to standard output.
		- `-b|--benchmark_type <"Compute"|"CommonSteps"|"Algorithm"|"None">`: Which part of the implementation should be measured, defaults to `"None"`.
		- `-l|--outer_loops`: Number of times the algorithm should be run after loading the data into memory. Each run is measured separately.
		- `-m|--min_time`: Minimum measurement time for measurements with adaptive iteration count, defaults to 1 second.
		- `--args_path`: Path to the JSON file containing argument values for the algorithm.
- **list**: Lists the available algorithms. Each of the listed names can be used as `alg` positional argument of the **run** command.
- **validate**: Compares two result files, executing the same validation as done by the **run** command with the `-v` option.
    - Positional arguments:
        1. `<template_data_path>`: The path to the known correct output to compare against.
        2. `<validate_data_path>`: The path to the data to be validated.
    - Options and switches:
        - `-n|--normalize`: Normalize validated data. This option is useful when results of FFT-based algorithm were stored without normalization.
        - `-c|--csv`: Print output in CSV format instead of in human readable format.
        - `-p|--print_header`: Print header before the output data. Useful when not appending to existing file.
- **input**: Validate that the given input matrices can be computed by given algorithm.
    - Positional arguments:
        1. `<alg_type>`: The algorithm to validate for.
        2. `<rows>`: The number of rows of each input matrix.
        3. `<cols>`: The number of columns of each input matrix.
        4. `<left_matrices>`: The number of left input matrices.
        5. `<right_matrices>`: The number of right input matrices.

## Recreating the results shown in the text of the thesis

The following benchmarks are used by the Jupyter notebook [text_diagrams.ipynb](./visualization/text_diagrams.ipynb) generating diagrams for the thesis text:
- [block_per_shift.yml](./benchmarking/text/block_per_shift.yml)
- [fft.yml](./benchmarking/text/fft.yml)
- [fft_small.yml](./benchmarking/text/fft_small.yml)
- [warp_shuffle_optimizations_speedup.yml](./benchmarking/text/warp_shuffle_optimizations_speedup.yml)
- [warp_per_shift_optimizations_speedup.yml](./benchmarking/text/warp_per_shift_optimizations_speedup.yml)
- [fft_speedup_compute_resolution.yml](./benchmarking/text/fft_speedup_compute_resolution.yml)
- [fft_speedup_compute_resolution2.yml](./benchmarking/text/fft_speedup_compute_resolution2.yml)
- [fft_speedup_compute_startup_resolution.yml](./benchmarking/text/fft_speedup_compute_startup_resolution.yml)
- [scipy_speedup.yml](./benchmarking/text/scipy_speedup.yml)

All produce results in a directory with the same name as the definition file without the `.yml` suffix. Apart from the matlab benchmark, all were run on both gpulab and notebook. Results of gpulab runs were copied to [benchmarking/text](./benchmarking/text/) directory with the suffix `_gpulab` added to them. The total time required to run all the benchmarks is several weeks.

As attached to the thesis and commited to git, the results are compressed into a `.tar.gz` archive in the [benchmarking/text](./benchmarking/text/). Extracting the archive should allow you to generate the diagrams yourself.
