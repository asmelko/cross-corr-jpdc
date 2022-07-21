# Benchmarking tool

The benchmarking tool is a Python CLI application designed to simplify running large number of benchmarks or validations using the `cross` executable or any of the existing cross-correlation implementations. In the current version of this repository, the available implementations are SciPy and Matlab.

The tool has the following commands:
- **benchmark**: Run benchmarks using yaml definition files.
- **list**: List available algorithms implemented by the `cross` executable.
- **generate**: Generating input data such as those in [data](../data/) directory.
- **validation**: Use Matlab or SciPy to generate valid results from given inputs, use `cross` executable to compute the result error comperd to precomputed valid result, or check that statistics computed by validation benchmark are within tolerance.
- **transform**: Copy subset of matrices from the csv file.
- **clear**: Clear benchmark results.

## Dependencies
The tool is designed to install all dependencies (except matlab) using **poetry**.

- poetry 1.1.0 or newer
- (optional) matlab R2021a or newer

When benchmarking against matlab or using matlab as the source of validation data (which is the default behavior), matlab needs to be available on the PATH.

## Example benchmarks

First, build the CUDA C++ executable with the following commands **run in the repository root directory**:
```
$ mkdir example_bench_build && cd example_bench_build
$ cmake -D CMAKE_BUILD_TYPE:STRING=Release -D SHUFFLE_MULTIMAT_RIGHT_RIGHT_MATRICES_PER_THREAD_LIMIT=1 -D SHUFFLE_MULTIROW_RIGHT_RIGHT_ROWS_LIMIT=1 -D SHUFFLE_MULTIROW_BOTH_SHIFTS_PER_THREAD_LIMIT=2 -D SHUFFLE_MULTIROW_BOTH_LEFT_ROWS_LIMIT=2 -D SHUFFLE_MULTIROW_BOTH_LOCAL_MEM_SHIFTS_PER_THREAD_LIMIT=1 -D SHUFFLE_MULTIROW_BOTH_LOCAL_MEM_LEFT_ROWS_LIMIT=1 -D SHUFFLE_MULTIROW_RIGHT_MULTIMAT_RIGHT_RIGHT_ROWS_LIMIT=1 -D SHUFFLE_MULTIROW_RIGHT_MULTIMAT_RIGHT_RIGHT_MATS_LIMIT=1 -D SHUFFLE_N_TO_M_MULTIMAT_BOTH_LEFT_MATRICES_PER_THREAD_LIMIT=1 -D SHUFFLE_N_TO_M_MULTIMAT_BOTH_RIGHT_MATRICES_PER_THREAD_LIMIT=1 -D SHUFFLE_N_TO_M_MULTIMAT_BOTH_LOCAL_MEM_LEFT_MATRICES_PER_THREAD_LIMIT=1 -D SHUFFLE_N_TO_M_MULTIMAT_BOTH_LOCAL_MEM_RIGHT_MATRICES_PER_THREAD_LIMIT=1 -D SHUFFLE_N_TO_M_MULTIROW_BOTH_MULTIMAT_BOTH_SHIFTS_PER_THREAD_PER_RIGHT_MATRIX_LIMIT=1 -D SHUFFLE_N_TO_M_MULTIROW_BOTH_MULTIMAT_BOTH_RIGHT_MATRICES_PER_THREAD_LIMIT=1 -D SHUFFLE_N_TO_M_MULTIROW_BOTH_MULTIMAT_BOTH_LEFT_MATRICES_PER_THREAD_LIMIT=1 -D SHUFFLE_N_TO_M_MULTIROW_BOTH_MULTIMAT_BOTH_LEFT_ROWS_PER_ITERATION_LIMIT=1 -D SHUFFLE_ONE_TO_MANY_MULTIROW_BOTH_MULTIMAT_RIGHT_SHIFTS_PER_RIGHT_MATRIX_LIMIT=1 -D SHUFFLE_ONE_TO_MANY_MULTIROW_BOTH_MULTIMAT_RIGHT_RIGHT_MATRICES_PER_THREAD_LIMIT=1 -D SHUFFLE_ONE_TO_MANY_MULTIROW_BOTH_MULTIMAT_RIGHT_LEFT_ROWS_PER_ITERATION_LIMIT=1 -D WARP_PER_SHIFT_SHARED_MEM_RIGHT_MATRICES_PER_BLOCK_LIMIT=1 ..
$ cmake --build . --config Release --parallel
```

These commands minimize compile time by limiting the arguments for all algorithms which are not used by the example benchmark. The build should take less than 5 minutes.

Next we run the `one_to_one` group of the [example/benchmark.yml](./example/benchmark.yml) benchmark, which benchmarks several chosen implementations of the `one-to-one` type utilizing GPU provided by CUDA C++. The results are stored in the same directory as the benchmark definition, in the directory named `results`. The benchmarks should run for around 64 minutes. If you want to reduce the time of the benchmarking, reduce the number of inner or outer iterations, set lower min_measure time or reduce the number of inputs by deleting them from the yaml definition.

From the current directory, run:
```
poetry run ./benchmarking.py -e "../example_bench_build/cross" benchmark -o results ./example/benchmark.yml one_to_one
```

To visualize the results, open the [visualization/example_benchmark.ipynb](/visualization/example_benchmark.ipynb) and run it. The results should look similar to the following graph:

![Example benchmark results](/visualization/example_benchmark_results_readme.svg)

## Running benchmarks

The entry point for the benchmarking tool is the [benchmarking.py](./benchmarking.py) script, which has to be run from a poetry virtual environment. To run a benchmark, execute the following command from within the poetry virtual environment:
```
./benchmarking.py benchmark <path_to_definition> [group]...
```

The benchmark definition file is a yaml file which describes one or more groups of benchmarks. Runs in a group share input sizes, data type, which part of the algorithm is measured, number of iterations, if validation is executed and if output data is kept.

Each run defines a user friendly name, algorithm to run (from the list provided by `cross list`), and settings for algorithm arguments. Each algorithm argument can be specified as array of values. Then the run is repeated for all combinations of all argument values.

See [example/benchmark.yml](./example/benchmark.yml) for an example of benchmark definition.

For the benchmark command itself, the most useful options are `-r|--rerun` and `-c|--continue`. By default, the tool refuses to execute benchmark if any results for given group already exist. With `-r`, the existing results are deleted. With `-c`, the existing measurements are skipped.
