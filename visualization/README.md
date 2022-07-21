# Visualization

This directory contains Jupyter notebooks for visualizing benchmarking results.
The data is used straight from the [benchmarking](/benchmarking/) directory, currently from the `text` for the diagrams in the text of the thesis and `args_test` for the benchmarks determining the optimal arguments for each implementation.

## Running

First install dependencies using poetry:
```
$ poetry install
```

You can then use and run the notebooks.

Each notebook contains several code blocks, each runnable independently and displaying one or more diagrams. The [text_diagrams.ipynb](./text_diagrams.ipynb) notebook is the only notebook which saves the diagrams into .svg files, which were then copied and used in the text of the thesis. The [text_diagrams.ipynb](./text_diagrams.ipynb) code blocks are just copies of code blocks from other notebooks.

The results used by the notebooks are stored in a .tar.gz archive in git. Before running the notebooks, extract the archive into the directory it is in. The archives are [benchmarking/args_test/results.tar.gz](/benchmarking/args_test/results.tar.gz) and [benchmarking/text/results.tar.gz](/benchmarking/text/results.tar.gz).

## Notebooks

| Notebook | Contents |
|----------|----------|
| [text_diagrams.ipynb](./text_diagrams.ipynb) | Noteboook producing every diagram used in the text of the thesis. Contains code blocks copied from all other notebooks. |
| [block_per_shift.ipynb](./block_per_shift.ipynb) | Arguments of the Block per shift algorithm and comparison with the Warp per shift base algorithm |
| [definition_based_speedup_warp_shuffle.ipynb](./definition_based_speedup_warp_shuffle.ipynb) | Comparison of definition based algorithms |
| [example_benchmark.ipynb](./example_benchmark.ipynb) | Part of the examples in the root directory README file |
| [fft_speedup_startup.ipynb](./fft_speedup_startup.ipynb) | Comparison of the FFT-based and definition-based algorithms with limited caching |
| [fft_speedup.ipynb](./fft_speedup.ipynb) | Comparison of the FFT-based and definition-based algorithms |
| [fft_speedup2.ipynb](./fft_speedup2.ipynb) | Comparison of the FFT-based and definition-based algorithms, processing results from the second round of measurements |
| [fft.ipynb](./fft.ipynb) | Visualization of the behavior of the FFT-based algorithm |
| [gpulab_args0.ipynb](./gpulab_args0.ipynb) | First round of measurements for determining the optimal arguments of the definition-based algorithms |
| [gpulab_args1.ipynb](./gpulab_args1.ipynb) | Second round of measurements for determining the optimal arguments of the definition-based algorithms |
| [gpulab_args2.ipynb](./gpulab_args2.ipynb) | Third round of measurements for determining the optimal arguments of the definition-based algorithms |
| [gpulab_args3.ipynb](./gpulab_args3.ipynb) | Fourth round of measurements for determining the optimal arguments of the definition-based algorithms |
| [local_mem.ipynb](./local_mem.ipynb) | The effects of local memory accesses on the performace of the multimat_both and multirow_both algorithms |
| [matlab_speedup.ipynb](./matlab_speedup.ipynb) | Comparison of matlab with other algorithms |
| [matlab_startup_speedup.ipynb](./matlab_startup_speedup.ipynb) | Comparison of matlab against other algorithms with limited caching |
| [scipy_speedup.ipynb](./scipy_speedup.ipynb) | Comparison of SciPy against other algorithms |
| [warp_per_shift_optimizations.ipynb](./warp_per_shift_optimizations.ipynb) | Comparison of Warp per shift algorithm family |
| [warp_per_shift_optimizations2.ipynb](./warp_per_shift_optimizations2.ipynb) | Second round of measurments of Warp per shift algorithm family |
| [warp_shuffle_optimizations.ipynb](./warp_shuffle_optimizations.ipynb) | Comparison of  the Warp shuffle algorithm family |

## Helper files

[shared.py](./shared.py) implements data structures and methods used to process results produced by the Benchmarking tool. It contains classes for object representation of Benchmark, Group and Run, parsing the results into object representation. It is used by all the notebooks.