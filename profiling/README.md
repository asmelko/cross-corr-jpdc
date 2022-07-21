# Results of profiling

This directory contains results of profiling the CUDA C++ implementation using Nsight Compute tool provided with the CUDA toolkit.

## How to run profiling

When profiling on localhost, you will need to either configure X server so that the NVidia device is run in non-interactive mode,
so that the kernels are not killed by watchdog for running too long, or you will need to run without XServer entirely and
use the NSight Compute CLI.

### Ubuntu 20.04 stop and start XServer

Logout from the current XServer session.

Switch to virtual terminal:
```
ctrl+alt+F2
```

Do not need need to stop the XServer if no user is logged in.
Then run your profiling using the NSight Compute CLI.


The helper script `run_profiling_oneshot.sh` accepts algorithm name, path to the argument file, left input and right input and runs the profiler with the required arguments.