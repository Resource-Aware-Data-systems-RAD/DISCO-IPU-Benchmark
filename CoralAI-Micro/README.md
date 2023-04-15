# CoralAI Dev Board Micro benchmark

This folder contains the project for running the benchmark on the CoralAI Dev Board Micro. 
The ```apps``` directory contains two benchmarks:
- ```cubesat_benchmark``` - This is the benchmark leveraging the unoptimized patchify operation for preprocessing. This copies the patches of images by value.
- ```cubesat_benchmark_memcpy``` - This is the benchmark that uses the optimized version of the patchify operation during preprocessing. This leverages the memcpy operation to copy rows of patches at a time.

These two directories are setup to run the models with depth multiplier of 0.50. To change this behavior, change the model path in ```CMakeLists.txt``` and the ```kModelPath``` variable in ```cubesat_benchmark.cc``` files.

In order to setup the device, build the benchmarks and run them on the device, please follow the instructions found here:
https://coral.ai/docs/dev-board-micro/get-started/#4-set-up-for-freertos-development