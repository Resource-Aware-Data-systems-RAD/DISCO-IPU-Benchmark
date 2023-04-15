# Benchmark for CoralAI Dev Board Mini and Raspberry Pi with the USB accelerator

This folder contains code for reproducing the experiments on the CoralAI Dev Board Mini and Rapsberry Pi with the USB accelerator. 

The ```coral/examples``` directory contains two relevant benchmarks:
- ```classify_image``` - This is the benchmark leveraging the unoptimized patchify operation for preprocessing. This copies the patches of images by value.
- ```classify_image_memcpy``` - This is the benchmark that uses the optimized version of the patchify operation during preprocessing. This leverages the memcpy operation to copy rows of patches at a time.

The models for these devices can be found in ```coral/examples/models/``` directory. The benchmarks currently expect the models to be in the ```models``` directory next to the built benchmarks. To change the path to the model you can use the ```--model_path``` flag.

To setup the CoralAI Dev Board Mini follow the guide found here: https://coral.ai/docs/dev-board-mini/get-started/#7-run-a-model-using-the-pycoral-api

To setup the RaspberryPi with the USB accelerator follow the guide found here: https://coral.ai/docs/accelerator/get-started/

