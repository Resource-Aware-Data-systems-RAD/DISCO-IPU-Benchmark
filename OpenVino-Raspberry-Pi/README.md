# Intel Neural Compute Stick 2 on Raspberry Pi

This folder contains an OpenVino build for RaspberryPi (Buster). It was produced by following the Approach 1 of this guide: https://github.com/openvinotoolkit/openvino_contrib/wiki/How-to-build-ARM-CPU-plugin#approach-1-build-opencv-openvino-and-the-plugin-using-pre-configured-dockerfile-cross-compiling-the-preferred-way

The ```samples/cpp``` directory contains two benchmarks:
- ```cubesat_benchmark``` - runs the benchmark in synchronous mode, with the preprocessing in the main thread.
- ```cubesat_benchmark_async``` - runs the benchmark in asynchronous mode with variable number of concurrent inference requests and preprocessing done in a separate thread.

These samples can be built by running ```./build_samples.sh``` from the ```samples/cpp``` directory.
The benchmark expect a model file which can be found in the model directory, e.g.: ```models/mobilenet_v1_050/mobilenet_v1_050.xml```.