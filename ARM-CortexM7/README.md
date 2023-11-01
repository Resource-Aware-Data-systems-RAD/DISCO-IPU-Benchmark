# Benchmark for ARM-CortexM7 MCU

This folder contains a XCube-AI project for running the MobileNetV1 model with depth multiplier of 0.25 (the only size that fits in device's memory).

To reproduce the results, run this benchmark on the Nucleo-144 board.

The benchmark files can be found in ```CM7/X-CUBE-AI/App``` directory. The main loop of the benchmark can be found in the ```aiSystemPerformance.c``` file.