# Performance benchmark for the Image Processing Units on DISCO CubeSats

This repository contains all of the code necessary to replicate the results of our work. This repository is split into 7 folders. Each of these folders contains further instructions for running the models on the devices under test.

These include:

- `ARM-CortexM7` - STM32CubeIDE project for reproducing results on STM32H745ZI-Q board.
- `CoralAI-Linux` - In-tree implementation of the benchmark targetting the Coral AI development boards running MendelOS (Coral AI Dev Board and Coral AI Dev Board Mini).
- `CoralAI-Micro` - In-tree implementation of the benchmark targetting the Coral AI Dev Board Micro running FreeRTOS.
- `Jetson-Nano-TensorFlow` - Python implementation of the benchmark using the TensorFlow framework for inference.
- `Jetson-Nano-TensorRT` - Python implementation of the benchmark using the TensorRT inference engine.
- `OpenVino-Raspberry-Pi` - In-tree impelemntation of the benchmark using the OpenVino framework on Intel Neural Compute Stick 2 hosted by Raspberry Pi model 3
- `models` - Folder holding all of the trained MobileNetV1 models with varying depth multipliers (sizes)
