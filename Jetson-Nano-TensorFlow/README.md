# NVIDIA Jetson Nano using TensorFlow

This folder contains all code needed to reproduce our results for running TensorFlow on the NVIDIA Jetson Nano.

The code for the benchmark is contained in the ```consumer_producer.py``` file. This script expects the batch size, depth multiplier of the model, log file and the number of iterations.

In order to run all configurations of model sizes and the batch sizes run the ```./collect_measurements``` script. This will additionally collect timestamped tegrastats metrics at 50ms interval.