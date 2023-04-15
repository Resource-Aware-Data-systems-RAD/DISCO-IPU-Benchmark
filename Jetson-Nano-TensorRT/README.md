# NVIDIA Jetson Nano using the TensorRT inference engine

This folder contains the code for reproducing our results for running TensorRT inference on NVIDIA Jetson Nano devices.

To run a model using the inference engine run the ```benchmark.py``` script and supply the model, batch size, log file and the number of iterations. 
The compiled inference engines for the benchmark can be found in ```engines``` directory. The engines accept a batch size of up to 64.

To run the data collection of all combinations of model sizes and batch sizes run the ```./collect_measurement``` script. This will also collect timestamped tegrastats metrics at 50ms intervals. 