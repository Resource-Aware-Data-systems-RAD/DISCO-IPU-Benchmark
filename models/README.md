# Models

This folder contains the trained MobileNetV1 models with depth multipliers of 0.25, 0.5 and 1. 

To retrain the models run: ```python3 model_training.py```. This will run the training of the model with depth multiplier of 0.5. To change the size of the network, simple change value of the ```depth_multiplier``` variable. Running this script will produce the model in FP32, as well as the TensorFlow Lite equivalents in FP16 and INT8.