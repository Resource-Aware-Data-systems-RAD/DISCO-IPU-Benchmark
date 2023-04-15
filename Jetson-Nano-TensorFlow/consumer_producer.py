import numpy as np
import time
import tensorflow as tf
import logging
from math import ceil
from queue import Queue
from threading import Thread
import argparse


tf.debugging.set_log_device_placement(False)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

logger = tf.get_logger()
logger.setLevel(logging.FATAL)

parser = argparse.ArgumentParser(
    prog='Tensorflow jetson nano demo')
parser.add_argument("-m", dest="model", type=str, required=True)
parser.add_argument("-b", dest="batch_size", type=int, required=True)
parser.add_argument("-r", dest="num_reps", type=int, required=True)
parser.add_argument("-l", dest="log_file", type=str, required=True)
args = parser.parse_args()

# Path to the model.
MODELS = {
    "025": '../models/mobilenetv1_025/1658844009_85.69_val_acc/',
    "050": '../models/mobilenetv1_050/1658844881_89.51_val_acc/',
    "100": '../models/mobilenetv1_100/1658844969_91.69_val_acc/',
}

NUM_IMAGES = 1
IMG_SIZES = (4512, 4512, 3)

'''
GENERATE TEST DATA.
'''
def generate_test_imgs():
    imgs = []
    for i in range(NUM_IMAGES):
        tmp_img = np.random.randint(256, size=IMG_SIZES)

        imgs.append(tmp_img)
    return imgs

imgs = generate_test_imgs()
print(f"Generated {len(imgs)} images of shape {imgs[0].shape}")

def producer(queue, data, batch_size):
    num_batches = ceil(400 / batch_size)
    
    x_i, y_i = 0, 0 

    for batch in range(num_batches):
        data_batch = []
        #print("batch", batch)
        for i in range(batch_size):
            #print("item", batch * batch_size + i)
            
            if batch * batch_size + i >= 400:
                break
            
            if (batch * batch_size + i) % 20 == 0 and (batch * batch_size + i) != 0:
                y_i = 0
                x_i += 1
            
            item = data[x_i * 224 : (x_i + 1) * 224, y_i * 224 : (y_i + 1) * 224, : ]
            data_batch.append(item)

            #print(x_i * 224, (x_i + 1) * 224, y_i * 224, (y_i + 1) * 224)

            y_i += 1

        data_batch = tf.constant(np.array(data_batch))
        queue.put(data_batch)
    queue.put(None)

def consumer(queue, model):
    while True:
        item = queue.get()

        if item is None:
            break

        out = model(item)

def main():
    base_model = tf.keras.models.load_model(MODELS[args.model])
    rescaling_layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)

    inputs = tf.keras.Input(shape=(224,224,3))
    x = rescaling_layer(inputs)
    outputs = base_model(x)
    model = tf.keras.Model(inputs, outputs)

    latencies = 0
    with open(args.log_file, "w+") as log_file:
        for i in range(args.num_reps):
            if i > 0:
                log_file.write(f"START {int(time.time())}\n")
            start_time = time.perf_counter()

            queue = Queue()

            consumer_thread = Thread(target=consumer, args=(queue, model,))
            consumer_thread.start()
        
            producer_thread = Thread(target=producer, args=(queue, imgs[0], args.batch_size,))
            producer_thread.start()
            
            producer_thread.join()
            consumer_thread.join()

            end_time = time.perf_counter()
            if i > 0:
                log_file.write(f"STOP {int(time.time())}\n")
                latencies += (end_time - start_time) * 1000
            print("Inference of this image took:", (end_time - start_time) * 1000)
        log_file.write(f"AVG {latencies / (args.num_reps - 1)}")
            
if __name__ == "__main__":
    main()


