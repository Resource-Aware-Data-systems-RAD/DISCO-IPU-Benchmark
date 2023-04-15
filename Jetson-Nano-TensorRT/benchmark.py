import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from multiprocessing import Process, Queue
from math import ceil
import time
import argparse

import matplotlib.pyplot as plt

TRT_LOGGER = trt.Logger()

parser = argparse.ArgumentParser(
    prog='TRT jetson nano demo')
parser.add_argument("-m", dest="model", type=str, required=True)
parser.add_argument("-b", dest="batch_size", type=int, required=True)
parser.add_argument("-r", dest="num_reps", type=int, required=True)
parser.add_argument("-l", dest="log_file", type=str, required=True)
args = parser.parse_args()

# Filename of TensorRT plan file
engine_file = args.model


def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


NUM_IMAGES = 1
IMG_SIZES = (4512, 4512, 3)
BATCH_SIZE = args.batch_size
NUM_BATCHES = ceil(400 / BATCH_SIZE)

print("generating array of size", (NUM_IMAGES, *IMG_SIZES))

'''
GENERATE TEST DATA.
'''


def generate_test_imgs():
    ''' '''
    return np.random.randint(256, size=(NUM_IMAGES, *IMG_SIZES))


imgs = generate_test_imgs()
print(f"Generated {len(imgs)} images of shape {imgs[0].shape}")


def producer(data, queue):
    x_i, y_i = 0, 0

    for batch in range(NUM_BATCHES):
        # start_time = time.perf_counter()
        data_batch = np.zeros((BATCH_SIZE, 224, 224, 3))
        # print("batch", batch)
        for i in range(BATCH_SIZE):
            # print("item", batch * BATCH_SIZE + i)

            if batch * BATCH_SIZE + i >= 400:
                break

            if (batch * BATCH_SIZE + i) % 20 == 0 and (batch * BATCH_SIZE + i) != 0:
                y_i = 0
                x_i += 1

            data_batch[i] = data[x_i *
                                 224: (x_i + 1) * 224, y_i * 224: (y_i + 1) * 224, :]

            # print(x_i * 224, (x_i + 1) * 224, y_i * 224, (y_i + 1) * 224)

            y_i += 1
        data_batch = data_batch.astype("float32") / float(255.0)
        # end_time = time.perf_counter()
        queue.put(data_batch)

        # print("Preparation of this batch took: ", end_time - start_time)


def main():
    example_batch = np.zeros((BATCH_SIZE, 224, 224, 3)).astype("float32")
    with load_engine(engine_file) as engine:
        print("Loaded engine.")

        with engine.create_execution_context() as context:
            print("Created execution context.")

            context.set_binding_shape(engine.get_binding_index(
                "input_layer"), (BATCH_SIZE, 224, 224, 3))
            # Allocate host and device buffers
            bindings = []
            for binding in engine:
                binding_idx = engine.get_binding_index(binding)
                size = trt.volume(context.get_binding_shape(binding_idx))
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                if engine.binding_is_input(binding):
                    input_memory = cuda.mem_alloc(example_batch.nbytes)
                    bindings.append(int(input_memory))
                    print("dtype of input", dtype)
                else:
                    output_buffer = cuda.pagelocked_empty(size, dtype)
                    output_memory = cuda.mem_alloc(output_buffer.nbytes)
                    bindings.append(int(output_memory))
            print("Allocated host and device memory.")

            latencies = 0
            with open(args.log_file, "w+") as log_file:

                # run inference for a single large image using batch size 64
                for img_idx in range(args.num_reps):
                    if img_idx > 0:
                        log_file.write(f"START {int(time.time())}\n")
                    start_time = time.perf_counter()
                    stream = cuda.Stream()

                    q = Queue(7)
                    p = Process(target=producer, args=(imgs[0], q,))

                    p.start()

                    for _ in range(NUM_BATCHES):
                        # start_time = time.perf_counter()
                        input_buffer = np.ascontiguousarray(q.get())
                        # Transfer input data to the GPU.
                        cuda.memcpy_htod_async(
                            input_memory, input_buffer, stream)
                        # Run inference
                        context.execute_async_v2(
                            bindings=bindings, stream_handle=stream.handle)
                        # Transfer prediction output from the GPU.
                        cuda.memcpy_dtoh_async(
                            output_buffer, output_memory, stream)
                        # Synchronize the stream
                        stream.synchronize()
                        # print(np.reshape(output_buffer, (64, 5)))
                        # end_time = time.perf_counter()
                        # print("inference of batch took", end_time-start_time)

                    # print(np.reshape(output_buffer, (64, 5)))
                    p.join()
                    end_time = time.perf_counter()
                    if img_idx > 0:
                        log_file.write(f"STOP {int(time.time())} \n")
                        latencies += (end_time - start_time) * 1000
                    print("Inference of this image took:",
                          (end_time - start_time) * 1000)

                log_file.write(f"AVG {latencies / (args.num_reps - 1)}")


if __name__ == "__main__":
    main()
