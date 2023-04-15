// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include <chrono>

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/base/timer.h"
#include "libs/tensorflow/classification.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

// [start-sphinx-snippet:classify-image]
namespace coralmicro {
namespace {
constexpr char kModelPath[] =
    "apps/cubesat_benchmark/model_edgetpu_050.tflite";
constexpr char kImagePath[] = "/apps/cubesat_benchmark/cat_224x224.rgb";
constexpr int kTensorArenaSize = 200000; //1000000 if 100%
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);


uint64_t START_TIMER;
uint64_t START_TIMER_INPUT;
uint64_t START_TIMER_INFERENCE;

int N_RUNS = 100;

void start_timer()
{
    START_TIMER = TimerMicros();
}

void start_timer_input() {
	START_TIMER_INPUT = TimerMicros();
}

void start_timer_inference() {
	START_TIMER_INFERENCE = TimerMicros();
}

void end_experiment(uint64_t start = START_TIMER, int n_images = 100)
{
  uint64_t end = TimerMicros();
  int elapsed_time = (int)(end - start);
  printf("Finished running %d reps. \r\n", n_images);
  printf("Total elapsed time: %d micros \r\n", elapsed_time);
  printf("Average time per rep: %d micros \r\n", (elapsed_time / n_images));
}

void end_setup(uint64_t start = START_TIMER)
{
  uint64_t end = TimerMicros();
  int elapsed_time = (int)(end - start);
  printf("Setup time %d µs \r\n", elapsed_time);
}

uint64_t end_timer(uint64_t start = START_TIMER)
{
  uint64_t end = TimerMicros();
  return end - start;
}


void Main() {
  // Turn on Status LED to show the board is on.
  LedSet(Led::kStatus, true);
  LedSet(Led::kUser, true);

  printf("Hello from benchamrk\n");

  // wait 10s for user to connect
  vTaskDelay(pdMS_TO_TICKS(10000));

  TimerInit();

  start_timer();

  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath, &model)) {
    printf("ERROR: Failed to load %s\r\n", kModelPath);
    return;
  }

  // [start-sphinx-snippet:edgetpu]
  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!tpu_context) {
    printf("ERROR: Failed to get EdgeTpu context\r\n");
    return;
  }

  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<1> resolver;
  resolver.AddCustom(kCustomOp, RegisterCustomOp());

  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena, kTensorArenaSize,
                                       &error_reporter);
  // [end-sphinx-snippet:edgetpu]
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed\r\n");
    return;
  }

  int bytes_used = (int)interpreter.arena_used_bytes();
  printf("Bytes used: %d \r\n", bytes_used);


  if (interpreter.inputs().size() != 1) {
    printf("ERROR: Model must have only one input tensor\r\n");
    return;
  }

  auto* input_tensor = interpreter.input_tensor(0);
  uint8_t* input = tflite::GetTensorData<uint8_t>(input_tensor);

  // mock data [4512, 4512, 3]
  // can only allocate [2272, 2272, 3] = 10*10 tiles with 32px borders as in original
  uint16_t tot_height = 2272;
	uint16_t tot_width = 2272; //1824 if 100%
	uint8_t tot_channels = 3;
	uint8_t* large_img = (uint8_t *) malloc(tot_height * tot_width * tot_channels * sizeof(uint8_t));

	for (uint16_t h = 0; h < tot_height; h++) {
		for (uint16_t w = 0; w < tot_width; w++) {
			for (uint8_t c = 0; c < tot_channels; c++) {
				large_img[h * tot_width * tot_channels + w * tot_channels + c] = (uint8_t)(rand() % 256);
			}
		}
	}

  end_setup();

  //vTaskDelay(pdMS_TO_TICKS(2000)); 

  uint8_t tile_size = 224;

  start_timer();

  for(int i = 0; i < N_RUNS; i++) {
    uint16_t counter = 0;
		uint64_t input_time_sum = 0;
		uint64_t inference_time_sum = 0;

		for ( uint16_t height_offset = 0; height_offset < tot_height; height_offset += tile_size ) {
			if ((tot_height - height_offset) / tile_size == 0) continue;
			
			for ( uint16_t width_offset = 0; width_offset < tot_width; width_offset += tile_size ) {
				if ((tot_width - width_offset) / tile_size == 0) continue;
				// made sure that we are processing full tiles
				counter++;
				start_timer_input();
				for (uint16_t h = height_offset; h < tile_size + height_offset; h++ ) {
					
					for (uint16_t w = width_offset; w < tile_size + width_offset; w++) {
						
						for (uint8_t c = 0; c < tot_channels; c++ ) {
							input[(h - height_offset) * tile_size * tot_channels + (w - width_offset) * tot_channels + c] = large_img[h * tot_width * tot_channels + w * tot_channels + c];
						}
					}
				}

        input_tensor->bytes = sizeof(uint8_t) * tile_size * tile_size * tot_channels;

				input_time_sum += end_timer(START_TIMER_INPUT);

				start_timer_inference();
				if (interpreter.Invoke() != kTfLiteOk) {
            printf("ERROR: Invoke() failed\r\n");
            return;
        }
				inference_time_sum += end_timer(START_TIMER_INFERENCE);

				auto results = tensorflow::GetClassificationResults(&interpreter, 0.0f, 3);
        // tile post-processing goes here
			}	
		}

    printf("Average time to fill input buffer: %d micros. \r\n", (int)(input_time_sum / ((tot_height / tile_size) * (tot_height / tile_size))));
    printf("average time for tile inference: %d micros \r\n", (int)(inference_time_sum / ((tot_height / tile_size) * (tot_height / tile_size))));
    //vTaskDelay(pdMS_TO_TICKS(100)); //delay for 100ms to distinguish between the runs
  }

  end_experiment(START_TIMER, N_RUNS);

  free(large_img);

  return;

}
}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
  vTaskSuspend(nullptr);
}
// [end-sphinx-snippet:classify-image]
