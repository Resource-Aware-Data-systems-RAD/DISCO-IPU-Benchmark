/* Copyright 2019-2021 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// An example to classify image.
// The input image size must match the input size of the model and be stored as
// RGB pixel array.
// In linux, you may resize and convert an existing image to pixel array like:
//   convert cat.bmp -resize 224x224! cat.rgb
#include <cmath>
#include <chrono>
#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "coral/classification/adapter.h"
#include "coral/examples/file_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/interpreter.h"

ABSL_FLAG(std::string, model_path, "models/model_edgetpu_050.tflite",
          "Path to the tflite model.");
ABSL_FLAG(int, n_runs, 10, "Number of repetitions of the experiment");

std::chrono::steady_clock::time_point START_TIMER;
std::chrono::steady_clock::time_point START_TIMER_INPUT;
std::chrono::steady_clock::time_point START_TIMER_INFERENCE;

void start_timer()
{
    START_TIMER = std::chrono::steady_clock::now();
}

void start_timer_input() {
	START_TIMER_INPUT = std::chrono::steady_clock::now();
}

void start_timer_inference() {
	START_TIMER_INFERENCE = std::chrono::steady_clock::now();
}

void end_experiment(std::chrono::steady_clock::time_point start = START_TIMER, int n_images = 100)
{
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "Finished running " << n_images << " reps." << std::endl;
  std::cout
      << "Average time per rep: "
      <<  elapsed_time / n_images << "µs"
      << std::endl;
}

void end_setup(std::chrono::steady_clock::time_point start = START_TIMER)
{
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout
      << "Setup time: "
      <<  elapsed_time << "µs"
      << std::endl;
}

uint64_t end_timer(std::chrono::steady_clock::time_point start = START_TIMER)
{
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

int main(int argc, char* argv[]) {
  start_timer();
  absl::ParseCommandLine(argc, argv);

  // Load the model.
  const auto model = coral::LoadModelOrDie(absl::GetFlag(FLAGS_model_path));
  auto edgetpu_context = coral::ContainsEdgeTpuCustomOp(*model)
                             ? coral::GetEdgeTpuContextOrDie()
                             : nullptr;
  auto interpreter =
      coral::MakeEdgeTpuInterpreterOrDie(*model, edgetpu_context.get());
  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  CHECK_EQ(interpreter->inputs().size(), 1);
  const auto* input_tensor = interpreter->input_tensor(0);
  CHECK_EQ(input_tensor->type, kTfLiteUInt8)
      << "Only support uint8 input type.";
  auto input = coral::MutableTensorData<uint8_t>(*input_tensor);

	uint16_t tot_height = 4512;
	uint16_t tot_width = 4512;
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

    uint8_t tile_size = 224;

  start_timer();

  // iterate for N_RUNS 4512x4512 images
  for(size_t i = 0; i < absl::GetFlag(FLAGS_n_runs); i++) {
    //coral::ReadFileToOrDie(absl::GetFlag(FLAGS_image_path), reinterpret_cast<char*>(input.data()), input.size());

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

                    memcpy(input.data() + ((h-height_offset) * tile_size * tot_channels), large_img + (h * tot_width * tot_channels + width_offset * tot_channels), sizeof(uint8_t) * tile_size * tot_channels);
				}
				input_time_sum += end_timer(START_TIMER_INPUT);

				start_timer_inference();
				CHECK_EQ(interpreter->Invoke(), kTfLiteOk);
				inference_time_sum += end_timer(START_TIMER_INFERENCE);

				for (auto result: coral::GetClassificationResults(*interpreter, 0.0f, 1)) {
					// tile postprocessing goes here
					continue;
				}
			}	
		}
		std::cout << "Average time to fill input buffer: " << input_time_sum / 400 << " micros" << std::endl;
		std::cout << "Average time for tile inference: " << inference_time_sum / 400 << " micros" << std::endl;
    //sleep(1);
  }

  end_experiment(START_TIMER, absl::GetFlag(FLAGS_n_runs));
  
  return 0;
}
