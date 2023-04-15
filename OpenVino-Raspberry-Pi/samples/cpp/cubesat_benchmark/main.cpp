#include <sys/stat.h>

#include <condition_variable>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <thread>

// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "samples/slog.hpp"
#include "format_reader_ptr.h"

#include "benchmark.h"
// clang-format on

using namespace ov::preprocess;

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
  slog::info << "Finished running " << n_images << " reps." << slog::endl;
  slog::info
      << "Average time per rep: "
      <<  elapsed_time / n_images << "µs"
      << slog::endl;
}

void end_setup(std::chrono::steady_clock::time_point start = START_TIMER)
{
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  slog::info
      << "Setup time: "
      <<  elapsed_time << "µs"
      << slog::endl;
}

uint64_t end_timer(std::chrono::steady_clock::time_point start = START_TIMER)
{
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

/**
 * @brief Checks input args
 * @param argc number of args
 * @param argv list of input arguments
 * @return bool status true(Success) or false(Fail)
 */
bool parse_and_check_command_line(int argc, char* argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        show_usage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_m.empty()) {
        show_usage();
        throw std::logic_error("Model is required but not set. Please set -m option.");
    }

    return true;
}

int main(int argc, char* argv[]) {
    int tile_size = 224;
    uint16_t tot_height = (uint16_t)(tile_size * 20 + 32); // limiting the size of the image
    uint16_t tot_width = (uint16_t)(tile_size * 20 + 32);
    uint8_t tot_channels = 3;
    int num_tiles = (tot_height / tile_size) * (tot_width / tile_size);
    uint8_t *large_img = (uint8_t*)malloc(tot_height * tot_width * tot_channels * sizeof(uint8_t));

    try {
        start_timer();
        // -------- Get OpenVINO Runtime version --------
        slog::info << ov::get_openvino_version() << slog::endl;

        // -------- Parsing and validation of input arguments --------
        if (!parse_and_check_command_line(argc, argv)) {
            return EXIT_SUCCESS;
        }

        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        ov::Core core;

        // -------- Step 2. Read a model --------
        slog::info << "Loading model files:" << slog::endl << FLAGS_m << slog::endl;
        std::shared_ptr<ov::Model> model = core.read_model(FLAGS_m);
        printInputAndOutputsInfo(*model);

        // -------- Step 3. Configure preprocessing --------
        const ov::Layout tensor_layout{"NHWC"};

        ov::preprocess::PrePostProcessor ppp(model);
        // 1) input() with no args assumes a model has a single input
        ov::preprocess::InputInfo& input_info = ppp.input();
        // 2) Set input tensor information:
        // - precision of tensor is supposed to be 'u8'
        // - layout of data is 'NHWC'
        input_info.tensor().set_element_type(ov::element::u8).set_layout(tensor_layout);
        // 3) Here we suppose model has 'NCHW' layout for input
        input_info.model().set_layout("NHWC");
        // 4) output() with no args assumes a model has a single result
        // - output() with no args assumes a model has a single result
        // - precision of tensor is supposed to be 'f32'
        ppp.output().tensor().set_element_type(ov::element::f32);

        // 5) Once the build() method is called, the pre(post)processing steps
        // for layout and precision conversions are inserted automatically
        model = ppp.build();

        // -------- Step 4. Generate input --------
        slog::info << "Generating a large image of size [" << tot_height << ", " << tot_width << ", " << (uint16_t)tot_channels << "]" << slog::endl;

        for (uint16_t h = 0; h < tot_height; h++)
        {
            for (uint16_t w = 0; w < tot_width; w++)
            {
                for (uint8_t c = 0; c < tot_channels; c++)
                {
                    //slog::info << "Index: " << h * tot_width * tot_channels + w * tot_channels + c << slog::endl;
                    large_img[h * tot_width * tot_channels + w * tot_channels + c] = (uint8_t)(std::rand() % 256);
                }
            }
        }

        // -------- Step 5. Loading model to the device --------
        slog::info << "Loading model to the device MYRIAD" << slog::endl;
        ov::CompiledModel compiled_model = core.compile_model(model, "MYRIAD");

        // -------- Step 6. Create infer request --------
        slog::info << "Create infer request" << slog::endl;
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        ov::Tensor input_tensor = infer_request.get_input_tensor();

        end_setup();

        start_timer();

        for (int run = 0; run < FLAGS_n; run++ ) {
            uint64_t input_time_sum = 0;
            uint64_t inference_time_sum = 0;


            for (uint16_t height_offset = 0; height_offset < tot_height; height_offset += tile_size)
            {
                if ((tot_height - height_offset) / tile_size == 0)
                    continue;

                for (uint16_t width_offset = 0; width_offset < tot_width; width_offset += tile_size)
                {
                    if ((tot_width - width_offset) / tile_size == 0)
                        continue;

                    // -------- Step 7. Populate input buffer with a tile --------
                    //slog::info << "Fill buffer." << slog::endl;
                    start_timer_input();
                    for (uint16_t h = height_offset; h < tile_size + height_offset; h++ ) {
                        uint8_t *input_ptr = input_tensor.data<std::uint8_t>() + ((h-height_offset) * tile_size * tot_channels);
                        uint8_t *tile_ptr = large_img + (h * tot_width * tot_channels + width_offset * tot_channels);
                        std::memcpy(input_ptr, tile_ptr, sizeof(uint8_t) * tile_size * tot_channels);
                    }
                    input_time_sum += end_timer(START_TIMER_INPUT);

                    // -------- Step 8. Infer single tile --------
                    //slog::info << "Infer." << slog::endl;

                    start_timer_inference();
                    infer_request.infer();
                    inference_time_sum += end_timer(START_TIMER_INFERENCE);

                    // -------- Step 9. Process output --------
                    //slog::info << "Get output." << slog::endl;
                    const ov::Tensor& output_tensor = infer_request.get_output_tensor();
                }
            }

            slog::info << "Average time to fill input buffer: " << input_time_sum / 400 << " micros" << slog::endl;
            slog::info << "Average time for tile inference: " << inference_time_sum / 400 << " micros" << slog::endl;
	   //std::this_thread::sleep_for(std::chrono::milliseconds(1000)); 
        }

        end_experiment(START_TIMER, FLAGS_n);

    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        free(large_img);
        return EXIT_FAILURE;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        free(large_img);
        return EXIT_FAILURE;
    }
    free(large_img);
    return EXIT_SUCCESS;
}
