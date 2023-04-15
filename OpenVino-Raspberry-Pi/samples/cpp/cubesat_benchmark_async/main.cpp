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
#include <queue>

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

using Ms = std::chrono::duration<double, std::ratio<1, 1000>>;

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

        std::condition_variable condVar;
        std::mutex data_mutex;
        std::queue< std::vector<uint8_t> > tile_queue;

        auto producer = [&] {
            for (uint16_t height_offset = 0; height_offset < tot_height; height_offset += tile_size)
            {
                if ((tot_height - height_offset) / tile_size == 0)
                    continue;

                for (uint16_t width_offset = 0; width_offset < tot_width; width_offset += tile_size)
                {
                    if ((tot_width - width_offset) / tile_size == 0)
                        continue;

                    std::vector<uint8_t> tile;
                    tile.resize(tile_size * tile_size * tot_channels);

                    for (uint16_t h = height_offset; h < tile_size + height_offset; h++ ) {
                        uint8_t *tile_ptr = tile.data() + ((h-height_offset) * tile_size * tot_channels);
                        uint8_t *large_img_ptr = large_img + (h * tot_width * tot_channels + width_offset * tot_channels);
                        std::memcpy(tile_ptr, large_img_ptr, sizeof(uint8_t) * tile_size * tot_channels);
                    }
                    // push the tile onto queue
                    std::lock_guard<std::mutex> lock(data_mutex);
                    tile_queue.push(tile);
                }
            }
        };

        // -------- Step 5. Loading model to the device --------
        slog::info << "Loading model to the device MYRIAD" << slog::endl;
        ov::CompiledModel compiled_model = core.compile_model(
            model, 
            "MYRIAD",
            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)
        );
        auto input_port = compiled_model.input();

        // -------- Step 6. Create infer requests --------
        slog::info << "Create infer requests" << slog::endl;

        end_setup();

        start_timer();

        for (int run = 0; run < FLAGS_n; run ++) {
        	uint32_t nireq = compiled_model.get_property(ov::optimal_number_of_infer_requests);
        	slog::info << "Optimal number of requests: " << nireq << slog::endl;

        	std::vector<ov::InferRequest> ireqs(FLAGS_r);
        	std::generate(ireqs.begin(), ireqs.end(), [&] {
            		return compiled_model.create_infer_request();
        	});
            	std::thread producer_thread(producer);

            for(;;) {
                std::unique_lock<std::mutex> lock(data_mutex);
                auto num_tiles = tile_queue.size();
                lock.unlock();
                if (num_tiles >= FLAGS_r) {
                    for (ov::InferRequest& ireq : ireqs) {
                        std::unique_lock<std::mutex> lock(data_mutex);
                        std::vector<uint8_t> tile;
                        tile = tile_queue.front();
                        tile_queue.pop();
                        lock.unlock();
                        ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), tile.data());
                        ireq.set_input_tensor(input_tensor);
                    }
                    break;
                }
            }

            // Warm up
            for (ov::InferRequest& ireq : ireqs) {
                ireq.start_async();
            }
            for (ov::InferRequest& ireq : ireqs) {
                ireq.wait();
            }

            std::vector<double> latencies;
            std::atomic<uint16_t> counter(0);
            std::mutex ireq_mutex;
            std::condition_variable cv;
            std::exception_ptr callback_exception;
            struct TimedIreq {
                ov::InferRequest& ireq;  // ref
                std::chrono::steady_clock::time_point start;
                bool has_start_time;
            };
            std::queue<TimedIreq> finished_ireqs;
            for (ov::InferRequest& ireq : ireqs) {
                finished_ireqs.push({ireq, std::chrono::steady_clock::time_point{}, false});
            }

            counter += FLAGS_r;

            slog::info << "Finished warm-up" << slog::endl;

            for (;;) {
                std::unique_lock<std::mutex> lock(ireq_mutex);
                while (!callback_exception && finished_ireqs.empty()) {
                    cv.wait(lock);
                }
                if (callback_exception) {
                    std::rethrow_exception(callback_exception);
                }
                if (!finished_ireqs.empty()) {
                    auto time_point = std::chrono::steady_clock::now();
                    
                    // retrieve a finished inference request
                    TimedIreq timedIreq = finished_ireqs.front();
                    finished_ireqs.pop();
                    lock.unlock();
                    ov::InferRequest& ireq = timedIreq.ireq;
                    if (timedIreq.has_start_time) {
                        latencies.push_back(std::chrono::duration_cast<Ms>(time_point - timedIreq.start).count());
                    }

                    //check that we still have images to process
                    uint16_t local_counter = ++counter;
                    
                    // make sure to get the last 3 latency measurements as well
                    if ( local_counter > 400) {
                        if (local_counter < 400 + FLAGS_r) {
                            cv.notify_one();
                            continue;
                        } else {
                            break;
                        }
                    };

                    //upon finishing, add the finished request back to queue
                    ireq.set_callback(
                        [&ireq, time_point, &ireq_mutex, &finished_ireqs, &callback_exception, &cv](std::exception_ptr ex) {
                            // Keep callback small. This improves performance for fast (tens of thousands FPS) models
                            std::unique_lock<std::mutex> lock(ireq_mutex);
                            {
                                try {
                                    if (ex) {
                                        std::rethrow_exception(ex);
                                    }
                                    finished_ireqs.push({ireq, time_point, true});
                                } catch (const std::exception&) {
                                    if (!callback_exception) {
                                        callback_exception = std::current_exception();
                                    }
                                }
                            }
                            cv.notify_one();
                        });
                    
                    // check that the thread has some data prepared
                    // perform inference if data in queue
                    for (;;) {
                        std::unique_lock<std::mutex> lock(data_mutex);
                        auto is_empty = tile_queue.empty();
                        if (!is_empty) {
                            std::vector<uint8_t> tile;
                            tile = tile_queue.front();
                            tile_queue.pop();
                            lock.unlock();
                            ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), tile.data());
                            ireq.set_input_tensor(input_tensor);
                            ireq.start_async();
                            break;
                        }
                    }
                }
            }

            producer_thread.join();
            slog::info << "Number of latency data points: " << latencies.size() << slog::endl;
            slog::info << "Average latency per tile: " << std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size() << "ms" << slog::endl;
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
