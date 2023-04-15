ie_add_sample(NAME cubesat_benchmark
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/benchmark.h"
              DEPENDENCIES ${GFLAGS_TARGET} format_reader ie_samples_utils)
