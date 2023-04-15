// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "gflags/gflags.h"

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model.";

/// @brief message for model argument
static const char nruns_message[] = "Number of runs of the benchmark";

/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);

/// @brief Define parameter for number of runs <br>
DEFINE_int32(n, 10, nruns_message);



/**
 * @brief This function show a help message
 */
static void show_usage() {
    std::cout << std::endl;
    std::cout << "cubesat_benchmark [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                      " << help_message << std::endl;
    std::cout << "    -m \"<path>\"             " << model_message << std::endl;
    std::cout << "    -n \"<number>\"           " << nruns_message << std::endl;
}

