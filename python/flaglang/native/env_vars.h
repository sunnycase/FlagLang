/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <set>
#include <string>

namespace nncase {
inline const std::set<std::string> CACHE_INVALIDATING_ENV_VARS = {
    // clang-format off
    "AMDGCN_ENABLE_DUMP",
    "AMDGCN_USE_BUFFER_ATOMICS",
    "AMDGCN_USE_BUFFER_OPS",
    "DISABLE_LLVM_OPT",
    "DISABLE_MMA_V3",
    "DISABLE_MMA_V5",
    "DISABLE_PTXAS_OPT",
    "LLVM_IR_ENABLE_DUMP",
    "LLVM_ENABLE_TIMING",
    "LLVM_PASS_PLUGIN_PATH",
    "MLIR_ENABLE_DIAGNOSTICS",
    "MLIR_ENABLE_DUMP",
    "MLIR_DUMP_PATH",
    "MLIR_ENABLE_TIMING",
    "MLIR_DISABLE_MULTITHREADING",
    "TRITON_DEFAULT_FP_FUSION",
    "TRITON_DISABLE_LINE_INFO",
    "TRITON_ENABLE_LLVM_DEBUG",
    "TRITON_HIP_GLOBAL_PREFETCH",
    "TRITON_HIP_LOCAL_PREFETCH",
    "TRITON_HIP_USE_ASYNC_COPY",
    "TRITON_HIP_USE_BLOCK_PINGPONG",
    "TRITON_HIP_USE_IN_THREAD_TRANSPOSE",
    "TRITON_LLVM_DEBUG_ONLY",
    "TRITON_ENABLE_ASAN",
    "TRITON_OVERRIDE_ARCH",
    "USE_IR_LOC",
    "NVPTX_ENABLE_DUMP",
    "ALLOW_LHS_TMEM_LAYOUT_CONVERSION",
    "TRITON_F32_DEFAULT",
    "TRITON_PREFER_TMEM_16x256_LAYOUT",
    // clang-format on
};

inline const std::set<std::string> CACHE_NEUTRAL_ENV_VARS = {
    // clang-format off
    "TRITON_REPRODUCER_PATH",
    "TRITON_ENABLE_PYTHON_STACKTRACE",
    // clang-format on
};
} // namespace nncase
