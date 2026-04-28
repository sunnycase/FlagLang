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
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_triton_interpreter(py::module &&m);
void init_triton_nvidia(py::module &&m);

namespace nncase {
void init_triton_env_vars(py::module &m);
void init_triton_ir(py::module &&m);
void init_ir_builder(py::module &m);
void init_hosting(py::module &&m);
} // namespace nncase
