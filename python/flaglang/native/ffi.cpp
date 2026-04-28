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
#include "ffi_modules.h"
#include "pytype_utils.h"
#include "runtime_tensor.h"
#include "type_casters.h"
#include <nncase/compiler.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/version.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace nncase;
using namespace nncase::clr;
using namespace nncase::runtime;

namespace {} // namespace

namespace pybind11::detail {
std::atomic_bool g_python_shutdown = false;
}

PYBIND11_MODULE(libtriton, m) {
    m.doc() = "FlagLang";
    m.attr("__version__") = NNCASE_VERSION NNCASE_VERSION_SUFFIX;

    m.add_object("_cleanup", py::capsule([]() {
                     nncase_clr_uninitialize();
                     pybind11::detail::g_python_shutdown.store(
                         true, std::memory_order_release);
                 }));

    py::enum_<nncase_dump_flags_t>(m, "DumpFlags", py::arithmetic())
        .value("Nothing", nncase_dump_flags_none)
        .value("ImportOps", nncase_dump_flags_import_ops)
        .value("PassIR", nncase_dump_flags_pass_ir)
        .value("EGraphCost", nncase_dump_flags_egraph_cost)
        .value("Rewrite", nncase_dump_flags_rewrite)
        .value("Calibration", nncase_dump_flags_calibration)
        .value("Evaluator", nncase_dump_flags_evaluator)
        .value("Compile", nncase_dump_flags_compile)
        .value("Tiling", nncase_dump_flags_tiling)
        .value("Schedule", nncase_dump_flags_schedule)
        .value("CodeGen", nncase_dump_flags_codegen);

    py::enum_<nncase_dimension_kind_t>(m, "DimensionKind")
        .value("Fixed", nncase_dimension_kind_fixed)
        .value("Dynamic", nncase_dimension_kind_dynamic)
        .value("Unknown", nncase_dimension_kind_unknown);

    register_runtime_tensor(m);
    init_triton_env_vars(m);
    init_triton_ir(m.def_submodule("ir"));
    init_hosting(m.def_submodule("hosting"));
    init_triton_interpreter(m.def_submodule("interpreter"));
    init_triton_nvidia(m.def_submodule("nvidia"));
}
