/* Copyright SunnyCase.
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
#include <nncase/compiler.h>

using namespace nncase;

void nncase::init_triton_ir(py::module &&m) {
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    py::enum_<nncase_padding_option_t>(m, "PADDING_OPTION",
                                       py::module_local())
        .value("PAD_ZERO", nncase_padding_option_pad_zero)
        .value("PAD_NAN", nncase_padding_option_pad_nan)
        .export_values();

    py::enum_<nncase_cache_modifier_t>(m, "CACHE_MODIFIER",
                                       py::module_local())
        .value("NONE", nncase_cache_modifier_none)
        .value("CA", nncase_cache_modifier_ca)
        .value("CG", nncase_cache_modifier_cg)
        .value("WB", nncase_cache_modifier_wb)
        .value("CS", nncase_cache_modifier_cs)
        .value("WT", nncase_cache_modifier_wt)
        .export_values();

    py::enum_<nncase_mem_semantic_t>(m, "MEM_SEMANTIC",
                                     py::module_local())
        .value("ACQUIRE_RELEASE", nncase_mem_semantic_acquire_release)
        .value("ACQUIRE", nncase_mem_semantic_acquire)
        .value("RELEASE", nncase_mem_semantic_release)
        .value("RELAXED", nncase_mem_semantic_relaxed)
        .export_values();

    py::enum_<nncase_mem_sync_scope_t>(m, "MEM_SYNC_SCOPE",
                                       py::module_local())
        .value("GPU", nncase_mem_sync_scope_gpu)
        .value("CTA", nncase_mem_sync_scope_cta)
        .value("SYSTEM", nncase_mem_sync_scope_system)
        .export_values();

    py::enum_<nncase_eviction_policy_t>(m, "EVICTION_POLICY",
                                        py::module_local())
        .value("NORMAL", nncase_eviction_policy_normal)
        .value("EVICT_FIRST", nncase_eviction_policy_evict_first)
        .value("EVICT_LAST", nncase_eviction_policy_evict_last)
        .export_values();

    py::enum_<nncase_atomic_op_t>(m, "ATOMIC_OP", py::module_local())
        .value("ADD", nncase_atomic_op_add)
        .value("FADD", nncase_atomic_op_fadd)
        .value("AND", nncase_atomic_op_and)
        .value("OR", nncase_atomic_op_or)
        .value("XOR", nncase_atomic_op_xor)
        .value("XCHG", nncase_atomic_op_xchg)
        .value("MAX", nncase_atomic_op_max)
        .value("MIN", nncase_atomic_op_min)
        .value("UMIN", nncase_atomic_op_umin)
        .value("UMAX", nncase_atomic_op_umax);

    py::enum_<nncase_rounding_mode_t>(m, "ROUNDING_MODE",
                                      py::module_local())
        .value("RTZ", nncase_rounding_mode_rtz)
        .value("RTNE", nncase_rounding_mode_rtne);

    py::enum_<nncase_propagate_nan_t>(m, "PROPAGATE_NAN",
                                      py::module_local())
        .value("NONE", nncase_propagate_nan_none)
        .value("ALL", nncase_propagate_nan_all);

    py::enum_<nncase_input_precision_t>(m, "INPUT_PRECISION",
                                        py::module_local())
        .value("TF32", nncase_input_precision_tf32)
        .value("TF32x3", nncase_input_precision_tf32x3)
        .value("IEEE", nncase_input_precision_ieee)
        .export_values();
}
