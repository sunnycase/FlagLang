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
#include "nncase/type.h"
#include <cstdint>
#include <cstring>
#include <map>
#include <nncase/compiler_defs.h>
#include <nncase/llm/paged_attention_config.h>
#include <nncase/runtime/simple_types.h>
#include <nncase/value.h>
#include <string>
#include <string_view>

extern "C" {
typedef void *clr_object_handle_t;
typedef void *nncase_stream_handle_t;

typedef enum {
    nncase_array_rtvalue = 0,
    nncase_array_var = 1,
    nncase_array_object = 2
} nncase_array_element_kind_t;

typedef enum {
    nncase_dimension_kind_fixed = 0,
    nncase_dimension_kind_dynamic = 1,
    nncase_dimension_kind_unknown = 2
} nncase_dimension_kind_t;

typedef enum {
    nncase_mqm_no_quant = 0,
    nncase_mqm_use_ptq = 1,
    nncase_mqm_use_qat = 2
} nncase_model_quant_mode_t;

typedef enum {
    nncase_qm_unsigned = 0,
    nncase_qm_signed_symmetric = 1,
    nncase_qm_signed_asymmetric = 2
} nncase_quant_mode_t;

typedef enum {
    nncase_qt_uint8 = 0,
    nncase_qt_int8 = 1,
    nncase_qt_int16 = 2
} nncase_quant_type_t;

typedef enum {
    nncase_calib_noclip = 0,
    nncase_calib_kld = 1
} nncase_calib_method_t;

typedef enum {
    nncase_no_finetune_weights = 0,
    nncase_finetune_weights_squant = 1,
    nncase_finetune_weights_adaround = 2
} nncase_finetune_weights_method_t;

typedef enum {
    nncase_dump_flags_none = 0,
    nncase_dump_flags_import_ops = 1 << 1,
    nncase_dump_flags_pass_ir = 1 << 2,
    nncase_dump_flags_egraph_cost = 1 << 3,
    nncase_dump_flags_rewrite = 1 << 4,
    nncase_dump_flags_calibration = 1 << 5,
    nncase_dump_flags_evaluator = 1 << 6,
    nncase_dump_flags_compile = 1 << 7,
    nncase_dump_flags_tiling = 1 << 8,
    nncase_dump_flags_schedule = 1 << 9,
    nncase_dump_flags_codegen = 1 << 10
} nncase_dump_flags_t;

typedef enum {
    nncase_it_uint8 = 0,
    nncase_it_int8 = 1,
    nncase_it_float32 = 2
} nncase_input_type_t;

typedef enum {
    nncase_binary_add = 0,
    nncase_binary_sub = 1,
    nncase_binary_mul = 2,
    nncase_binary_div = 3,
    nncase_binary_mod = 4,
    nncase_binary_min = 5,
    nncase_binary_max = 6,
    nncase_binary_pow = 7,
    nncase_binary_bitwise_and = 8,
    nncase_binary_bitwise_or = 9,
    nncase_binary_bitwise_xor = 10,
    nncase_binary_logical_and = 11,
    nncase_binary_logical_or = 12,
    nncase_binary_logical_xor = 13,
    nncase_binary_left_shift = 14,
    nncase_binary_right_shift = 15,
    nncase_binary_floor_div = 16,
    nncase_binary_ceil_div = 17
} nncase_binary_op_t;

typedef enum {
    nncase_cast_default = 0,
    nncase_cast_exact = 1,
    nncase_cast_check_overflow = 2,
    nncase_cast_reinterpret = 3
} nncase_cast_mode_t;

typedef enum {
    nncase_compare_equal = 0,
    nncase_compare_not_equal = 1,
    nncase_compare_lower_than = 2,
    nncase_compare_lower_or_equal = 3,
    nncase_compare_greater_than = 4,
    nncase_compare_greater_or_equal = 5
} nncase_compare_op_t;

enum huggingface_attenion_backend : uint8_t {
    _default = 0,
    paged_attention = 1,
};

typedef enum {
    nncase_padding_option_pad_zero = 1,
    nncase_padding_option_pad_nan = 2
} nncase_padding_option_t;

typedef enum {
    nncase_cache_modifier_none = 1,
    nncase_cache_modifier_ca = 2,
    nncase_cache_modifier_cg = 3,
    nncase_cache_modifier_wb = 4,
    nncase_cache_modifier_cs = 5,
    nncase_cache_modifier_wt = 6
} nncase_cache_modifier_t;

typedef enum {
    nncase_mem_semantic_acquire_release = 4,
    nncase_mem_semantic_acquire = 2,
    nncase_mem_semantic_release = 3,
    nncase_mem_semantic_relaxed = 1
} nncase_mem_semantic_t;

typedef enum {
    nncase_mem_sync_scope_gpu = 1,
    nncase_mem_sync_scope_cta = 2,
    nncase_mem_sync_scope_system = 3
} nncase_mem_sync_scope_t;

typedef enum {
    nncase_eviction_policy_normal = 1,
    nncase_eviction_policy_evict_first = 2,
    nncase_eviction_policy_evict_last = 3
} nncase_eviction_policy_t;

typedef enum {
    nncase_atomic_op_add = 4,
    nncase_atomic_op_fadd = 5,
    nncase_atomic_op_and = 1,
    nncase_atomic_op_or = 2,
    nncase_atomic_op_xor = 3,
    nncase_atomic_op_xchg = 10,
    nncase_atomic_op_max = 6,
    nncase_atomic_op_min = 7,
    nncase_atomic_op_umin = 9,
    nncase_atomic_op_umax = 8
} nncase_atomic_op_t;

typedef enum {
    nncase_rounding_mode_rtz = 0,
    nncase_rounding_mode_rtne = 1
} nncase_rounding_mode_t;

typedef enum {
    nncase_propagate_nan_none = 0,
    nncase_propagate_nan_all = 0xFFFF
} nncase_propagate_nan_t;

typedef enum {
    nncase_input_precision_tf32 = 0,
    nncase_input_precision_tf32x3 = 1,
    nncase_input_precision_ieee = 2
} nncase_input_precision_t;

// clang-format off
/* This block is generated by tools/stackvm_gen/CApiGen at 12/20/2024 3:41:05 PM +08:00. */
enum memory_access_architecture_t : uint8_t {
  memory_access_architecture_uma = 0,
  memory_access_architecture_numa = 1,
};
enum noc_architecture_t : uint8_t {
  noc_architecture_mesh = 0,
  noc_architecture_cross_bar = 1,
};
enum hierarchy_kind_t : uint8_t {
  hierarchy_kind_parallel = 0,
  hierarchy_kind_smt = 1,
};
/* end the auto generated block by tools/stackvm_gen/CApiGen at 12/20/2024 3:41:05 PM +08:00. */
// clang-format on

typedef struct {
    void (*add_ref)(nncase_stream_handle_t handle);
    void (*release)(nncase_stream_handle_t handle);
    bool (*can_read)(nncase_stream_handle_t handle);
    bool (*can_seek)(nncase_stream_handle_t handle);
    bool (*can_write)(nncase_stream_handle_t handle);
    void (*flush)(nncase_stream_handle_t handle);
    int64_t (*get_length)(nncase_stream_handle_t handle);
    int64_t (*set_length)(nncase_stream_handle_t handle, uint64_t value);
    int64_t (*get_position)(nncase_stream_handle_t handle);
    size_t (*read)(nncase_stream_handle_t handle, uint8_t *buffer,
                   size_t length);
    int64_t (*seek)(nncase_stream_handle_t handle, int64_t offset, int origin);
    void (*write)(nncase_stream_handle_t handle, const uint8_t *buffer,
                  size_t length);
} nncase_stream_mt_t;

typedef struct nncase_ir_builder_mt nncase_ir_builder_mt_t;

typedef struct {
    // CLR functions.
    clr_object_handle_t (*array_create)(nncase_array_element_kind_t kind,
                                        const clr_object_handle_t *elements,
                                        size_t count);
    clr_object_handle_t (*array_get_item)(clr_object_handle_t array,
                                          size_t index);
    size_t (*array_get_length)(clr_object_handle_t array);
    void (*handle_dispose)(clr_object_handle_t handle);
    clr_object_handle_t (*handle_duplicate)(clr_object_handle_t handle);
    void (*handle_free)(clr_object_handle_t handle);
    clr_object_handle_t (*stream_create)(const nncase_stream_mt_t *mt,
                                         void *handle);

    // Hosting functions.
    void (*compiler_initialize)();
    clr_object_handle_t (*target_create)(const char *target_name,
                                         size_t target_name_length);

    // Compile functions.
    clr_object_handle_t (*compile_options_create)();
    clr_object_handle_t (*compile_session_create)(
        clr_object_handle_t target, clr_object_handle_t compile_options);
    clr_object_handle_t (*compile_session_create_pass_manager)(
        clr_object_handle_t session, const char *name, size_t name_length);

    bool (*compiler_services_inference_type)(clr_object_handle_t expr);

    void (*pass_manager_add_optimize_ttir)(clr_object_handle_t pass_manager,
                                           int capability);
    clr_object_handle_t (*pass_manager_run)(clr_object_handle_t pass_manager,
                                            clr_object_handle_t module);
    clr_object_handle_t (*ir_module_compile_to_cubin)(
        clr_object_handle_t module, const char *options_json,
        size_t options_json_length);
    size_t (*native_cuda_compile_result_get_json)(clr_object_handle_t result,
                                                  char *buffer,
                                                  size_t buffer_length);
    size_t (*native_cuda_compile_result_get_cubin)(clr_object_handle_t result,
                                                   char *buffer,
                                                   size_t buffer_length);

    // IR functions.
    clr_object_handle_t (*file_location_create)(const char *file_path,
                                                size_t file_path_length,
                                                int line, int column,
                                                int end_line, int end_column);
    clr_object_handle_t (*name_location_create)(
        const char *name, size_t name_length,
        clr_object_handle_t child_location);

    clr_object_handle_t (*ir_module_create)();
    void (*ir_module_add)(clr_object_handle_t module,
                          clr_object_handle_t function);
    clr_object_handle_t (*ir_module_get_function_by_name)(
        clr_object_handle_t module, const char *name, size_t name_length);

    clr_object_handle_t (*data_types_get_boolean)();
    clr_object_handle_t (*data_types_get_int8)();
    clr_object_handle_t (*data_types_get_int16)();
    clr_object_handle_t (*data_types_get_int32)();
    clr_object_handle_t (*data_types_get_int64)();
    clr_object_handle_t (*data_types_get_uint8)();
    clr_object_handle_t (*data_types_get_uint16)();
    clr_object_handle_t (*data_types_get_uint32)();
    clr_object_handle_t (*data_types_get_uint64)();
    clr_object_handle_t (*data_types_get_float16)();
    clr_object_handle_t (*data_types_get_bfloat16)();
    clr_object_handle_t (*data_types_get_float32)();
    clr_object_handle_t (*data_types_get_float64)();
    clr_object_handle_t (*pointer_type_create)(clr_object_handle_t elem_type,
                                               int address_space);

    clr_object_handle_t (*callable_type_create)(
        clr_object_handle_t ret_type, const clr_object_handle_t *param_types,
        size_t param_count);
    clr_object_handle_t (*void_type_get)();
    clr_object_handle_t (*tuple_type_create)(
        const clr_object_handle_t *field_types, size_t field_count);
    clr_object_handle_t (*tensor_type_create)(clr_object_handle_t element_type,
                                              clr_object_handle_t shape);
    clr_object_handle_t (*tensor_type_get_shape)(
        clr_object_handle_t tensor_type);

    clr_object_handle_t (*shape_create_fixed)(const int64_t *dims,
                                              size_t dim_count);

    void (*base_expr_set_int32_attribute)(clr_object_handle_t expr,
                                          const char *name, size_t name_length,
                                          int value);

    clr_object_handle_t (*base_expr_get_location)(clr_object_handle_t expr);
    void (*base_expr_set_location)(clr_object_handle_t expr,
                                   clr_object_handle_t location);
    clr_object_handle_t (*base_expr_get_shape)(clr_object_handle_t expr);

    clr_object_handle_t (*program_id)(int axis);
    clr_object_handle_t (*scalar_int32)(int value);
    clr_object_handle_t (*scalar_int64)(long value);
    clr_object_handle_t (*scalar_float16)(float value);
    clr_object_handle_t (*scalar_float32)(float value);
    clr_object_handle_t (*scalar_float64)(double value);
    clr_object_handle_t (*cast)(clr_object_handle_t value,
                                clr_object_handle_t target_type,
                                nncase_cast_mode_t mode);

    clr_object_handle_t (*binary_op)(nncase_binary_op_t op,
                                     clr_object_handle_t a,
                                     clr_object_handle_t b);
    clr_object_handle_t (*math_compare)(nncase_compare_op_t op,
                                        clr_object_handle_t a,
                                        clr_object_handle_t b);

    clr_object_handle_t (*shapes_as_tensor)(clr_object_handle_t dimension);

    clr_object_handle_t (*tensors_broadcast)(clr_object_handle_t value,
                                             clr_object_handle_t shape);
    clr_object_handle_t (*tensors_range)(clr_object_handle_t begin,
                                         clr_object_handle_t end,
                                         clr_object_handle_t step);

    clr_object_handle_t (*tir_return)(const clr_object_handle_t *value_ptrs,
                                      size_t value_count);

    clr_object_handle_t (*call_create)(clr_object_handle_t target,
                                       const clr_object_handle_t *arg_ptrs,
                                       size_t arg_count);
    size_t (*call_get_num_results)(clr_object_handle_t call);
    clr_object_handle_t (*call_get_result)(clr_object_handle_t call,
                                           size_t index);

    clr_object_handle_t (*triton_load)(
        clr_object_handle_t ptr, clr_object_handle_t mask,
        clr_object_handle_t other, nncase_cache_modifier_t cache_modifier,
        nncase_eviction_policy_t eviction_policy);
    clr_object_handle_t (*triton_store)(
        clr_object_handle_t ptr, clr_object_handle_t value,
        clr_object_handle_t mask, nncase_cache_modifier_t cache_modifier,
        nncase_eviction_policy_t eviction_policy);

    size_t (*sequential_get_parameters_count)(clr_object_handle_t block);
    size_t (*sequential_get_fields_count)(clr_object_handle_t block);
    clr_object_handle_t (*sequential_get_parameter)(clr_object_handle_t block,
                                                    size_t index);
    bool (*sequential_has_terminator)(clr_object_handle_t block);
    void (*sequential_insert_at)(clr_object_handle_t block, int index,
                                 clr_object_handle_t expr);

    clr_object_handle_t (*prim_function_create)(
        const char *name, size_t name_length,
        clr_object_handle_t callable_type);
    clr_object_handle_t (*prim_function_add_body)(clr_object_handle_t function);
    clr_object_handle_t (*prim_function_get_body)(clr_object_handle_t function);

    clr_object_handle_t (*tuple_create)(const clr_object_handle_t *field_ptrs,
                                        size_t field_count);

    size_t (*base_expr_print)(clr_object_handle_t expr, char *buffer,
                              size_t buffer_length);
    size_t (*ir_module_get_entry_name)(clr_object_handle_t module,
                                       char *buffer, size_t buffer_length);
    size_t (*ir_module_describe_vector_add)(clr_object_handle_t module,
                                            char *buffer,
                                            size_t buffer_length);
} nncase_api_mt_t;

NNCASE_API nncase_api_mt_t *nncase_clr_api();
NNCASE_API int nncase_clr_initialize(const char *root_assembly_path);
NNCASE_API int nncase_clr_uninitialize();
}

DEFINE_ENUM_BITMASK_OPERATORS(nncase_dump_flags_t)

namespace nncase::clr {
class shape;

class clr_object_ptr {
  public:
    constexpr clr_object_ptr(std::nullptr_t = nullptr) noexcept
        : handle_(nullptr) {}
    constexpr clr_object_ptr(clr_object_handle_t handle) noexcept
        : handle_(handle) {}
    constexpr clr_object_ptr(clr_object_ptr &&other) noexcept
        : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    clr_object_ptr(const clr_object_ptr &other) noexcept
        : handle_(nncase_clr_api()->handle_duplicate(other.handle_)) {}

    ~clr_object_ptr() { release(); }

    clr_object_ptr &operator=(const clr_object_ptr &other) noexcept {
        if (this != &other) {
            release();
            handle_ = nncase_clr_api()->handle_duplicate(other.handle_);
        }
        return *this;
    }

    clr_object_ptr &operator=(clr_object_ptr &&other) noexcept {
        release();
        handle_ = other.handle_;
        other.handle_ = nullptr;
        return *this;
    }

    clr_object_handle_t get() const noexcept { return handle_; }

    clr_object_handle_t detach() noexcept {
        auto handle = handle_;
        handle_ = nullptr;
        return handle;
    }

    clr_object_handle_t *release_and_addressof() noexcept {
        release();
        return &handle_;
    }

  private:
    void release() {
        if (auto handle = handle_) {
            handle_ = nullptr;
            nncase_clr_api()->handle_free(handle);
        }
    }

  private:
    clr_object_handle_t handle_;
};

#define CHECK_CLR(x) x

class clr_object_base {
  public:
    constexpr clr_object_base(std::nullptr_t = nullptr) noexcept
        : obj_(nullptr) {}

    clr_object_base(std::in_place_t, clr_object_ptr ptr) noexcept
        : obj_(std::move(ptr)) {}

    clr_object_base(clr_object_base &&) = default;
    clr_object_base &operator=(clr_object_base &&) = default;

    clr_object_base(const clr_object_base &other) noexcept = default;
    clr_object_base &operator=(const clr_object_base &other) noexcept = default;

    bool empty() const noexcept { return !obj_.get(); }

    clr_object_handle_t get() const noexcept { return obj_.get(); }
    clr_object_handle_t *release_and_addressof() noexcept {
        return obj_.release_and_addressof();
    }

    template <class T, std::enable_if_t<std::is_base_of_v<clr_object_base, T>>>
    T &cast() noexcept {
        return static_cast<T &>(*this);
    }

  protected:
    clr_object_ptr obj_;
};

class array : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    array(nncase_array_element_kind_t kind, const clr_object_handle_t *elements,
          size_t length) {
        obj_ = nncase_clr_api()->array_create(kind, elements, length);
    }

    template <class T = clr_object_base> T at(size_t index) {
        return {std::in_place,
                nncase_clr_api()->array_get_item(obj_.get(), index)};
    }

    size_t length() { return nncase_clr_api()->array_get_length(obj_.get()); }

    template <class T = clr_object_base> std::vector<T> to_vector() {
        std::vector<T> vector(length());
        for (size_t i = 0; i < vector.size(); i++) {
            vector[i] = at<T>(i);
        }
        return vector;
    }
};

class cstream : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    cstream(const nncase_stream_mt_t *mt, void *handle) {
        obj_ = nncase_clr_api()->stream_create(mt, handle);
    }

    ~cstream() { nncase_clr_api()->handle_dispose(obj_.get()); }
};

class target : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    target(std::string_view name) {
        obj_ = nncase_clr_api()->target_create(name.data(), name.length());
    }
};

class compile_options : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    compile_options() { obj_ = nncase_clr_api()->compile_options_create(); }
};

class compile_session : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    compile_session(const target &target, const compile_options &options) {
        obj_ = nncase_clr_api()->compile_session_create(target.get(),
                                                        options.get());
    }
};

class location : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;
};

class file_location : public location {
  public:
    using location::location;

    file_location(std::string_view file_path, int line, int column,
                  int end_line, int end_column) {
        obj_ = nncase_clr_api()->file_location_create(
            file_path.data(), file_path.length(), line, column, end_line,
            end_column);
    }

    file_location(std::string_view file_path, int line, int column)
        : file_location(file_path, line, column, line, column) {}
};

class name_location : public location {
  public:
    using location::location;

    name_location(std::string_view name,
                  std::optional<location> childLoc = std::nullopt) {
        obj_ = nncase_clr_api()->name_location_create(
            name.data(), name.length(), childLoc ? childLoc->get() : nullptr);
    }
};

class ir_type : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;
};

class datatype : public ir_type {
  public:
    using ir_type::ir_type;

    static datatype boolean() {
        return {std::in_place, nncase_clr_api()->data_types_get_boolean()};
    }
    static datatype int8() {
        return {std::in_place, nncase_clr_api()->data_types_get_int8()};
    }
    static datatype int16() {
        return {std::in_place, nncase_clr_api()->data_types_get_int16()};
    }
    static datatype int32() {
        return {std::in_place, nncase_clr_api()->data_types_get_int32()};
    }
    static datatype int64() {
        return {std::in_place, nncase_clr_api()->data_types_get_int64()};
    }
    static datatype uint8() {
        return {std::in_place, nncase_clr_api()->data_types_get_uint8()};
    }
    static datatype uint16() {
        return {std::in_place, nncase_clr_api()->data_types_get_uint16()};
    }
    static datatype uint32() {
        return {std::in_place, nncase_clr_api()->data_types_get_uint32()};
    }
    static datatype uint64() {
        return {std::in_place, nncase_clr_api()->data_types_get_uint64()};
    }
    static datatype float16() {
        return {std::in_place, nncase_clr_api()->data_types_get_float16()};
    }
    static datatype bfloat16() {
        return {std::in_place, nncase_clr_api()->data_types_get_bfloat16()};
    }
    static datatype float32() {
        return {std::in_place, nncase_clr_api()->data_types_get_float32()};
    }
    static datatype float64() {
        return {std::in_place, nncase_clr_api()->data_types_get_float64()};
    }
};

class pointer_type : public datatype {
  public:
    using datatype::datatype;

    pointer_type(datatype elem_type, int address_space = 0) {
        obj_ = nncase_clr_api()->pointer_type_create(elem_type.get(),
                                                     address_space);
    }
};

class callable_type : public ir_type {
  public:
    using ir_type::ir_type;

    callable_type(ir_type ret_type, const std::vector<ir_type> &param_types) {
        std::vector<clr_object_handle_t> param_type_handles;
        param_type_handles.reserve(param_types.size());
        for (const auto &pt : param_types) {
            param_type_handles.push_back(pt.get());
        }
        obj_ = nncase_clr_api()->callable_type_create(
            ret_type.get(), param_type_handles.data(),
            param_type_handles.size());
    }
};

class tuple_type : public ir_type {
  public:
    using ir_type::ir_type;

    tuple_type(const std::vector<ir_type> &field_types) {
        std::vector<clr_object_handle_t> field_type_handles;
        field_type_handles.reserve(field_types.size());
        for (const auto &ft : field_types) {
            field_type_handles.push_back(ft.get());
        }
        obj_ = nncase_clr_api()->tuple_type_create(field_type_handles.data(),
                                                   field_type_handles.size());
    }

    static tuple_type void_type() {
        return {std::in_place, nncase_clr_api()->void_type_get()};
    }
};

class tensor_type : public ir_type {
  public:
    using ir_type::ir_type;

    NNCASE_API tensor_type(datatype element_type, clr::shape shape);

    NNCASE_API clr::shape shape();
};

class expr : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    std::string to_text() const {
        auto length = nncase_clr_api()->base_expr_print(obj_.get(), nullptr, 0);
        std::string text(length, '\0');
        if (length != 0) {
            nncase_clr_api()->base_expr_print(obj_.get(), text.data(),
                                              text.size());
        }
        return text;
    }

    void set_int32_attribute(std::string_view name, int value) {
        nncase_clr_api()->base_expr_set_int32_attribute(obj_.get(), name.data(),
                                                        name.length(), value);
    }

    location get_location() {
        return {std::in_place,
                nncase_clr_api()->base_expr_get_location(obj_.get())};
    }

    void set_location(const location &loc) {
        nncase_clr_api()->base_expr_set_location(obj_.get(), loc.get());
    }

    NNCASE_API shape get_shape();
};

class ir_tuple : public expr {
  public:
    using expr::expr;

    static ir_tuple create(const std::span<expr> &fields) {
        std::vector<clr_object_handle_t> field_handles;
        field_handles.reserve(fields.size());
        for (const auto &f : fields) {
            field_handles.push_back(f.get());
        }
        return {std::in_place, nncase_clr_api()->tuple_create(
                                   field_handles.data(), field_handles.size())};
    }
};

class call : public expr {
  public:
    using expr::expr;

    call(expr target, const std::vector<expr> &args) {
        std::vector<clr_object_handle_t> arg_handles;
        arg_handles.reserve(args.size());
        for (const auto &arg : args) {
            arg_handles.push_back(arg.get());
        }
        obj_ = nncase_clr_api()->call_create(
            target.get(), arg_handles.data(), arg_handles.size());
    }

    size_t get_num_results() {
        return nncase_clr_api()->call_get_num_results(obj_.get());
    }

    expr get_result(size_t index) {
        return {std::in_place,
                nncase_clr_api()->call_get_result(obj_.get(), index)};
    }
};

class dimension : public expr {
  public:
    using expr::expr;
};

class shape : public expr {
  public:
    using expr::expr;

    static shape fixed(const std::vector<int64_t> &dims) {
        return {std::in_place,
                nncase_clr_api()->shape_create_fixed(dims.data(), dims.size())};
    }
};

class var : public expr {
  public:
    using expr::expr;
};

class program_id_dim : public dimension {
  public:
    using dimension::dimension;

    program_id_dim(int axis) { obj_ = nncase_clr_api()->program_id(axis); }
};

class tensor_const : public expr {
  public:
    using expr::expr;

    static tensor_const scalar(int value) {
        return {std::in_place, nncase_clr_api()->scalar_int32(value)};
    }

    static tensor_const scalar(long value) {
        return {std::in_place, nncase_clr_api()->scalar_int64(value)};
    }

    static tensor_const scalar_float16(float value) {
        return {std::in_place, nncase_clr_api()->scalar_float16(value)};
    }

    static tensor_const scalar_float32(float value) {
        return {std::in_place, nncase_clr_api()->scalar_float32(value)};
    }

    static tensor_const scalar_float64(double value) {
        return {std::in_place, nncase_clr_api()->scalar_float64(value)};
    }
};

struct ir_builder {
    static expr cast(expr value, ir_type target_type,
                     nncase_cast_mode_t mode = nncase_cast_default) {
        return {std::in_place,
                nncase_clr_api()->cast(value.get(), target_type.get(), mode)};
    }

    struct math {
        static expr binary(nncase_binary_op_t op, expr a, expr b) {
            return {std::in_place,
                    nncase_clr_api()->binary_op(op, a.get(), b.get())};
        }

        static expr compare(nncase_compare_op_t op, expr a, expr b) {
            return {std::in_place,
                    nncase_clr_api()->math_compare(op, a.get(), b.get())};
        }
    };

    struct shapes {
        static expr as_tensor(dimension dimension) {
            return {std::in_place,
                    nncase_clr_api()->shapes_as_tensor(dimension.get())};
        }
    };

    struct tensors {
        static expr broadcast(expr value, clr::shape shape) {
            return {std::in_place, nncase_clr_api()->tensors_broadcast(
                                       value.get(), shape.get())};
        }

        static expr range(expr begin, expr end, expr step) {
            return {std::in_place, nncase_clr_api()->tensors_range(
                                       begin.get(), end.get(), step.get())};
        }
    };

    struct tir {
        static expr return_(const std::span<expr> &values) {
            std::vector<clr_object_handle_t> value_handles;
            value_handles.reserve(values.size());
            for (const auto &v : values) {
                value_handles.push_back(v.get());
            }
            return {std::in_place,
                    nncase_clr_api()->tir_return(value_handles.data(),
                                                 value_handles.size())};
        }
    };

    struct triton {
        static expr load(
            expr ptr, std::optional<expr> mask = std::nullopt,
            std::optional<expr> other = std::nullopt,
            nncase_cache_modifier_t cache_modifier = nncase_cache_modifier_none,
            nncase_eviction_policy_t eviction_policy =
                nncase_eviction_policy_normal) {
            return {std::in_place, nncase_clr_api()->triton_load(
                                       ptr.get(), mask ? mask->get() : nullptr,
                                       other ? other->get() : nullptr,
                                       cache_modifier, eviction_policy)};
        }

        static expr store(
            expr ptr, expr value, std::optional<expr> mask = std::nullopt,
            nncase_cache_modifier_t cache_modifier = nncase_cache_modifier_none,
            nncase_eviction_policy_t eviction_policy =
                nncase_eviction_policy_normal) {
            return {std::in_place,
                    nncase_clr_api()->triton_store(
                        ptr.get(), value.get(), mask ? mask->get() : nullptr,
                        cache_modifier, eviction_policy)};
        }
    };
};

class sequential : public expr {
  public:
    using expr::expr;

    size_t parameters_count() {
        return nncase_clr_api()->sequential_get_parameters_count(obj_.get());
    }

    size_t fields_count() {
        return nncase_clr_api()->sequential_get_fields_count(obj_.get());
    }

    var get_parameter(size_t index) {
        return {std::in_place,
                nncase_clr_api()->sequential_get_parameter(obj_.get(), index)};
    }

    bool has_terminator() {
        return nncase_clr_api()->sequential_has_terminator(obj_.get());
    }

    void insert_at(int index, expr expr) {
        nncase_clr_api()->sequential_insert_at(obj_.get(), index, expr.get());
    }
};

class base_function : public expr {
  public:
    using expr::expr;
};

class prim_function : public base_function {
  public:
    using base_function::base_function;

    prim_function(std::string_view name, callable_type type) {
        obj_ = nncase_clr_api()->prim_function_create(
            name.data(), name.length(), type.get());
    }

    sequential add_body() {
        return {std::in_place,
                nncase_clr_api()->prim_function_add_body(obj_.get())};
    }

    sequential get_body() {
        return {std::in_place,
                nncase_clr_api()->prim_function_get_body(obj_.get())};
    }
};

class ir_module : public expr {
  public:
    using expr::expr;

    ir_module() { obj_ = nncase_clr_api()->ir_module_create(); }

    void add(base_function func) {
        nncase_clr_api()->ir_module_add(obj_.get(), func.get());
    }

    prim_function get_function_by_name(std::string_view name) {
        return {std::in_place, nncase_clr_api()->ir_module_get_function_by_name(
                                   obj_.get(), name.data(), name.length())};
    }

    bool has_function(std::string_view name) {
        return !get_function_by_name(name).empty();
    }

    std::string get_entry_func_name() const {
        auto length =
            nncase_clr_api()->ir_module_get_entry_name(obj_.get(), nullptr, 0);
        std::string text(length, '\0');
        if (length != 0) {
            nncase_clr_api()->ir_module_get_entry_name(obj_.get(), text.data(),
                                                       text.size());
        }
        return text;
    }

    std::string describe_vector_add() const {
        auto length = nncase_clr_api()->ir_module_describe_vector_add(
            obj_.get(), nullptr, 0);
        std::string text(length, '\0');
        if (length != 0) {
            nncase_clr_api()->ir_module_describe_vector_add(
                obj_.get(), text.data(), text.size());
        }
        return text;
    }
};

class native_cuda_compile_result : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    native_cuda_compile_result(ir_module module, std::string_view options_json) {
        obj_ = nncase_clr_api()->ir_module_compile_to_cubin(
            module.get(), options_json.data(), options_json.length());
    }

    std::string metadata_json() const {
        auto length = nncase_clr_api()->native_cuda_compile_result_get_json(
            obj_.get(), nullptr, 0);
        std::string text(length, '\0');
        if (length != 0) {
            nncase_clr_api()->native_cuda_compile_result_get_json(
                obj_.get(), text.data(), text.size());
        }
        return text;
    }

    std::string cubin() const {
        auto length = nncase_clr_api()->native_cuda_compile_result_get_cubin(
            obj_.get(), nullptr, 0);
        std::string bytes(length, '\0');
        if (length != 0) {
            nncase_clr_api()->native_cuda_compile_result_get_cubin(
                obj_.get(), bytes.data(), bytes.size());
        }
        return bytes;
    }
};

struct compiler_services {
    static bool inference_type(expr expr) {
        return nncase_clr_api()->compiler_services_inference_type(expr.get());
    }
};

class pass_manager : public clr_object_base {
  public:
    using clr_object_base::clr_object_base;

    pass_manager(compile_session &session, std::string_view name) {
        obj_ = nncase_clr_api()->compile_session_create_pass_manager(
            session.get(), name.data(), name.length());
    }

    void add_optimize_ttir(int capability) {
        nncase_clr_api()->pass_manager_add_optimize_ttir(obj_.get(),
                                                         capability);
    }

    void run(ir_module &module) {
        module = {std::in_place,
                  nncase_clr_api()->pass_manager_run(obj_.get(), module.get())};
    }
};
} // namespace nncase::clr
