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
#include <cctype>
#include <fstream>
#include <nncase/compiler.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <regex>
#include <sstream>
#include <stdexcept>

using namespace nncase;

namespace {
struct insertion_point_snapshot {
    clr::sequential block;
    size_t index = 0;
};

struct mlir_source_function {
    std::string name;
    std::vector<std::string> signature;
};

std::string trim(std::string_view text) {
    auto first = text.begin();
    auto last = text.end();
    while (first != last &&
           std::isspace(static_cast<unsigned char>(*first)) != 0) {
        ++first;
    }

    while (first != last &&
           std::isspace(static_cast<unsigned char>(*(last - 1))) != 0) {
        --last;
    }

    return std::string(first, last);
}

std::string read_text_file(const std::string &path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open MLIR source file: " + path);
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

size_t find_matching_paren(const std::string &text, size_t open_pos) {
    size_t depth = 0;
    for (size_t i = open_pos; i < text.size(); i++) {
        if (text[i] == '(') {
            depth++;
        } else if (text[i] == ')') {
            if (depth == 0) {
                break;
            }
            depth--;
            if (depth == 0) {
                return i;
            }
        }
    }

    throw std::runtime_error("Malformed MLIR function signature: unmatched '('");
}

std::vector<std::string> split_top_level_commas(const std::string &text) {
    std::vector<std::string> parts;
    size_t start = 0;
    int paren_depth = 0;
    int bracket_depth = 0;
    int brace_depth = 0;
    int angle_depth = 0;
    for (size_t i = 0; i < text.size(); i++) {
        const auto ch = text[i];
        switch (ch) {
        case '(':
            paren_depth++;
            break;
        case ')':
            paren_depth--;
            break;
        case '[':
            bracket_depth++;
            break;
        case ']':
            bracket_depth--;
            break;
        case '{':
            brace_depth++;
            break;
        case '}':
            brace_depth--;
            break;
        case '<':
            angle_depth++;
            break;
        case '>':
            angle_depth--;
            break;
        case ',':
            if (paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 &&
                angle_depth == 0) {
                auto part = trim(std::string_view(text).substr(start, i - start));
                if (!part.empty()) {
                    parts.push_back(std::move(part));
                }
                start = i + 1;
            }
            break;
        default:
            break;
        }
    }

    auto tail = trim(std::string_view(text).substr(start));
    if (!tail.empty()) {
        parts.push_back(std::move(tail));
    }
    return parts;
}

std::string strip_top_level_attrs(const std::string &text) {
    int angle_depth = 0;
    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == '<') {
            angle_depth++;
        } else if (text[i] == '>') {
            angle_depth--;
        } else if (text[i] == '{' && angle_depth == 0) {
            return trim(std::string_view(text).substr(0, i));
        }
    }

    return trim(text);
}

std::string convert_mlir_type(std::string type_text) {
    type_text = strip_top_level_attrs(type_text);

    std::smatch ptr_match;
    static const std::regex ptr_pattern(R"(!tt\.ptr<\s*([^,>]+))");
    if (std::regex_search(type_text, ptr_match, ptr_pattern)) {
        return "*" + convert_mlir_type(trim(ptr_match[1].str()));
    }

    return type_text;
}

std::string convert_mlir_argument_type(const std::string &arg_text) {
    if (arg_text.find("tt.nv_tma_desc") != std::string::npos) {
        return "nvTmaDesc";
    }

    auto colon = arg_text.find(':');
    if (colon == std::string::npos) {
        throw std::runtime_error("Malformed MLIR function argument: " + arg_text);
    }

    return convert_mlir_type(arg_text.substr(colon + 1));
}

std::string normalize_symbol_name(std::string_view name) {
    auto text = trim(name);
    if (!text.empty() && text[0] == '@') {
        text.erase(text.begin());
    }
    return text;
}

std::string escape_regex(std::string_view text) {
    std::string escaped;
    escaped.reserve(text.size() * 2);
    for (auto ch : text) {
        if (std::string_view(R"(\.^$|()[]{}*+?)").find(ch) !=
            std::string_view::npos) {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    return escaped;
}

class mlir_source_module {
  public:
    mlir_source_module(std::string path, std::string source)
        : path_(std::move(path)), source_(std::move(source)) {
        parse_functions();
    }

    const std::string &get_entry_func_name() const {
        if (functions_.empty()) {
            throw std::runtime_error("MLIR source does not contain a tt.func.");
        }
        return functions_.front().name;
    }

    mlir_source_function get_function(std::string_view name) const {
        const auto normalized = normalize_symbol_name(name);
        for (const auto &function : functions_) {
            if (function.name == normalized) {
                return function;
            }
        }

        throw std::runtime_error("MLIR source does not contain function: " +
                                 normalized);
    }

    std::vector<std::string>
    get_function_signature(const mlir_source_function &function) const {
        return function.signature;
    }

    std::optional<int64_t> get_int_attr(std::string_view name) const {
        const auto escaped_name = escape_regex(name);
        const std::regex attr_pattern("\"?" + escaped_name +
                                      "\"?\\s*=\\s*(-?\\d+)\\s*:\\s*i\\d+");
        std::smatch match;
        if (!std::regex_search(source_, match, attr_pattern)) {
            return std::nullopt;
        }

        return std::stoll(match[1].str());
    }

    const std::string &to_text() const { return source_; }
    const std::string &str_nodebug() const { return source_; }
    bool verify() const { return !functions_.empty(); }
    void create_location_snapshot([[maybe_unused]] std::string_view path) const {}

  private:
    void parse_functions() {
        static const std::regex function_pattern(
            R"(\b(?:tt|func)\.func\s+(?:public\s+)?@([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\()");
        auto begin =
            std::sregex_iterator(source_.begin(), source_.end(), function_pattern);
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            const auto &match = *it;
            const auto open_pos =
                static_cast<size_t>(match.position(0) + match.length(0) - 1);
            const auto close_pos = find_matching_paren(source_, open_pos);
            const auto args_text =
                source_.substr(open_pos + 1, close_pos - open_pos - 1);

            std::vector<std::string> signature;
            for (const auto &arg : split_top_level_commas(args_text)) {
                signature.push_back(convert_mlir_argument_type(arg));
            }

            mlir_source_function function;
            function.name = match[1].str();
            function.signature = std::move(signature);
            functions_.push_back(std::move(function));
        }

        if (functions_.empty()) {
            throw std::runtime_error("MLIR source does not contain a tt.func.");
        }
    }

    std::string path_;
    std::string source_;
    std::vector<mlir_source_function> functions_;
};

class triton_op_builder {
  public:
    triton_op_builder(clr::compile_session session) : session_(session) {}

    const clr::location &last_location() const { return last_location_; }
    void set_last_location(clr::location loc) {
        last_location_ = std::move(loc);
    }

    clr::ir_module create_module() { return clr::ir_module(); }

    std::optional<clr::sequential> get_insertion_block() {
        if (insertion_point_.block.empty()) {
            return std::nullopt;
        }

        return insertion_point_.block;
    }

    insertion_point_snapshot get_insertion_point() const {
        return insertion_point_;
    }

    void restore_insertion_point(insertion_point_snapshot point) {
        insertion_point_ = std::move(point);
    }

    void set_insertion_point_to_start(clr::sequential block) {
        insertion_point_.block = std::move(block);
        insertion_point_.index = 0;
    }

    void set_insertion_point_to_end(clr::sequential block) {
        insertion_point_.index = block.fields_count(); // after last
        insertion_point_.block = std::move(block);
    }

    clr::expr insert_expr(clr::expr expr) {
        insertion_point_.block.insert_at(insertion_point_.index++, expr);
        return expr;
    }

  private:
    clr::compile_session session_;
    clr::location last_location_;
    insertion_point_snapshot insertion_point_;
};
} // namespace

void nncase::init_triton_ir(py::module &&m) {
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    py::enum_<nncase_padding_option_t>(m, "PADDING_OPTION", py::module_local())
        .value("PAD_ZERO", nncase_padding_option_pad_zero)
        .value("PAD_NAN", nncase_padding_option_pad_nan)
        .export_values();

    py::enum_<nncase_cache_modifier_t>(m, "CACHE_MODIFIER", py::module_local())
        .value("NONE", nncase_cache_modifier_none)
        .value("CA", nncase_cache_modifier_ca)
        .value("CG", nncase_cache_modifier_cg)
        .value("WB", nncase_cache_modifier_wb)
        .value("CS", nncase_cache_modifier_cs)
        .value("WT", nncase_cache_modifier_wt)
        .export_values();

    py::enum_<nncase_mem_semantic_t>(m, "MEM_SEMANTIC", py::module_local())
        .value("ACQUIRE_RELEASE", nncase_mem_semantic_acquire_release)
        .value("ACQUIRE", nncase_mem_semantic_acquire)
        .value("RELEASE", nncase_mem_semantic_release)
        .value("RELAXED", nncase_mem_semantic_relaxed)
        .export_values();

    py::enum_<nncase_mem_sync_scope_t>(m, "MEM_SYNC_SCOPE", py::module_local())
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

    py::enum_<nncase_rounding_mode_t>(m, "ROUNDING_MODE", py::module_local())
        .value("RTZ", nncase_rounding_mode_rtz)
        .value("RTNE", nncase_rounding_mode_rtne);

    py::enum_<nncase_propagate_nan_t>(m, "PROPAGATE_NAN", py::module_local())
        .value("NONE", nncase_propagate_nan_none)
        .value("ALL", nncase_propagate_nan_all);

    py::enum_<nncase_input_precision_t>(m, "INPUT_PRECISION",
                                        py::module_local())
        .value("TF32", nncase_input_precision_tf32)
        .value("TF32x3", nncase_input_precision_tf32x3)
        .value("IEEE", nncase_input_precision_ieee)
        .export_values();

    py::class_<clr::target>(m, "target").def(py::init<std::string_view>());
    py::class_<clr::compile_options>(m, "compile_options").def(py::init<>());
    py::class_<clr::compile_session>(m, "compile_session")
        .def(py::init<const clr::target &, const clr::compile_options &>())
        .def("disable_multithreading", [](clr::compile_session &) {});

    py::class_<mlir_source_function>(m, "mlir_source_function");
    py::class_<mlir_source_module>(m, "mlir_source_module", py::dynamic_attr())
        .def("get_entry_func_name",
             &mlir_source_module::get_entry_func_name, ret::reference_internal)
        .def("get_function", &mlir_source_module::get_function)
        .def("get_function_signature",
             &mlir_source_module::get_function_signature)
        .def("get_int_attr", &mlir_source_module::get_int_attr)
        .def("to_text", &mlir_source_module::to_text, ret::reference_internal)
        .def("str_nodebug", &mlir_source_module::str_nodebug,
             ret::reference_internal)
        .def("verify", &mlir_source_module::verify)
        .def("create_location_snapshot",
             &mlir_source_module::create_location_snapshot)
        .def("__str__", &mlir_source_module::to_text, ret::reference_internal);

    m.def("context", []() {
        auto target = clr::target("cuda");
        auto options = clr::compile_options();
        return clr::compile_session(target, options);
    });
    m.def("load_dialects", [](clr::compile_session &) {});
    m.def(
        "parse_mlir_module",
        [](const std::string &path, py::object) {
            return mlir_source_module(path, read_text_file(path));
        },
        "path"_a, "context"_a = py::none());

    // Diagnostics
    py::class_<clr::location>(m, "location");
    py::class_<clr::file_location, clr::location>(m, "file_location")
        .def(py::init<std::string_view, int, int, int, int>(), "file_path"_a,
             "line"_a, "column"_a, "end_line"_a, "end_column"_a)
        .def(py::init<std::string_view, int, int>(), "file_path"_a, "line"_a,
             "column"_a);
    py::class_<clr::name_location, clr::location>(m, "name_location");

    // Types
    py::class_<clr::ir_type>(m, "type");
    py::class_<clr::datatype, clr::ir_type>(m, "datatype");
    py::class_<clr::pointer_type, clr::datatype>(m, "pointer_type")
        .def(py::init<clr::datatype, int>(), "elem_type"_a,
             "address_space"_a = 0);
    py::class_<clr::callable_type, clr::ir_type>(m, "callable_type");
    py::class_<clr::tuple_type, clr::ir_type>(m, "tuple_type");
    py::class_<clr::tensor_type, clr::ir_type>(m, "tensor_type");

    // Exprs
    py::class_<clr::expr>(m, "expr")
        .def("to_text", &clr::expr::to_text)
        .def("get_loc", &clr::expr::get_location)
        .def("set_loc", &clr::expr::set_location);

    py::class_<clr::dimension, clr::expr>(m, "dimension");
    py::class_<clr::program_id_dim, clr::dimension>(m, "program_id_dim");

    py::class_<clr::shape, clr::expr>(m, "shape");

    py::class_<clr::var, clr::expr>(m, "var");
    py::class_<clr::tensor_const, clr::expr>(m, "tensor_const");

    py::class_<clr::base_function, clr::expr>(m, "base_function");

    py::class_<clr::ir_module, clr::expr>(m, "module", py::dynamic_attr())
        .def("push_back",
             [](clr::ir_module &self, clr::base_function func) {
                 self.add(std::move(func));
             })
        .def("has_function", &clr::ir_module::has_function)
        .def("get_function", &clr::ir_module::get_function_by_name)
        .def("__str__", &clr::ir_module::to_text)
        .def("str_nodebug", &clr::ir_module::to_text)
        .def("get_entry_func_name", &clr::ir_module::get_entry_func_name)
        .def("describe_vector_add", &clr::ir_module::describe_vector_add)
        .def("verify", [](clr::ir_module &self) {
            return clr::compiler_services::inference_type(self);
        })
        .def("verify_with_diagnostics", [](clr::ir_module &self) {
            return clr::compiler_services::inference_type(self);
        });

    m.def("compile_to_cubin", [](clr::ir_module &module, py::dict options) {
        auto json = py::module_::import("json");
        auto options_json = json.attr("dumps")(options).cast<std::string>();
        clr::native_cuda_compile_result result(module, options_json);
        auto payload = json.attr("loads")(result.metadata_json()).cast<py::dict>();
        if (payload.contains("error") && !payload["error"].is_none()) {
            throw std::runtime_error(payload["error"].cast<std::string>());
        }
        auto cubin = result.cubin();
        payload["cubin"] = py::bytes(cubin.data(), cubin.size());
        return payload;
    });

    py::class_<clr::sequential, clr::expr>(m, "sequential")
        .def("has_terminator", &clr::sequential::has_terminator);

    py::class_<clr::prim_function, clr::base_function>(m, "prim_function")
        .def("add_entry_block", &clr::prim_function::add_body)
        .def("get_num_args",
             [](clr::prim_function &self) {
                 auto body = self.get_body();
                 return body.parameters_count();
             })
        .def("args",
             [](clr::prim_function &self, unsigned idx) {
                 auto body = self.get_body();
                 if (idx >= body.parameters_count())
                     throw pybind11::index_error(
                         "Function argument index out of range");
                 return body.get_parameter(idx);
             })
        .def("set_arg_attr",
             [](clr::prim_function &self, unsigned idx, std::string_view name,
                int value) {
                 auto body = self.get_body();
                 if (idx >= body.parameters_count())
                     throw pybind11::index_error(
                         "Function argument index out of range");
                 auto param = body.get_parameter(idx);
                 param.set_int32_attribute(name, value);
             })
        .def("finalize", [](clr::prim_function &self) {
            auto body = self.get_body();
            if (!body.has_terminator()) {
                throw std::runtime_error(
                    "Function body must have a terminator before finalize.");
            }
        });

    py::class_<insertion_point_snapshot>(m, "insertion_point");

    py::class_<triton_op_builder>(m, "builder", py::dynamic_attr())
        .def(py::init<clr::compile_session>())

        // locations
        .def("create_loc",
             [](triton_op_builder &, std::string_view fileName, int line,
                int column) {
                 return clr::file_location(fileName, line, column);
             })
        .def("create_name_loc",
             [](triton_op_builder &, std::string_view name,
                std::optional<clr::location> childLoc) {
                 return clr::name_location(name, childLoc);
             })
        .def("get_loc", &triton_op_builder::last_location)
        .def("set_loc",
             [](triton_op_builder &self, clr::location loc) {
                 self.set_last_location(std::move(loc));
             })
        .def("set_loc",
             [](triton_op_builder &self, std::string_view fileName, int line,
                int column) {
                 self.set_last_location(
                     clr::file_location(fileName, line, column));
             })

        // insertion point
        .def("get_insertion_block", &triton_op_builder::get_insertion_block)
        .def("get_insertion_point", &triton_op_builder::get_insertion_point)
        .def("restore_insertion_point",
             &triton_op_builder::restore_insertion_point)
        .def("set_insertion_point_to_start",
             &triton_op_builder::set_insertion_point_to_start)
        .def("set_insertion_point_to_end",
             &triton_op_builder::set_insertion_point_to_end)

        // module
        .def("create_module", &triton_op_builder::create_module)

        // types
        .def("get_int8_ty",
             [](triton_op_builder &) { return clr::datatype::int8(); })
        .def("get_int16_ty",
             [](triton_op_builder &) { return clr::datatype::int16(); })
        .def("get_int32_ty",
             [](triton_op_builder &) { return clr::datatype::int32(); })
        .def("get_int64_ty",
             [](triton_op_builder &) { return clr::datatype::int64(); })
        .def("get_half_ty",
             [](triton_op_builder &) { return clr::datatype::float16(); })
        .def("get_bf16_ty",
             [](triton_op_builder &) { return clr::datatype::bfloat16(); })
        .def("get_float_ty",
             [](triton_op_builder &) { return clr::datatype::float32(); })
        .def("get_double_ty",
             [](triton_op_builder &) { return clr::datatype::float64(); })
        .def(
            "get_ptr_ty",
            [](triton_op_builder &, clr::datatype elem_type,
               int address_space) {
                return clr::pointer_type(elem_type, address_space);
            },
            "elem_type"_a, "address_space"_a = 0)
        .def("get_block_ty",
             [](triton_op_builder &, clr::datatype &elementType,
                std::vector<int64_t> &shape) {
                 auto shape_expr = clr::shape::fixed(shape);
                 return clr::tensor_type(elementType, shape_expr);
             })
        .def("get_function_ty",
             [](triton_op_builder &, std::vector<clr::ir_type> inTypes,
                std::vector<clr::ir_type> outTypes) {
                 clr::ir_type retType;
                 if (outTypes.size() == 0) {
                     retType = clr::tuple_type::void_type();
                 } else if (outTypes.size() == 1) {
                     retType = outTypes[0];
                 } else {
                     retType = clr::tuple_type(outTypes);
                 }
                 return clr::callable_type(retType, std::move(inTypes));
             })

        // Constants
        .def("get_int32",
             [](triton_op_builder &, int value) {
                 return clr::tensor_const::scalar(value);
             })
        .def("get_int64",
             [](triton_op_builder &, long value) {
                 return clr::tensor_const::scalar(value);
             })
        .def("get_fp16",
             [](triton_op_builder &, float value) {
                 return clr::tensor_const::scalar_float16(value);
             })
        .def("get_fp32",
             [](triton_op_builder &, float value) {
                 return clr::tensor_const::scalar_float32(value);
             })
        .def("get_fp64",
             [](triton_op_builder &, double value) {
                 return clr::tensor_const::scalar_float64(value);
             })

        // Ops
        .def("get_or_insert_function",
             [](triton_op_builder &, clr::ir_module &module,
                std::string_view funcName, clr::callable_type funcType,
                [[maybe_unused]] std::string_view visibility,
                [[maybe_unused]] bool noinline) {
                 auto func = module.get_function_by_name(funcName);
                 if (func.empty()) {
                     func = clr::prim_function(funcName, funcType);
                 }
                 return func;
             })
        // Function
        .def("ret",
             [](triton_op_builder &self, std::vector<clr::expr> &vals) {
                 return self.insert_expr(
                     clr::ir_builder::tir::return_(std::span(vals)));
             })
        // miscellaneous
        .def("create_make_range",
             [](triton_op_builder &self, [[maybe_unused]] clr::ir_type retTy,
                int start, int end) {
                 return self.insert_expr(clr::ir_builder::tensors::range(
                     clr::tensor_const::scalar(start),
                     clr::tensor_const::scalar(end),
                     clr::tensor_const::scalar(1)));
             })

        // Built-in instruction
        .def("create_get_program_id",
             [](triton_op_builder &self, int axis) {
                 if (axis < 0 || axis > 3)
                     throw pybind11::index_error("program_id must be in [0,3]");
                 return self.insert_expr(
                     clr::ir_builder::cast(clr::ir_builder::shapes::as_tensor(
                                               clr::program_id_dim(axis)),
                                           clr::datatype::int32()));
             })

        // Conversions
        .def("create_int_cast",
             [](triton_op_builder &self, clr::expr value,
                clr::ir_type target_type, [[maybe_unused]] bool isSigned) {
                 return self.insert_expr(clr::ir_builder::cast(
                     value, target_type, nncase_cast_default));
             })
        .def("create_fmul",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(
                     clr::ir_builder::math::binary(nncase_binary_mul, a, b));
             })
        .def("create_fdiv",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(
                     clr::ir_builder::math::binary(nncase_binary_div, a, b));
             })
        .def("create_add",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(
                     clr::ir_builder::math::binary(nncase_binary_add, a, b));
             })
        .def("create_frem",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(
                     clr::ir_builder::math::binary(nncase_binary_mod, a, b));
             })
        .def("create_fadd",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(
                     clr::ir_builder::math::binary(nncase_binary_add, a, b));
             })
        .def("create_fsub",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(
                     clr::ir_builder::math::binary(nncase_binary_sub, a, b));
             })
        // AddPtr (similar to GEP)
        .def("create_addptr",
             [](triton_op_builder &self, clr::expr ptr, clr::expr offset) {
                 return self.insert_expr(clr::ir_builder::math::binary(
                     nncase_binary_add, ptr, offset));
             })

        // Comparison (int)
        .def("create_icmpSLE",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_lower_or_equal, a, b));
             })
        .def("create_icmpSLT",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_lower_than, a, b));
             })
        .def("create_icmpSGE",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_greater_or_equal, a, b));
             })
        .def("create_icmpSGT",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_greater_than, a, b));
             })
        .def("create_icmpULE",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_lower_or_equal, a, b));
             })
        .def("create_icmpULT",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_lower_than, a, b));
             })
        .def("create_icmpUGE",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_greater_or_equal, a, b));
             })
        .def("create_icmpUGT",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_greater_than, a, b));
             })
        .def("create_icmpEQ",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_equal, a, b));
             })
        .def("create_icmpNE",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_not_equal, a, b));
             })
        // Comparison (float)
        .def("create_fcmpOLT",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_lower_than, a, b));
             })
        .def("create_fcmpOGT",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_greater_than, a, b));
             })
        .def("create_fcmpOLE",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_lower_or_equal, a, b));
             })
        .def("create_fcmpOGE",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_greater_or_equal, a, b));
             })
        .def("create_fcmpOEQ",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_equal, a, b));
             })
        .def("create_fcmpONE",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_not_equal, a, b));
             })
        .def("create_fcmpULT",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_lower_than, a, b));
             })
        .def("create_fcmpUGT",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_greater_than, a, b));
             })
        .def("create_fcmpULE",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_lower_or_equal, a, b));
             })
        .def("create_fcmpUGE",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_greater_or_equal, a, b));
             })
        .def("create_fcmpUEQ",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_equal, a, b));
             })
        .def("create_fcmpUNE",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::compare(
                     nncase_compare_not_equal, a, b));
             })
        // Logical
        .def("create_and",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::binary(
                     nncase_binary_logical_and, a, b));
             })
        .def("create_or",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::binary(
                     nncase_binary_logical_or, a, b));
             })
        .def("create_xor",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(clr::ir_builder::math::binary(
                     nncase_binary_logical_xor, a, b));
             })
        // Input/Output
        .def("create_load",
             [](triton_op_builder &self, clr::expr ptr,
                nncase_cache_modifier_t cache_modifier,
                nncase_eviction_policy_t eviction_policy,
                [[maybe_unused]] bool isVolatile,
                [[maybe_unused]] std::optional<std::string> flagtree_hints) {
                 return self.insert_expr(clr::ir_builder::triton::load(
                     ptr, std::nullopt, std::nullopt, cache_modifier,
                     eviction_policy));
             })
        .def("create_masked_load",
             [](triton_op_builder &self, clr::expr ptr, clr::expr mask,
                std::optional<clr::expr> other,
                nncase_cache_modifier_t cache_modifier,
                nncase_eviction_policy_t eviction_policy,
                [[maybe_unused]] bool isVolatile,
                [[maybe_unused]] std::optional<std::string> flagtree_hints) {
                 return self.insert_expr(clr::ir_builder::triton::load(
                     ptr, mask, other, cache_modifier, eviction_policy));
             })
        .def("create_masked_store",
             [](triton_op_builder &self, clr::expr ptr, clr::expr value,
                clr::expr mask, nncase_cache_modifier_t cache_modifier,
                nncase_eviction_policy_t eviction_policy) {
                 return self.insert_expr(clr::ir_builder::triton::store(
                     ptr, value, mask, cache_modifier, eviction_policy));
             })
        .def("create_mul",
             [](triton_op_builder &self, clr::expr a, clr::expr b) {
                 return self.insert_expr(
                     clr::ir_builder::math::binary(nncase_binary_mul, a, b));
             })

        // Implements tl.trans and tl.permute.
        .def("create_splat",
             [](triton_op_builder &self, clr::tensor_type type, clr::expr arg) {
                 return self.insert_expr(
                     clr::ir_builder::tensors::broadcast(arg, type.shape()));
             });

    py::class_<clr::pass_manager>(m, "pass_manager")
        .def(py::init<clr::compile_session &, std::string_view>())
        .def("add_optimize_ttir",
             [](clr::pass_manager &self, int capability) {
                 self.add_optimize_ttir(capability);
             })
        .def("run", &clr::pass_manager::run);
}
