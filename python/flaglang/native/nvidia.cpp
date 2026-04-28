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
#include "cublas_instance.h"
#include "ffi_modules.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

void check_matmul_constraints(const std::string &a_dtype,
                              const std::string &b_dtype,
                              const std::string &c_dtype,
                              const std::vector<int> &a_shape,
                              const std::vector<int> &b_shape,
                              const std::vector<int> &c_shape) {
    if (a_dtype != b_dtype || a_dtype != c_dtype) {
        throw std::runtime_error("Data types do not match.");
    }
    if (a_dtype != "torch.float8_e4m3fn" && a_dtype != "torch.float16" &&
        a_dtype != "torch.float32" && a_dtype != "torch.bfloat16") {
        throw std::runtime_error("Unsupported data type.");
    }

    if (a_shape.size() != 2 || b_shape.size() != 2 || c_shape.size() != 2) {
        throw std::runtime_error("Only 2D matrices are supported.");
    }

    const auto k = a_shape[1];
    if (k != b_shape[1]) {
        throw std::runtime_error(
            "Matrix dimensions do not match. A is [" +
            std::to_string(a_shape[0]) + ", " + std::to_string(a_shape[1]) +
            "], B is [" + std::to_string(b_shape[0]) + ", " +
            std::to_string(b_shape[1]) +
            "]. Expected A.shape[1] == B.shape[1]. Note that B needs to be "
            "transposed.");
    }

    const auto m = a_shape[0];
    if (m != c_shape[0]) {
        throw std::runtime_error(
            "Matrix dimensions do not match. A is [" +
            std::to_string(a_shape[0]) + ", " + std::to_string(a_shape[1]) +
            "], C is [" + std::to_string(c_shape[0]) + ", " +
            std::to_string(c_shape[1]) + "]. Expected A.shape[0] == C.shape[0].");
    }

    const auto n = b_shape[0];
    if (n != c_shape[1]) {
        throw std::runtime_error(
            "Matrix dimensions do not match. B is [" +
            std::to_string(b_shape[0]) + ", " + std::to_string(b_shape[1]) +
            "], C is [" + std::to_string(c_shape[0]) + ", " +
            std::to_string(c_shape[1]) +
            "]. Expected B.shape[0] == C.shape[1]. Note that B needs to be "
            "transposed.");
    }
}

cudaDataType_t to_cuda_dtype(const std::string &torch_dtype) {
    const auto dot = torch_dtype.find_last_of('.');
    const auto dtype = dot == std::string::npos ? torch_dtype
                                                : torch_dtype.substr(dot + 1);
    if (dtype == "float8_e4m3fn") {
        return CUDA_R_8F_E4M3;
    }
    if (dtype == "float16") {
        return CUDA_R_16F;
    }
    if (dtype == "float32") {
        return CUDA_R_32F;
    }
    if (dtype == "bfloat16") {
        return CUDA_R_16BF;
    }

    throw std::runtime_error("Unsupported dtype for cublasLt.matmul: " + dtype);
}

std::vector<int> shape_of(py::object &tensor) {
    return tensor.attr("shape").cast<std::vector<int>>();
}

std::string dtype_of(py::object &tensor) {
    return tensor.attr("dtype").attr("__str__")().cast<std::string>();
}

uint64_t data_ptr_of(py::object &tensor) {
    return tensor.attr("data_ptr")().cast<uint64_t>();
}

} // namespace

void init_triton_nvidia(py::module &&m) {
    auto cublas = m.def_submodule("cublas");

    py::class_<CublasLtInstance>(cublas, "CublasLt")
        .def(py::init<>([](py::object &workspace) {
            const auto workspace_ptr = data_ptr_of(workspace);
            const auto workspace_size =
                workspace.attr("numel")().cast<size_t>() *
                workspace.attr("element_size")().cast<size_t>();
            return new CublasLtInstance(workspace_ptr, workspace_size);
        }))
        .def("matmul",
             [](CublasLtInstance &self, py::object &a, py::object &b,
                py::object &c) {
                 const auto a_shape = shape_of(a);
                 const auto b_shape = shape_of(b);
                 const auto c_shape = shape_of(c);
                 const auto a_dtype = dtype_of(a);
                 const auto b_dtype = dtype_of(b);
                 const auto c_dtype = dtype_of(c);

                 check_matmul_constraints(a_dtype, b_dtype, c_dtype, a_shape,
                                          b_shape, c_shape);
                 self.matmul(a_shape[0], b_shape[0], a_shape[1],
                             data_ptr_of(a), data_ptr_of(b), data_ptr_of(c),
                             to_cuda_dtype(a_dtype));
             })
        .def("gemm",
             [](CublasLtInstance &self, py::object &a, py::object &b,
                py::object &c, py::object &d, float alpha, float beta) {
                 const auto a_shape = shape_of(a);
                 const auto b_shape = shape_of(b);
                 const auto c_shape = shape_of(c);
                 const auto d_shape = shape_of(d);
                 const auto a_dtype = dtype_of(a);
                 const auto b_dtype = dtype_of(b);
                 const auto c_dtype = dtype_of(c);
                 const auto d_dtype = dtype_of(d);

                 check_matmul_constraints(a_dtype, b_dtype, d_dtype, a_shape,
                                          b_shape, d_shape);
                 if (c_dtype != "torch.float16") {
                     throw std::runtime_error("C dtype must be float16, got " +
                                              c_dtype);
                 }
                 if (c_shape != d_shape) {
                     throw std::runtime_error("C and D shapes must match");
                 }

                 self.gemm(a_shape[0], b_shape[0], a_shape[1], data_ptr_of(a),
                           data_ptr_of(b), data_ptr_of(c), data_ptr_of(d),
                           to_cuda_dtype(a_dtype), alpha, beta);
             });
}
