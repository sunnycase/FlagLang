/* Copyright 2019-2024 Canaan Inc.
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
#include "nncase/ntt/shape.h"
#include "nncase/ntt/tensor.h"
#include "nncase/ntt/tensor_traits.h"
#include "nncase/ntt/vector.h"
#include "ntt_test.h"
#include "ortki_helper.h"
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>

using namespace nncase;
using namespace ortki;


TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_fixed_scalar_raw_tensor_shape1) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for input that are tensor of scalar
    auto ntt_input_scalar = (ntt_input).view();
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_fixed_scalar_view_dim2_add3_shape1) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for input that are tensor of scalar
    auto ntt_input_scalar = (ntt_input).view();
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_fixed_1D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,3,16,16,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16, 1>);
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_fixed_1D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,3,16,16,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16, 1>);
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_fixed_2D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4, 5>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,3,16,16,4,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4, 5>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, 4, P>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, 4, P>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16, 4, P>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16, 1, 1>);
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4, 5>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4, 5>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_fixed_2D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4, 5>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,3,16,16,4,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4, 5>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, 4, P>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, 4, P>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16, 4, P>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16, 1, 1>);
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4, 5>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4, 5>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_fixed_scalar_raw_tensor_shape2) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for input that are tensor of scalar
    auto ntt_input_scalar = (ntt_input).view();
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 7>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_fixed_scalar_view_dim2_add3_shape2) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, (16) +3, 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 1, 16, 7>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 7>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for input that are tensor of scalar
    auto ntt_input_scalar = (ntt_input).view();
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 7>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_fixed_1D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,1,16,7,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7, P>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7, P>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 7, P>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 1, 16, 7, 1>);
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_fixed_1D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 1, (16) +3, 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 1, 16, 7>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 7>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,1,16,7,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7, P>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7, P>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 7, P>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 1, 16, 7, 1>);
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_fixed_2D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4, 5>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,1,16,7,4,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4, 5>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7, 4, P>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7, 4, P>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 7, 4, P>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7, 1, 1>);
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4, 5>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4, 5>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_fixed_2D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 1, (16) +3, 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 1, 16, 7>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 7>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4, 5>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,1,16,7,4,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4, 5>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7, 4, P>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7, 4, P>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 7, 4, P>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7, 1, 1>);
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4, 5>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4, 5>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_dynamic_scalar_raw_tensor_shape1) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for input that are tensor of scalar
    auto ntt_input_scalar = (ntt_input).view();
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_dynamic_scalar_view_dim2_add3_shape1) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for input that are tensor of scalar
    auto ntt_input_scalar = (ntt_input).view();
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_dynamic_1D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,3,16,16,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16, P));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 3, 16, 16, 1));
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_dynamic_1D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,3,16,16,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16, P));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 3, 16, 16, 1));
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_dynamic_2D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4, 5>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,3,16,16,4,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4, 5>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, 4, P>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16, 4, P));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16, 4, P));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 3, 16, 16, 1, 1));
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4, 5>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4, 5>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_dynamic_2D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4, 5>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,3,16,16,4,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4, 5>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, 4, P>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16, 4, P));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16, 4, P));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 3, 16, 16, 1, 1));
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4, 5>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4, 5>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_dynamic_scalar_raw_tensor_shape2) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for input that are tensor of scalar
    auto ntt_input_scalar = (ntt_input).view();
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 7));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_dynamic_scalar_view_dim2_add3_shape2) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, (16) +3, 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 1, 16, 7>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 7>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for input that are tensor of scalar
    auto ntt_input_scalar = (ntt_input).view();
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 7));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_dynamic_1D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,1,16,7,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7, P>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 1, 16, 7, P));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 7, P));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 1, 16, 7, 1));
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_dynamic_1D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 1, (16) +3, 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 1, 16, 7>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 7>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,1,16,7,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7, P>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 1, 16, 7, P));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 7, P));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 1, 16, 7, 1));
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_dynamic_2D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4, 5>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,1,16,7,4,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4, 5>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7, 4, P>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 1, 16, 7, 4, P));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 7, 4, P));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 1, 16, 7, 1, 1));
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4, 5>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4, 5>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_fixed_output_dynamic_2D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 1, (16) +3, 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 1, 16, 7>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 7>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4, 5>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,1,16,7,4,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4, 5>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7, 4, P>);
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 1, 16, 7, 4, P));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 7, 4, P));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 1, 16, 7, 1, 1));
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4, 5>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4, 5>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_fixed_scalar_raw_tensor_shape1) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for input that are tensor of scalar
    auto ntt_input_scalar = (ntt_input).view();
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_fixed_scalar_view_dim2_add3_shape1) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for input that are tensor of scalar
    auto ntt_input_scalar = (ntt_input).view();
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_fixed_1D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,3,16,16,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16, 1>);
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_fixed_1D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,3,16,16,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16, 1>);
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_fixed_2D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4, 5>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,3,16,16,4,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4, 5>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, 4, P));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, 4, P>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16, 4, P>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16, 1, 1>);
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4, 5>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4, 5>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_fixed_2D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4, 5>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,3,16,16,4,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4, 5>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, 4, P));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, 4, P>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16, 4, P>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16, 1, 1>);
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4, 5>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4, 5>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_fixed_scalar_raw_tensor_shape2) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 7));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for input that are tensor of scalar
    auto ntt_input_scalar = (ntt_input).view();
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 7));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 7>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_fixed_scalar_view_dim2_add3_shape2) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 1, (16) +3, 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 1, 16, 7),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 7), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for input that are tensor of scalar
    auto ntt_input_scalar = (ntt_input).view();
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 7));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 7>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_fixed_1D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,1,16,7,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 7, P));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7, P>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 7, P>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 1, 16, 7, 1>);
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_fixed_1D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 1, (16) +3, 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 1, 16, 7),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 7), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,1,16,7,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 7, P));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7, P>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 7, P>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<2, 1, 16, 7, 1>);
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_fixed_2D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4, 5>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,1,16,7,4,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4, 5>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 7, 4, P));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7, 4, P>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 7, 4, P>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7, 1, 1>);
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4, 5>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4, 5>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_fixed_2D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 1, (16) +3, 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 1, 16, 7),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 7), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4, 5>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,1,16,7,4,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4, 5>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 7, 4, P));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 7, 4, P>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 7, 4, P>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7, 1, 1>);
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4, 5>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4, 5>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_dynamic_scalar_raw_tensor_shape1) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for input that are tensor of scalar
    auto ntt_input_scalar = (ntt_input).view();
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_dynamic_scalar_view_dim2_add3_shape1) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for input that are tensor of scalar
    auto ntt_input_scalar = (ntt_input).view();
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_dynamic_1D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,3,16,16,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16, P));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 3, 16, 16, 1));
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_dynamic_1D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,3,16,16,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16, P));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 3, 16, 16, 1));
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_dynamic_2D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4, 5>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,3,16,16,4,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4, 5>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, 4, P));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16, 4, P));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16, 4, P));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 3, 16, 16, 1, 1));
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4, 5>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4, 5>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_dynamic_2D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4, 5>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,3,16,16,4,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4, 5>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, 4, P));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16, 4, P));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16, 4, P));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 3, 16, 16, 1, 1));
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4, 5>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4, 5>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_dynamic_scalar_raw_tensor_shape2) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 7));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for input that are tensor of scalar
    auto ntt_input_scalar = (ntt_input).view();
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 7));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 7));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_dynamic_scalar_view_dim2_add3_shape2) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 1, (16) +3, 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 1, 16, 7),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 7), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for input that are tensor of scalar
    auto ntt_input_scalar = (ntt_input).view();
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 7));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 7));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_dynamic_1D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,1,16,7,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 7, P));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 1, 16, 7, P));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 7, P));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 1, 16, 7, 1));
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_dynamic_1D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 1, (16) +3, 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 1, 16, 7),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 7), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,1,16,7,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 7, P));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 1, 16, 7, P));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 7, P));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(2, 1, 16, 7, 1));
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_dynamic_2D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4, 5>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,1,16,7,4,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4, 5>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 7, 4, P));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 1, 16, 7, 4, P));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 7, 4, P));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 1, 16, 7, 1, 1));
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4, 5>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4, 5>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(UnaryTestexp_Float8e5m2, Float8e5m2_input_dynamic_output_dynamic_2D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 1, (16) +3, 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 1, 16, 7),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 7), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::exp>(ntt_input, ntt_output);
    
    
    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_unsqueezed = (ntt_input).unsqueeze(fixed_shape_v<4, 5>);
    auto ntt_input_scalar = ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2,1,16,7,4,P>);
    ntt::unpack(ntt_input_unsqueezed, ntt_input_scalar, fixed_shape_v<4, 5>);
    //1.2 get ntt_input_double
    auto ntt_input_double = ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 7, 4, P));
    
    ntt::cast(ntt_input_scalar, ntt_input_double);
    
    // 2. calculated ort_output
    auto ort_input = NttTest::ntt2ort(ntt_input_double);
    // Execute Ort operation
    auto ort_output = ortki_Exp(ort_input);
    
    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar = ntt::make_unique_tensor<double>(ntt::make_shape(2, 1, 16, 7, 4, P));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);
    
    auto ntt_golden_float_e5m2_t_scalar = ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 7, 4, P));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);
    
    // 4.b if ntt_output is tensor of vector
    auto ntt_golden_unsqueeze = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(ntt::make_shape(2, 1, 16, 7, 1, 1));
    ntt::pack(*ntt_golden_float_e5m2_t_scalar, ntt_golden_unsqueeze, fixed_shape_v<4, 5>);
    auto ntt_golden = ntt_golden_unsqueeze.squeeze( (fixed_shape_v<4, 5>));
    
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
