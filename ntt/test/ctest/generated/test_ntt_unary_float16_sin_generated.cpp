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


TEST(UnaryTestsin_Float16, Float16_input_fixed_output_fixed_scalar_raw_tensor_shape1) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_fixed_scalar_view_dim2_add3_shape1) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<half>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_fixed_1D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_fixed_1D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<half, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_fixed_2D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_fixed_2D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<half, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_fixed_scalar_raw_tensor_shape2) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_fixed_scalar_view_dim2_add3_shape2) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 1, (16) +3, 7>);
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<half>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 1, 16, 7>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 7>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_fixed_1D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_fixed_1D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 1, (16) +3, 7>);
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<half, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 1, 16, 7>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 7>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_fixed_2D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_fixed_2D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 1, (16) +3, 7>);
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<half, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 1, 16, 7>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 7>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_dynamic_scalar_raw_tensor_shape1) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<half>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_dynamic_scalar_view_dim2_add3_shape1) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<half>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<half>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_dynamic_1D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_dynamic_1D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<half, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_dynamic_2D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_dynamic_2D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<half, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_dynamic_scalar_raw_tensor_shape2) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<half>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_dynamic_scalar_view_dim2_add3_shape2) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 1, (16) +3, 7>);
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<half>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 1, 16, 7>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 7>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<half>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_dynamic_1D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_dynamic_1D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 1, (16) +3, 7>);
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<half, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 1, 16, 7>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 7>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_dynamic_2D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_fixed_output_dynamic_2D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 1, (16) +3, 7>);
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<half, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<2, 1, 16, 7>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 7>, big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_fixed_scalar_raw_tensor_shape1) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<half>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_fixed_scalar_view_dim2_add3_shape1) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<half>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<half>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_fixed_1D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_fixed_1D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<half, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_fixed_2D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_fixed_2D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<half, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_fixed_scalar_raw_tensor_shape2) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<half>(ntt::make_shape(2, 1, 16, 7));
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_fixed_scalar_view_dim2_add3_shape2) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<half>(ntt::make_shape(2, 1, (16) +3, 7));
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<half>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 1, 16, 7),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 7), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_fixed_1D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_fixed_1D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 1, (16) +3, 7));
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<half, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 1, 16, 7),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 7), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_fixed_2D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_fixed_2D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 1, (16) +3, 7));
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<half, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 1, 16, 7),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 7), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_dynamic_scalar_raw_tensor_shape1) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<half>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<half>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_dynamic_scalar_view_dim2_add3_shape1) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<half>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<half>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<half>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_dynamic_1D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_dynamic_1D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<half, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_dynamic_2D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_dynamic_2D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<half, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_dynamic_scalar_raw_tensor_shape2) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<half>(ntt::make_shape(2, 1, 16, 7));
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<half>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_dynamic_scalar_view_dim2_add3_shape2) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<half>(ntt::make_shape(2, 1, (16) +3, 7));
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<half>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 1, 16, 7),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 7), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<half>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_dynamic_1D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_dynamic_1D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 1, (16) +3, 7));
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<half, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 1, 16, 7),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 7), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_dynamic_2D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::init_tensor(ntt_input, half(-100.0f), half(100.0f), true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
TEST(UnaryTestsin_Float16, Float16_input_dynamic_output_dynamic_2D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(half) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 1, (16) +3, 7));
    NttTest::init_tensor(big_tensor, half(-100.0f), half(100.0f), true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<half, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(2, 1, 16, 7),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 7), big_tensor.strides())
        );
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 1, 16, 7));
    
    // Execute unary operation
    ntt::unary<ntt::ops::sin>(ntt_input, ntt_output);
    
    
    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ortki_Cast(ort_input_org, 1, ortki::DataType_DOUBLE);
    // Execute Ort operation
    auto ort_output = ortki_Sin(ort_input);
    
    // Cast outputs from double to original datatype
    auto ort_golden = ortki_Cast(ort_output, 1, ortki::DataType_FLOAT16);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 2));
    }
    
int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
