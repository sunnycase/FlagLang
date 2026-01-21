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


TEST(BinaryTestmul_Float64, Float64_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_fixed_scalar_raw_tensor_rhs_fixed_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_fixed_scalar_raw_tensor_rhs_fixed_2D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_scalar_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_2D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_fixed_2D_vector_raw_tensor_rhs_fixed_scalar_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_fixed_2D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_fixed_2D_vector_raw_tensor_rhs_fixed_2D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_fixed_scalar_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_fixed_scalar_raw_tensor_rhs_dynamic_2D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_scalar_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_2D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_fixed_2D_vector_raw_tensor_rhs_dynamic_scalar_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_fixed_2D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_fixed_2D_vector_raw_tensor_rhs_dynamic_2D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_dynamic_scalar_raw_tensor_rhs_fixed_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_dynamic_scalar_raw_tensor_rhs_fixed_2D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_scalar_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_2D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_dynamic_2D_vector_raw_tensor_rhs_fixed_scalar_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_dynamic_2D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_dynamic_2D_vector_raw_tensor_rhs_fixed_2D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_2D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_scalar_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_2D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_dynamic_2D_vector_raw_tensor_rhs_dynamic_scalar_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_dynamic_2D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestmul_Float64, Float64_lhs_dynamic_2D_vector_raw_tensor_rhs_dynamic_2D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(double) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1.7e150, 1.7e150, true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1.7e150, 1.7e150, true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::mul>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, false, false);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);
    
    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
