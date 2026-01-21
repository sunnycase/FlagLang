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


static ortki::OrtKITensor* ortki_SwishB(ortki::OrtKITensor* ort_input, ortki::OrtKITensor* beta_tensor) {
    auto ntt_1_tensor = make_tensor<double>(ntt::fixed_shape_v<1>);
    ntt_1_tensor(0) = 1.0f;
    auto ort_1 = NttTest::ntt2ort(ntt_1_tensor);
    auto ort_neg = ortki_Neg(ort_input);
    auto ort_mul = ortki_Mul(ort_neg, beta_tensor);
    auto ort_exp = ortki_Exp(ort_mul);
    auto ort_add = ortki_Add(ort_1, ort_exp);
    return ortki_Div(ort_input, ort_add);
}


TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_scalar_view_2_add3_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<uint32_t>(
        big_tensor_lhs.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_scalar_raw_tensor_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_scalar_view_2_add3_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<uint32_t>(
        big_tensor_lhs.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_scalar_raw_tensor_rhs_fixed_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_scalar_view_2_add3_rhs_fixed_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<uint32_t>(
        big_tensor_lhs.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_1D_vector_view_2_add3_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, P>>(
        big_tensor_lhs.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_1D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, P>>(
        big_tensor_lhs.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_1D_vector_view_2_add3_rhs_fixed_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, P>>(
        big_tensor_lhs.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_2D_vector_raw_tensor_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_2D_vector_view_2_add3_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, 4, P>>(
        big_tensor_lhs.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_2D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_2D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, 4, P>>(
        big_tensor_lhs.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_2D_vector_raw_tensor_rhs_fixed_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_2D_vector_view_2_add3_rhs_fixed_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, 4, P>>(
        big_tensor_lhs.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_scalar_view_2_add3_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<uint32_t>(
        big_tensor_lhs.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_scalar_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_scalar_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<uint32_t>(
        big_tensor_lhs.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_scalar_raw_tensor_rhs_dynamic_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_scalar_view_2_add3_rhs_dynamic_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<uint32_t>(
        big_tensor_lhs.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<uint32_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_1D_vector_view_2_add3_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, P>>(
        big_tensor_lhs.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_1D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, P>>(
        big_tensor_lhs.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_1D_vector_view_2_add3_rhs_dynamic_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, P>>(
        big_tensor_lhs.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_2D_vector_raw_tensor_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_2D_vector_view_2_add3_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, 4, P>>(
        big_tensor_lhs.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_2D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_2D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, 4, P>>(
        big_tensor_lhs.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_2D_vector_raw_tensor_rhs_dynamic_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_fixed_2D_vector_view_2_add3_rhs_dynamic_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, (16) +3, 16>);
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, 4, P>>(
        big_tensor_lhs.elements().data(),
        ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>, big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<2, 3, 16, 16>);
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_scalar_view_2_add3_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<uint32_t>(
        big_tensor_lhs.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_scalar_raw_tensor_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_scalar_view_2_add3_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<uint32_t>(
        big_tensor_lhs.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_scalar_raw_tensor_rhs_fixed_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_scalar_view_2_add3_rhs_fixed_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<uint32_t>(
        big_tensor_lhs.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_1D_vector_view_2_add3_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, P>>(
        big_tensor_lhs.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_1D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, P>>(
        big_tensor_lhs.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_1D_vector_view_2_add3_rhs_fixed_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, P>>(
        big_tensor_lhs.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_2D_vector_raw_tensor_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_2D_vector_view_2_add3_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, 4, P>>(
        big_tensor_lhs.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_2D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_2D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, 4, P>>(
        big_tensor_lhs.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_2D_vector_raw_tensor_rhs_fixed_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_2D_vector_view_2_add3_rhs_fixed_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, 4, P>>(
        big_tensor_lhs.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_scalar_view_2_add3_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<uint32_t>(
        big_tensor_lhs.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_scalar_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<uint32_t>(
        big_tensor_lhs.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_scalar_view_2_add3_rhs_dynamic_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<uint32_t>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<uint32_t>(
        big_tensor_lhs.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<uint32_t>(ntt::make_shape(2, 3, 16, 16));
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_1D_vector_view_2_add3_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, P>>(
        big_tensor_lhs.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_1D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, P>>(
        big_tensor_lhs.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_1D_vector_view_2_add3_rhs_dynamic_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, P>>(
        big_tensor_lhs.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(2, 3, 16, 16));
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_2D_vector_raw_tensor_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_2D_vector_view_2_add3_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, 4, P>>(
        big_tensor_lhs.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<uint32_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_2D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_2D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, 4, P>>(
        big_tensor_lhs.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_2D_vector_raw_tensor_rhs_dynamic_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, 0, 15536, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(ntt_input_lhs,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
TEST(BinaryTestswishb_Uint32, Uint32_lhs_dynamic_2D_vector_view_2_add3_rhs_dynamic_2D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, (16) +3, 16));
    NttTest::init_tensor(big_tensor_lhs, 0, 15536, true, true);
    
    auto ntt_input_lhs = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, 4, P>>(
        big_tensor_lhs.elements().data(),
        ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16), big_tensor_lhs.strides())
        );
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, 0, 15536, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    
    // Execute binary operation
    ntt::binary<ntt::ops::swishb>(ntt_input_lhs, ntt_input_rhs, ntt_output);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_lhs_contiguous = ntt::make_unique_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 16; k++) {
                    for (size_t l = 0; l < 16; l++) {
                        (*ntt_input_lhs_contiguous)(i, j, k, l) = ntt_input_lhs(i, j, k, l);
                    }
                }
            }
        }
    
    // ort_input_lhs, ort_input_rhs would be tensor of double in ort format
    
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(*ntt_input_lhs_contiguous,ntt_input_rhs, true, false);
    // Execute Ort operation
    auto ort_output = ortki_SwishB(ort_input_lhs, ort_input_rhs);
    
    // Cast outputs from double to original datatype
    auto ort_goldenint = ortki_Cast(ort_output, 1, ortki::DataType_INT64);
    auto ort_golden = ortki_Cast(ort_goldenint, 1, ortki::DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) 
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, 4, P>>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
    }
    
int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
