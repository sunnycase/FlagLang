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


TEST(CastTest_Uint16, from_Uint16_to_Bool_fixed_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 1, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 1, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 1, 1, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_BOOL);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 1, 1, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_BOOL);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 1, 1, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_BOOL);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 1, 1, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_BOOL);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 1, 1, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_BOOL);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 1, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bool>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bool>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bool>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bool>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bool>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 1, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 1, 1, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_BOOL);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 1, 1, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_BOOL);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 1, 1, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_BOOL);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 1, 1, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_BOOL);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 1, 1, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_BOOL);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_fixed_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 1, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 1, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8 / 1, 1, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_BOOL);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8 / 1, 1, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_BOOL);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_dynamic_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 1, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bool>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BOOL);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bool>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 1, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8 / 1, 1, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_BOOL);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bool_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 1, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8 / 1, 1, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_BOOL);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_fixed_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 16, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 16, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 16, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 16, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_fixed_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 16, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 16, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::fixed_shape_v<8, 16, 8, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::fixed_shape_v<8, 16, 8, 8 / 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::fixed_shape_v<8, 16, 8, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::fixed_shape_v<8, 16, 8, 8 / 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_dynamic_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 16, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 16, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::make_shape(8, 16, 8, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::make_shape(8, 16, 8, 8 / 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint8_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::make_shape(8, 16, 8, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint8_t, P * 2>>(ntt::make_shape(8, 16, 8, 8 / 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_fixed_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_fixed_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::fixed_shape_v<8, 16, 8, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::fixed_shape_v<8, 16, 8, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::fixed_shape_v<8, 16, 8, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::fixed_shape_v<8, 16, 8, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_dynamic_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::make_shape(8, 16, 8, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::make_shape(8, 16, 8, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint32_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::make_shape(8, 16, 8, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint32_t, P / 2>>(ntt::make_shape(8, 16, 8, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_fixed_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_fixed_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::fixed_shape_v<8, 16, 8, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::fixed_shape_v<8, 16, 8, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::fixed_shape_v<8, 16, 8, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::fixed_shape_v<8, 16, 8, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_dynamic_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_UINT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::make_shape(8, 16, 8, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::make_shape(8, 16, 8, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Uint64_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::make_shape(8, 16, 8, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_UINT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<uint64_t, P / 4>>(ntt::make_shape(8, 16, 8, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_fixed_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 11, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 11, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 11, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int8_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int8_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int8_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int8_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int8_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 11, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_fixed_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 11, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 11, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::fixed_shape_v<8, 16, 8, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::fixed_shape_v<8, 16, 8, 8 / 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::fixed_shape_v<8, 16, 8, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::fixed_shape_v<8, 16, 8, 8 / 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_dynamic_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 11, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int8_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT8);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int8_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 11, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::make_shape(8, 16, 8, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::make_shape(8, 16, 8, 8 / 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int8_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 11, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::make_shape(8, 16, 8, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8 / 2, 2, P};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT8);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int8_t, P * 2>>(ntt::make_shape(8, 16, 8, 8 / 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 181, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_1D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 181, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_1D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_1D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_1D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_1D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_2D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 181, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_2D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_2D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_2D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_2D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 181, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int16_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int16_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int16_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int16_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int16_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_1D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 181, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_1D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_1D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_1D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_1D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_2D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 181, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_2D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_2D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_2D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_2D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 181, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_1D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 181, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_1D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_2D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 181, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_fixed_2D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 181, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int16_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int16_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_1D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 181, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_1D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_2D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 181, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int16_dynamic_2D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 181, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int16_t, 4, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_fixed_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int32_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int32_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int32_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int32_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int32_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_fixed_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::fixed_shape_v<8, 16, 8, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::fixed_shape_v<8, 16, 8, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::fixed_shape_v<8, 16, 8, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::fixed_shape_v<8, 16, 8, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_dynamic_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int32_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT32);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int32_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::make_shape(8, 16, 8, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::make_shape(8, 16, 8, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int32_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::make_shape(8, 16, 8, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT32);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int32_t, P / 2>>(ntt::make_shape(8, 16, 8, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_fixed_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_fixed_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::fixed_shape_v<8, 16, 8, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::fixed_shape_v<8, 16, 8, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::fixed_shape_v<8, 16, 8, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::fixed_shape_v<8, 16, 8, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_dynamic_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_INT64);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::make_shape(8, 16, 8, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::make_shape(8, 16, 8, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Int64_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::make_shape(8, 16, 8, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_INT64);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P / 4>>(ntt::make_shape(8, 16, 8, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 100, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_1D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 100, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_1D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_1D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_1D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_1D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_2D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 100, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_2D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_2D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_2D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_2D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 100, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_1D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 100, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_1D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_1D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_1D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_1D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_2D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 100, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_2D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_2D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_2D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_2D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 100, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_1D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 100, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_1D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_2D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 100, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_fixed_2D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 100, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<half>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_1D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 100, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_1D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_2D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 100, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float16_dynamic_2D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 100, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<half, 4, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_fixed_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_FLOAT);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_FLOAT);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_FLOAT);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_FLOAT);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_FLOAT);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::fixed_shape_v<8, 80, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<float>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<float>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<float>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<float>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<float>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_FLOAT);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_FLOAT);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_FLOAT);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_FLOAT);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_FLOAT);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::make_shape(8, 80, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_fixed_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::fixed_shape_v<8, 16, 8, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_FLOAT);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::fixed_shape_v<8, 16, 8, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::fixed_shape_v<8, 16, 8, 8 * 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_FLOAT);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::fixed_shape_v<8, 16, 8, 8 * 2>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_dynamic_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<float>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_FLOAT);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<float>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::make_shape(8, 16, 8, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_FLOAT);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::make_shape(8, 16, 8, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float32_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::make_shape(8, 16, 8, 8 * 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 2, P / 2};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_FLOAT);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<float, P / 2>>(ntt::make_shape(8, 16, 8, 8 * 2));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_fixed_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_DOUBLE);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_DOUBLE);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_DOUBLE);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_DOUBLE);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_DOUBLE);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_DOUBLE);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_DOUBLE);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_DOUBLE);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_DOUBLE);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_DOUBLE);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::fixed_shape_v<8, 80, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_DOUBLE);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<double>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_DOUBLE);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<double>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_DOUBLE);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<double>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_DOUBLE);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<double>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_DOUBLE);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<double>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_DOUBLE);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_DOUBLE);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_DOUBLE);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_DOUBLE);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 80, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_DOUBLE);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::make_shape(8, 80, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_fixed_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_DOUBLE);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_DOUBLE);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::fixed_shape_v<8, 16, 8, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_DOUBLE);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::fixed_shape_v<8, 16, 8, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::fixed_shape_v<8, 16, 8, 8 * 4>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_DOUBLE);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::fixed_shape_v<8, 16, 8, 8 * 4>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_dynamic_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_DOUBLE);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<double>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_DOUBLE);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<double>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 256, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::make_shape(8, 16, 8, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(ntt_input);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_DOUBLE);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::make_shape(8, 16, 8, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float64_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 256, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::make_shape(8, 16, 8, 8 * 4));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    // Reshape and transpose for 1D vector cast
    int64_t reshape_data[] = {8, 16, 8, 8, 4, P / 4};
    int64_t reshape_shape[] = {std::size(reshape_data)};
    auto ort_type = NttTest::primitive_type2ort_type<int64_t>();
    auto shape_tensor1 = make_tensor(reinterpret_cast<void *>(reshape_data), ort_type,
                             reshape_shape, std::size(reshape_shape));
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    auto reshaped_tensor1 = ortki_Reshape(ort_input, shape_tensor1, 0);
    
    int64_t perms_data[] = {0, 1, 2, 3, 4, 5};
    auto ort_cast_input = ortki_Transpose(reshaped_tensor1, perms_data, std::size(perms_data));
    
    auto ort_output = ortki_Cast(ort_cast_input, 1, DataType_DOUBLE);
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<double, P / 4>>(ntt::make_shape(8, 16, 8, 8 * 4));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 10, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_1D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 10, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_1D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_1D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_1D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_1D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_2D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 10, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_2D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_2D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_2D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_2D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 10, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_1D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 10, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_1D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_1D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_1D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_1D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_2D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 10, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_2D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_2D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_2D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_2D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*ntt_input_contiguous)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::make_shape(8, 80, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 10, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_1D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 10, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_1D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_2D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 10, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_fixed_2D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 10, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_1D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 10, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_1D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_2D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 10, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ort_input = NttTest::ntt2ort(ntt_input);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Bfloat16_dynamic_2D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, 4, P>>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 10, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, 4, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto ntt_input_contiguous = ntt::make_unique_tensor<ntt::vector<uint16_t, 4, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*ntt_input_contiguous)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ort_input = NttTest::ntt2ort(*ntt_input_contiguous);
    // ORT reference implementation
    auto ort_output = ortki_Cast(ort_input, 1, DataType_BFLOAT16);
    
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden) and compare with tested NTT output
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<bfloat16, 4, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::ort2ntt(ort_output, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
    }
    
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_fixed_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 16, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 16, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::fixed_shape_v<8, 80, 4>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::fixed_shape_v<8, 80, 4>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::fixed_shape_v<8, 80, 4>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::fixed_shape_v<8, 80, 4>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::fixed_shape_v<8, 80, 4>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 16, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e4m3_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e4m3_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e4m3_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e4m3_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e4m3_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 16, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::make_shape(8, 80, 4));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::make_shape(8, 80, 4));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::make_shape(8, 80, 4));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::make_shape(8, 80, 4));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::make_shape(8, 80, 4));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_fixed_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 16, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 16, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P * 2>>(ntt::fixed_shape_v<8, 16, 8, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::fixed_shape_v<8, 16, 8, 4>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P * 2>>(ntt::fixed_shape_v<8, 16, 8, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::fixed_shape_v<8, 16, 8, 4>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_dynamic_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 16, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e4m3_t>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e4m3_t>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 16, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P * 2>>(ntt::make_shape(8, 16, 8, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::make_shape(8, 16, 8, 4));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e4m3_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 16, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P * 2>>(ntt::make_shape(8, 16, 8, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::make_shape(8, 16, 8, 4));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_fixed_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 32, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, 0, 32, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e5m2_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::fixed_shape_v<8, 80, 4>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) +7>);
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e5m2_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::fixed_shape_v<8, 80, 4>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 80, (8) *2>);
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e5m2_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::fixed_shape_v<8, 80, 4>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) *2, 8>);
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e5m2_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::fixed_shape_v<8, 80, 4>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (80) +7, 8>);
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e5m2_t, P * 2>>(ntt::fixed_shape_v<8, 80, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 80, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::fixed_shape_v<8, 80, 4>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 32, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, 0, 32, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e5m2_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::make_shape(8, 80, 4));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) +7));
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e5m2_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::make_shape(8, 80, 4));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 80, (8) *2));
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e5m2_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::make_shape(8, 80, 4));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) *2, 8));
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e5m2_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::make_shape(8, 80, 4));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (80) +7, 8));
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e5m2_t, P * 2>>(ntt::make_shape(8, 80, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 80, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 80; j++) {
                for (size_t k = 0; k < 8; k++) {
                    (*continuous_input)(i, j, k) = ntt_input(i, j, k);
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::make_shape(8, 80, 4));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_fixed_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 32, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, 0, 32, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e5m2_t, P * 2>>(ntt::fixed_shape_v<8, 16, 8, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::fixed_shape_v<8, 16, 8, 4>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::fixed_shape_v<8, (16) *2, 8, 8>);
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>, big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e5m2_t, P * 2>>(ntt::fixed_shape_v<8, 16, 8, 8 / 2>);
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::fixed_shape_v<8, 16, 8, 8>);
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::fixed_shape_v<8, 16, 8, 4>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_dynamic_scalar_contiguous_4D) {
    auto ntt_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 32, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<uint16_t>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<uint16_t>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto& ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, 0, 32, true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e5m2_t, P * 2>>(ntt::make_shape(8, 16, 8, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::make_shape(8, 16, 8, 4));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Uint16, from_Uint16_to_Float8e5m2_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint16_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<uint16_t, P>>(ntt::make_shape(8, (16) *2, 8, 8));
    NttTest::init_tensor(big_tensor, 0, 32, true, false);
    
    auto ntt_input = ntt::make_tensor_view_from_address<ntt::vector<uint16_t, P>>(
        big_tensor.elements().data(),
        ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8), big_tensor.strides())
        );
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e5m2_t, P * 2>>(ntt::make_shape(8, 16, 8, 8 / 2));
    
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);
    
    
        // Copy to contiguous tensor for ORT reference
        auto continuous_input = ntt::make_unique_tensor<ntt::vector<uint16_t, NTT_VLEN / (sizeof(uint16_t) * 8)>>(ntt::make_shape(8, 16, 8, 8));
        
        for (size_t i = 0; i < 8; i++) {
            for (size_t j = 0; j < 16; j++) {
                for (size_t k = 0; k < 8; k++) {
                    for (size_t l = 0; l < 8; l++) {
                        (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                    }
                }
            }
        }
    
    auto ntt_scalar_input = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<float_e5m2_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto& index){
      (ntt_golden_scalar)(index) = static_cast<float_e5m2_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(uint16_t) * 8) * 2>>(ntt::make_shape(8, 16, 8, 4));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto& ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
