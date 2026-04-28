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

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_view_dim2_add3_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_view_2_add3_rhs_fixed_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_view_2_add3_rhs_fixed_scalar_view_dim2_add3_no_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim2_add3_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_view_2_add3_rhs_fixed_1D_vector_view_dim2_add3_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_lhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_view_dim2_add3_lhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_lhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim2_add3_lhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_view_2_add3_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_view_dim2_add3_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_view_0_add3_rhs_fixed_scalar_raw_tensor_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<16>);

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_lhs_contiguous)(i) = ntt_input_lhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_view_0_add3_rhs_fixed_scalar_view_dim2_add3_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<16>);

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_lhs_contiguous)(i) = ntt_input_lhs(i);
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim2_add3_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_view_0_add3_rhs_fixed_1D_vector_raw_tensor_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<16>);

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_lhs_contiguous)(i) = ntt_input_lhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_view_0_add3_rhs_fixed_1D_vector_view_dim2_add3_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<16>);

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_lhs_contiguous)(i) = ntt_input_lhs(i);
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_view_dim0_add3_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<16>);

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_rhs_contiguous)(i) = ntt_input_rhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_view_2_add3_rhs_fixed_scalar_raw_tensor_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_view_2_add3_rhs_fixed_scalar_view_dim0_add3_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<16>);

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_rhs_contiguous)(i) = ntt_input_rhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim0_add3_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<16>);

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_rhs_contiguous)(i) = ntt_input_rhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_view_2_add3_rhs_fixed_1D_vector_view_dim0_add3_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<16>);

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_rhs_contiguous)(i) = ntt_input_rhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_multi_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, 16, 1>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, 3, 1, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_view_dim2_add3_multi_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, 16, 1>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, 3, (1) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<1, 3, 1, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<1, 3, 1, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<1, 3, 1, 16>);

    for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 1; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_view_2_add3_rhs_fixed_scalar_raw_tensor_multi_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, (16) + 3, 1>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 1, 16, 1>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 1>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, 3, 1, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 1, 16, 1>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 1; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 1; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_view_2_add3_rhs_fixed_scalar_view_dim2_add3_multi_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, (16) + 3, 1>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 1, 16, 1>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 1>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, 3, (1) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<1, 3, 1, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<1, 3, 1, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 1, 16, 1>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 1; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 1; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<1, 3, 1, 16>);

    for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 1; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, 16, 1>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<1, 3, 1, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim2_add3_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, 16, 1>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<1, 3, (1) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<1, 3, 1, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<1, 3, 1, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<1, 3, 1, 16>);

    for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 1; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, (16) + 3, 1>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 1, 16, 1>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 1>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<1, 3, 1, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 1, 16, 1>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 1; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 1; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_view_2_add3_rhs_fixed_1D_vector_view_dim2_add3_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, (16) + 3, 1>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 1, 16, 1>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 1>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<1, 3, (1) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<1, 3, 1, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<1, 3, 1, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 1, 16, 1>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 1; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 1; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<1, 3, 1, 16>);

    for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 1; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_view_dim2_add3_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_view_2_add3_rhs_dynamic_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_view_2_add3_rhs_dynamic_scalar_view_dim2_add3_no_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim2_add3_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_view_2_add3_rhs_dynamic_1D_vector_view_dim2_add3_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_lhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_view_dim2_add3_lhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_lhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim2_add3_lhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<int64_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_view_2_add3_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<int64_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_view_dim2_add3_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_view_0_add3_rhs_dynamic_scalar_raw_tensor_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<16>);

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_lhs_contiguous)(i) = ntt_input_lhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_view_0_add3_rhs_dynamic_scalar_view_dim2_add3_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<16>);

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_lhs_contiguous)(i) = ntt_input_lhs(i);
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim2_add3_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_view_0_add3_rhs_dynamic_1D_vector_raw_tensor_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<16>);

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_lhs_contiguous)(i) = ntt_input_lhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_view_0_add3_rhs_dynamic_1D_vector_view_dim2_add3_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<16>);

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_lhs_contiguous)(i) = ntt_input_lhs(i);
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<int64_t>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_view_dim0_add3_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs = ntt::make_tensor<int64_t>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(16),
        ntt::canonicalize_strides(ntt::make_shape(16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(16));

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_rhs_contiguous)(i) = ntt_input_rhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_view_2_add3_rhs_dynamic_scalar_raw_tensor_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<int64_t>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_view_2_add3_rhs_dynamic_scalar_view_dim0_add3_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs = ntt::make_tensor<int64_t>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(16),
        ntt::canonicalize_strides(ntt::make_shape(16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(16));

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_rhs_contiguous)(i) = ntt_input_rhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim0_add3_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(16),
            ntt::canonicalize_strides(ntt::make_shape(16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(16));

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_rhs_contiguous)(i) = ntt_input_rhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_view_2_add3_rhs_dynamic_1D_vector_view_dim0_add3_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(16),
            ntt::canonicalize_strides(ntt::make_shape(16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(16));

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_rhs_contiguous)(i) = ntt_input_rhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_multi_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, 16, 1>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(1, 3, 1, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_view_dim2_add3_multi_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, 16, 1>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(1, 3, (1) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(1, 3, 1, 16),
        ntt::canonicalize_strides(ntt::make_shape(1, 3, 1, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(1, 3, 1, 16));

    for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 1; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_view_2_add3_rhs_dynamic_scalar_raw_tensor_multi_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, (16) + 3, 1>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 1, 16, 1>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 1>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(1, 3, 1, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 1, 16, 1>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 1; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 1; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_scalar_view_2_add3_rhs_dynamic_scalar_view_dim2_add3_multi_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, (16) + 3, 1>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 1, 16, 1>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 1>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(1, 3, (1) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(1, 3, 1, 16),
        ntt::canonicalize_strides(ntt::make_shape(1, 3, 1, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 1, 16, 1>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 1; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 1; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(1, 3, 1, 16));

    for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 1; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, 16, 1>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(1, 3, 1, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim2_add3_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, 16, 1>);
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(1, 3, (1) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(1, 3, 1, 16),
            ntt::canonicalize_strides(ntt::make_shape(1, 3, 1, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(1, 3, 1, 16));

    for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 1; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, (16) + 3, 1>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 1, 16, 1>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 1>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(1, 3, 1, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 1, 16, 1>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 1; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 1; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_fixed_1D_vector_view_2_add3_rhs_dynamic_1D_vector_view_dim2_add3_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, (16) + 3, 1>);
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 1, 16, 1>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 1>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(1, 3, (1) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(1, 3, 1, 16),
            ntt::canonicalize_strides(ntt::make_shape(1, 3, 1, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 1, 16, 1>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 1; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 1; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(1, 3, 1, 16));

    for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 1; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_view_dim2_add3_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_view_2_add3_rhs_fixed_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_view_2_add3_rhs_fixed_scalar_view_dim2_add3_no_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim2_add3_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_view_2_add3_rhs_fixed_1D_vector_view_dim2_add3_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_lhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<int64_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_view_dim2_add3_lhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<int64_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_lhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim2_add3_lhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_view_2_add3_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<int64_t>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_view_dim2_add3_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<int64_t>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_view_0_add3_rhs_fixed_scalar_raw_tensor_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs = ntt::make_tensor<int64_t>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(16),
        ntt::canonicalize_strides(ntt::make_shape(16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(16));

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_lhs_contiguous)(i) = ntt_input_lhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_view_0_add3_rhs_fixed_scalar_view_dim2_add3_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs = ntt::make_tensor<int64_t>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(16),
        ntt::canonicalize_strides(ntt::make_shape(16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(16));

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_lhs_contiguous)(i) = ntt_input_lhs(i);
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim2_add3_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_view_0_add3_rhs_fixed_1D_vector_raw_tensor_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(16),
            ntt::canonicalize_strides(ntt::make_shape(16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(16));

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_lhs_contiguous)(i) = ntt_input_lhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_view_0_add3_rhs_fixed_1D_vector_view_dim2_add3_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(16),
            ntt::canonicalize_strides(ntt::make_shape(16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(16));

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_lhs_contiguous)(i) = ntt_input_lhs(i);
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<2, 3, 16, 16>);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_view_dim0_add3_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<16>);

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_rhs_contiguous)(i) = ntt_input_rhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_view_2_add3_rhs_fixed_scalar_raw_tensor_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_view_2_add3_rhs_fixed_scalar_view_dim0_add3_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<16>);

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_rhs_contiguous)(i) = ntt_input_rhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim0_add3_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<16>);

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_rhs_contiguous)(i) = ntt_input_rhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_view_2_add3_rhs_fixed_1D_vector_view_dim0_add3_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<16>);

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_rhs_contiguous)(i) = ntt_input_rhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_multi_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, 16, 1));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, 3, 1, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_view_dim2_add3_multi_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, 16, 1));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, 3, (1) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<1, 3, 1, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<1, 3, 1, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<1, 3, 1, 16>);

    for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 1; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_view_2_add3_rhs_fixed_scalar_raw_tensor_multi_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, (16) + 3, 1));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 1, 16, 1),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 1),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, 3, 1, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 1, 16, 1));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 1; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 1; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_view_2_add3_rhs_fixed_scalar_view_dim2_add3_multi_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, (16) + 3, 1));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 1, 16, 1),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 1),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1, 3, (1) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<1, 3, 1, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<1, 3, 1, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 1, 16, 1));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 1; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 1; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::fixed_shape_v<1, 3, 1, 16>);

    for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 1; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(2, 1, 16, 1));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<1, 3, 1, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim2_add3_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(2, 1, 16, 1));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<1, 3, (1) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<1, 3, 1, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<1, 3, 1, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<1, 3, 1, 16>);

    for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 1; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 1, (16) + 3, 1));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 1, 16, 1),
            ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 1),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<1, 3, 1, 16>);
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 1, 16, 1));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 1; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 1; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_view_2_add3_rhs_fixed_1D_vector_view_dim2_add3_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 1, (16) + 3, 1));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 1, 16, 1),
            ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 1),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<1, 3, (1) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<1, 3, 1, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<1, 3, 1, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 1, 16, 1));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 1; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 1; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::fixed_shape_v<1, 3, 1, 16>);

    for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 1; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_view_dim2_add3_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_view_2_add3_rhs_dynamic_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_view_2_add3_rhs_dynamic_scalar_view_dim2_add3_no_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim2_add3_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_view_2_add3_rhs_dynamic_1D_vector_view_dim2_add3_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_lhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<int64_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_view_dim2_add3_lhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<int64_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_lhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim2_add3_lhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<int64_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_view_2_add3_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<int64_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<int64_t>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_view_dim2_add3_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<int64_t>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_view_0_add3_rhs_dynamic_scalar_raw_tensor_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs = ntt::make_tensor<int64_t>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(16),
        ntt::canonicalize_strides(ntt::make_shape(16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(16));

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_lhs_contiguous)(i) = ntt_input_lhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_view_0_add3_rhs_dynamic_scalar_view_dim2_add3_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs = ntt::make_tensor<int64_t>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(16),
        ntt::canonicalize_strides(ntt::make_shape(16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(16));

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_lhs_contiguous)(i) = ntt_input_lhs(i);
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim2_add3_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_view_0_add3_rhs_dynamic_1D_vector_raw_tensor_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(16),
            ntt::canonicalize_strides(ntt::make_shape(16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(16));

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_lhs_contiguous)(i) = ntt_input_lhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_view_0_add3_rhs_dynamic_1D_vector_view_dim2_add3_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(16),
            ntt::canonicalize_strides(ntt::make_shape(16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(16));

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_lhs_contiguous)(i) = ntt_input_lhs(i);
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<int64_t>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_view_dim0_add3_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs = ntt::make_tensor<int64_t>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(16),
        ntt::canonicalize_strides(ntt::make_shape(16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(16));

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_rhs_contiguous)(i) = ntt_input_rhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_view_2_add3_rhs_dynamic_scalar_raw_tensor_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<int64_t>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_view_2_add3_rhs_dynamic_scalar_view_dim0_add3_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs = ntt::make_tensor<int64_t>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(16),
        ntt::canonicalize_strides(ntt::make_shape(16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(16));

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_rhs_contiguous)(i) = ntt_input_rhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim0_add3_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(16),
            ntt::canonicalize_strides(ntt::make_shape(16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(16));

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_rhs_contiguous)(i) = ntt_input_rhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_view_2_add3_rhs_dynamic_1D_vector_view_dim0_add3_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(16),
            ntt::canonicalize_strides(ntt::make_shape(16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 3, 16, 16));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(16));

    for (size_t i = 0; i < 16; i++) {
        (*ntt_input_rhs_contiguous)(i) = ntt_input_rhs(i);
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_multi_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, 16, 1));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(1, 3, 1, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_view_dim2_add3_multi_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, 16, 1));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(1, 3, (1) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(1, 3, 1, 16),
        ntt::canonicalize_strides(ntt::make_shape(1, 3, 1, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(1, 3, 1, 16));

    for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 1; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_view_2_add3_rhs_dynamic_scalar_raw_tensor_multi_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, (16) + 3, 1));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 1, 16, 1),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 1),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(1, 3, 1, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 1, 16, 1));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 1; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 1; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_scalar_view_2_add3_rhs_dynamic_scalar_view_dim2_add3_multi_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, (16) + 3, 1));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 1, 16, 1),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 1),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<int64_t>(ntt::make_shape(1, 3, (1) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(1, 3, 1, 16),
        ntt::canonicalize_strides(ntt::make_shape(1, 3, 1, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(2, 1, 16, 1));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 1; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 1; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<int64_t>(ntt::make_shape(1, 3, 1, 16));

    for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 1; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(2, 1, 16, 1));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(1, 3, 1, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim2_add3_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(2, 1, 16, 1));
    NttTest::init_tensor(ntt_input_lhs, -1000000, 1000000, true, true);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(1, 3, (1) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(1, 3, 1, 16),
            ntt::canonicalize_strides(ntt::make_shape(1, 3, 1, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(1, 3, 1, 16));

    for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 1; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_input_lhs, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 1, (16) + 3, 1));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 1, 16, 1),
            ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 1),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(1, 3, 1, 16));
    NttTest::init_tensor(ntt_input_rhs, -1000000, 1000000, true, true);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 1, 16, 1));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 1; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 1; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, ntt_input_rhs, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestouter_product_Int64,
    Int64_lhs_dynamic_1D_vector_view_2_add3_rhs_dynamic_1D_vector_view_dim2_add3_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 1, (16) + 3, 1));
    NttTest::init_tensor(big_tensor_lhs, -1000000, 1000000, true, true);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 1, 16, 1),
            ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 1),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(1, 3, (1) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, -1000000, 1000000, true, true);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(1, 3, 1, 16),
            ntt::canonicalize_strides(ntt::make_shape(1, 3, 1, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::outer_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_lhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(2, 1, 16, 1));

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 1; j++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t l = 0; l < 1; l++) {
                    (*ntt_input_lhs_contiguous)(i, j, k, l) =
                        ntt_input_lhs(i, j, k, l);
                }
            }
        }
    }

    // Copy to contiguous tensor for ORT reference
    auto ntt_input_rhs_contiguous =
        ntt::make_unique_tensor<ntt::vector<int64_t, P>>(
            ntt::make_shape(1, 3, 1, 16));

    for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 1; k++) {
                for (size_t l = 0; l < 16; l++) {
                    (*ntt_input_rhs_contiguous)(i, j, k, l) =
                        ntt_input_rhs(i, j, k, l);
                }
            }
        }
    }

    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        *ntt_input_lhs_contiguous, *ntt_input_rhs_contiguous, false, true);
    // Execute Ort operation
    auto ort_output = ortki_Mul(ort_input_lhs, ort_input_rhs);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
