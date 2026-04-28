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

TEST(CastTest_Float8e5m2, from_Float8e5m2_to_Bool_fixed_scalar_contiguous_3D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_1D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_1D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_1D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_1D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_1D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_2D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_2D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_2D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_2D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_2D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, 4, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_1D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_1D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_1D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_1D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_1D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_2D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, 4, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_2D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, 4, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_2D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, 4, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_2D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, 4, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_2D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, 4, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2, from_Float8e5m2_to_Bool_fixed_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

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
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_1D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_1D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, P>>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_2D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bool, 4, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8 * 4, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2, 3>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2, 3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_fixed_2D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bool, 4, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2, 3>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2, 3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bool>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));

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
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_1D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 16, 8, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_1D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, P>>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 16, 8, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 16, 8, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_2D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, 4, P>>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8 * 4, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2, 3>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 16, 8, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2, 3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bool_dynamic_2D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(1.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<bool, 4, P>>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 16, 8, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2, 3>);
    auto ntt_golden_scalar = ntt::make_tensor<bool>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<bool>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bool, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 16, 8, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2, 3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2, from_Float8e5m2_to_Uint8_fixed_scalar_contiguous_3D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_1D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<uint8_t, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_1D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<uint8_t, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_1D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<uint8_t, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_1D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<uint8_t, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_1D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<uint8_t, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_2D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_2D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_2D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_2D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_2D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_1D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<uint8_t, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_1D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<uint8_t, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_1D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<uint8_t, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_1D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<uint8_t, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_1D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<uint8_t, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_2D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<uint8_t, 4, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_2D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<uint8_t, 4, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_2D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<uint8_t, 4, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_2D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<uint8_t, 4, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_2D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<uint8_t, 4, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2, from_Float8e5m2_to_Uint8_fixed_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<uint8_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

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
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_1D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_1D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_2D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, 4, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8 * 4, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2, 3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2, 3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_fixed_2D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, 4, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2, 3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2, 3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint8_t>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));

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
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_1D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<uint8_t, P>>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 16, 8, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_1D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<uint8_t, P>>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 16, 8, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 16, 8, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_2D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, 4, P>>(
        ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8 * 4, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2, 3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 16, 8, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2, 3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint8_dynamic_2D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint8_t, 4, P>>(
        ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 16, 8, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2, 3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 16, 8, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2, 3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint16_fixed_scalar_contiguous_3D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint16_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint16_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint16_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint16_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint16_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint16_t, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint16_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint16_t, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint16_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint16_t, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint16_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint16_t, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint16_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint16_t, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint16_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint16_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint16_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint16_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint16_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint16_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint16_t, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint16_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint16_t, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint16_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint16_t, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint16_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint16_t, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint16_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint16_t, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint16_fixed_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint16_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<uint16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

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
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint16_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint16_t, P / 2>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 16, 8, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint16_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint16_t, P / 2>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 16, 8, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint16_dynamic_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint16_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint16_t>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));

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
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint16_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint16_t, P / 2>>(
        ntt::make_shape(8, 16, 8, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 16, 8, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint16_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint16_t, P / 2>>(
        ntt::make_shape(8, 16, 8, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 16, 8, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 16, 8, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint32_fixed_scalar_contiguous_3D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint32_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint32_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint32_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint32_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint32_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 4>>(
        ntt::fixed_shape_v<8, 80, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 80, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint32_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 4>>(
        ntt::fixed_shape_v<8, 80, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 80, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint32_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 4>>(
        ntt::fixed_shape_v<8, 80, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 80, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint32_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 4>>(
        ntt::fixed_shape_v<8, 80, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 80, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint32_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 4>>(
        ntt::fixed_shape_v<8, 80, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 80, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint32_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint32_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint32_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint32_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint32_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint32_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 4>>(
        ntt::make_shape(8, 80, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 80, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint32_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 4>>(
        ntt::make_shape(8, 80, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 80, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint32_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 4>>(
        ntt::make_shape(8, 80, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 80, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint32_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 4>>(
        ntt::make_shape(8, 80, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 80, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint32_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 4>>(
        ntt::make_shape(8, 80, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 80, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint32_fixed_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint32_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

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
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint32_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 4>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 16, 8, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint32_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 4>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 16, 8, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint32_dynamic_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint32_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint32_t>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));

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
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint32_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 4>>(
        ntt::make_shape(8, 16, 8, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 16, 8, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint32_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint32_t, P / 4>>(
        ntt::make_shape(8, 16, 8, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 16, 8, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 16, 8, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint64_fixed_scalar_contiguous_3D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint64_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint64_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint64_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint64_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint64_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 8>>(
        ntt::fixed_shape_v<8, 80, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 80, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint64_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 8>>(
        ntt::fixed_shape_v<8, 80, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 80, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint64_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 8>>(
        ntt::fixed_shape_v<8, 80, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 80, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint64_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 8>>(
        ntt::fixed_shape_v<8, 80, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 80, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint64_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 8>>(
        ntt::fixed_shape_v<8, 80, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 80, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint64_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint64_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint64_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint64_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint64_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint64_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 8>>(
        ntt::make_shape(8, 80, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 80, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint64_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 8>>(
        ntt::make_shape(8, 80, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 80, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint64_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 8>>(
        ntt::make_shape(8, 80, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 80, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint64_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 8>>(
        ntt::make_shape(8, 80, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 80, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint64_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 8>>(
        ntt::make_shape(8, 80, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 80, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint64_fixed_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint64_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

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
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint64_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 8>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 16, 8, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint64_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 8>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 16, 8, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint64_dynamic_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint64_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<uint64_t>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));

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
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Uint64_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 8>>(
        ntt::make_shape(8, 16, 8, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 16, 8, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Uint64_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(0.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<uint64_t, P / 8>>(
        ntt::make_shape(8, 16, 8, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 16, 8, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<uint64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<uint64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<uint64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 16, 8, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2, from_Float8e5m2_to_Int8_fixed_scalar_contiguous_3D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_1D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<int8_t, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_1D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<int8_t, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_1D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<int8_t, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_1D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<int8_t, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_1D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<int8_t, P>>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_2D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_2D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_2D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_2D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_2D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_1D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<int8_t, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_1D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<int8_t, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_1D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<int8_t, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_1D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<int8_t, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_1D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<int8_t, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_2D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<int8_t, 4, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_2D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<int8_t, 4, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_2D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<int8_t, 4, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_2D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<int8_t, 4, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_2D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<int8_t, 4, P>>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2, from_Float8e5m2_to_Int8_fixed_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<int8_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

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
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_1D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_1D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_2D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, 4, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8 * 4, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2, 3>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2, 3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_fixed_2D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, 4, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2, 3>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2, 3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int8_t>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));

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
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_1D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<int8_t, P>>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 16, 8, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_1D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<ntt::vector<int8_t, P>>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 16, 8, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 16, 8, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_2D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, 4, P>>(
        ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8 * 4, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2, 3>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 16, 8, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2, 3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int8_dynamic_2D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-11.0f), float_e5m2_t(11.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int8_t, 4, P>>(
        ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 16, 8, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2, 3>);
    auto ntt_golden_scalar = ntt::make_tensor<int8_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int8_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int8_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 16, 8, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2, 3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2, from_Float8e5m2_to_Int16_fixed_scalar_contiguous_3D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int16_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int16_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int16_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int16_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int16_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int16_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int16_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int16_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int16_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int16_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int16_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int16_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int16_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int16_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int16_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int16_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int16_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int16_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int16_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2, from_Float8e5m2_to_Int16_fixed_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int16_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<int16_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

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
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int16_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P / 2>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 16, 8, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int16_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P / 2>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 16, 8, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int16_dynamic_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int16_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int16_t>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));

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
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int16_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P / 2>>(
        ntt::make_shape(8, 16, 8, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 16, 8, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int16_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int16_t, P / 2>>(
        ntt::make_shape(8, 16, 8, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 16, 8, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int16_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int16_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int16_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 16, 8, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2, from_Float8e5m2_to_Int32_fixed_scalar_contiguous_3D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int32_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int32_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int32_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int32_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int32_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 4>>(
        ntt::fixed_shape_v<8, 80, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 80, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int32_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 4>>(
        ntt::fixed_shape_v<8, 80, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 80, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int32_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 4>>(
        ntt::fixed_shape_v<8, 80, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 80, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int32_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 4>>(
        ntt::fixed_shape_v<8, 80, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 80, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int32_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 4>>(
        ntt::fixed_shape_v<8, 80, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 80, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int32_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int32_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int32_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int32_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int32_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int32_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 4>>(
        ntt::make_shape(8, 80, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 80, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int32_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 4>>(
        ntt::make_shape(8, 80, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 80, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int32_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 4>>(
        ntt::make_shape(8, 80, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 80, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int32_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 4>>(
        ntt::make_shape(8, 80, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 80, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int32_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 4>>(
        ntt::make_shape(8, 80, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 80, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2, from_Float8e5m2_to_Int32_fixed_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int32_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<int32_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

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
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int32_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 4>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 16, 8, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int32_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 4>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 16, 8, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int32_dynamic_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int32_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int32_t>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));

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
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int32_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 4>>(
        ntt::make_shape(8, 16, 8, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 16, 8, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int32_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int32_t, P / 4>>(
        ntt::make_shape(8, 16, 8, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 16, 8, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int32_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int32_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int32_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 16, 8, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2, from_Float8e5m2_to_Int64_fixed_scalar_contiguous_3D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int64_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int64_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int64_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int64_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int64_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 8>>(
        ntt::fixed_shape_v<8, 80, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 80, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int64_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 8>>(
        ntt::fixed_shape_v<8, 80, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 80, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int64_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 8>>(
        ntt::fixed_shape_v<8, 80, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 80, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int64_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 8>>(
        ntt::fixed_shape_v<8, 80, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 80, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int64_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 8>>(
        ntt::fixed_shape_v<8, 80, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 80, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int64_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int64_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int64_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int64_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int64_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int64_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 8>>(
        ntt::make_shape(8, 80, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 80, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int64_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 8>>(
        ntt::make_shape(8, 80, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 80, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int64_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 8>>(
        ntt::make_shape(8, 80, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 80, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int64_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 8>>(
        ntt::make_shape(8, 80, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 80, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int64_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 8>>(
        ntt::make_shape(8, 80, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 80, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2, from_Float8e5m2_to_Int64_fixed_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int64_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

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
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int64_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 8>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 16, 8, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int64_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 8>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 16, 8, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int64_dynamic_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int64_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<int64_t>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));

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
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Int64_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 8>>(
        ntt::make_shape(8, 16, 8, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 16, 8, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Int64_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<int64_t, P / 8>>(
        ntt::make_shape(8, 16, 8, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 16, 8, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<int64_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<int64_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<int64_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 16, 8, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float16_fixed_scalar_contiguous_3D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float16_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float16_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float16_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float16_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float16_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<half, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float16_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<half, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float16_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<half, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float16_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<half, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float16_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<half, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float16_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float16_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float16_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float16_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float16_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float16_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<half, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float16_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<half, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float16_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<half, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float16_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<half, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float16_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<half, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float16_fixed_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float16_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

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
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float16_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P / 2>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<half, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 16, 8, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float16_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P / 2>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<half, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 16, 8, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float16_dynamic_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float16_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<half>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));

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
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float16_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P / 2>>(
        ntt::make_shape(8, 16, 8, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<half, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 16, 8, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float16_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<half, P / 2>>(
        ntt::make_shape(8, 16, 8, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 16, 8, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<half>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) = static_cast<half>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<half, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 16, 8, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float32_fixed_scalar_contiguous_3D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float32_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float32_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float32_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float32_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float32_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 4>>(
        ntt::fixed_shape_v<8, 80, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 80, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float32_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 4>>(
        ntt::fixed_shape_v<8, 80, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 80, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float32_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 4>>(
        ntt::fixed_shape_v<8, 80, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 80, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float32_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 4>>(
        ntt::fixed_shape_v<8, 80, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 80, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float32_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 4>>(
        ntt::fixed_shape_v<8, 80, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 80, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float32_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float32_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float32_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float32_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float32_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float32_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 4>>(
        ntt::make_shape(8, 80, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 80, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float32_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 4>>(
        ntt::make_shape(8, 80, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 80, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float32_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 4>>(
        ntt::make_shape(8, 80, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 80, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float32_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 4>>(
        ntt::make_shape(8, 80, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 80, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float32_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 4>>(
        ntt::make_shape(8, 80, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 80, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float32_fixed_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float32_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

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
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float32_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 4>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 16, 8, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float32_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 4>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 4>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::fixed_shape_v<8, 16, 8, 32>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float32_dynamic_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float32_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<float>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));

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
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float32_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 4>>(
        ntt::make_shape(8, 16, 8, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 16, 8, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float32_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float, P / 4>>(
        ntt::make_shape(8, 16, 8, 8 * 4));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 16, 8, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<float>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 4>>(
        ntt::make_shape(8, 16, 8, 32));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float64_fixed_scalar_contiguous_3D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float64_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float64_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float64_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float64_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float64_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 8>>(
        ntt::fixed_shape_v<8, 80, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<double, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 80, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float64_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 8>>(
        ntt::fixed_shape_v<8, 80, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<double, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 80, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float64_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 8>>(
        ntt::fixed_shape_v<8, 80, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<double, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 80, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float64_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 8>>(
        ntt::fixed_shape_v<8, 80, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<double, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 80, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float64_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 8>>(
        ntt::fixed_shape_v<8, 80, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<double, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 80, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float64_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float64_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float64_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float64_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float64_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float64_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 8>>(
        ntt::make_shape(8, 80, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<double, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 80, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float64_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 8>>(
        ntt::make_shape(8, 80, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<double, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 80, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float64_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 8>>(
        ntt::make_shape(8, 80, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<double, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 80, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float64_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 8>>(
        ntt::make_shape(8, 80, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<double, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 80, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float64_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 8>>(
        ntt::make_shape(8, 80, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<double, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 80, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float64_fixed_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<double>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float64_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<double>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

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
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float64_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 8>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<double, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 16, 8, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float64_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 8>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<double, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::fixed_shape_v<8, 16, 8, 64>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float64_dynamic_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float64_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<double>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));

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
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float64_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 8>>(
        ntt::make_shape(8, 16, 8, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<double, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 16, 8, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float64_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-32.0f), float_e5m2_t(32.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<double, P / 8>>(
        ntt::make_shape(8, 16, 8, 8 * 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 16, 8, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar = ntt::make_tensor<double>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<double>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<double, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 8>>(
        ntt::make_shape(8, 16, 8, 64));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bfloat16_fixed_scalar_contiguous_3D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bfloat16_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bfloat16_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bfloat16_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bfloat16_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bfloat16_fixed_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bfloat16, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Bfloat16_fixed_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bfloat16, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Bfloat16_fixed_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bfloat16, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Bfloat16_fixed_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bfloat16, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Bfloat16_fixed_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P / 2>>(
        ntt::fixed_shape_v<8, 80, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bfloat16, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 80, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bfloat16_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bfloat16_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bfloat16_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bfloat16_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bfloat16_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Bfloat16_dynamic_1D_vector_contiguous_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bfloat16, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Bfloat16_dynamic_1D_vector_non_contiguous_dim2_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bfloat16, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Bfloat16_dynamic_1D_vector_non_contiguous_dim2_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bfloat16, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Bfloat16_dynamic_1D_vector_non_contiguous_dim1_mul2_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bfloat16, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Bfloat16_dynamic_1D_vector_non_contiguous_dim1_add5_3D_repack_axis_2) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P / 2>>(
        ntt::make_shape(8, 80, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<2>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bfloat16, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 80, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bfloat16_fixed_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bfloat16_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<bfloat16>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

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
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bfloat16_fixed_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P / 2>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bfloat16, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 16, 8, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Bfloat16_fixed_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P / 2>>(
        ntt::fixed_shape_v<8, 16, 8, 8 * 2>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bfloat16, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::fixed_shape_v<8, 16, 8, 16>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bfloat16_dynamic_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Bfloat16_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<bfloat16>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));

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
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Bfloat16_dynamic_1D_vector_contiguous_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P / 2>>(
        ntt::make_shape(8, 16, 8, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bfloat16, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 16, 8, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Bfloat16_dynamic_1D_vector_non_contiguous_dim1_mul2_4D_repack_axis_3) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-10.0f), float_e5m2_t(10.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<bfloat16, P / 2>>(
        ntt::make_shape(8, 16, 8, 8 * 2));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1, ntt::fixed_shape_v<3>);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 16, 8, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<bfloat16>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<bfloat16>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<bfloat16, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 2>>(
        ntt::make_shape(8, 16, 8, 16));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_fixed_scalar_contiguous_3D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_fixed_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_fixed_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_fixed_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_fixed_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_fixed_1D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_fixed_1D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_fixed_1D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_fixed_1D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_fixed_1D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_fixed_2D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<
        float_e4m3_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_fixed_2D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, (8) + 7>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<
        float_e4m3_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_fixed_2D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, (8) * 2>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<
        float_e4m3_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_fixed_2D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, (80) * 2, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<
        float_e4m3_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_fixed_2D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, (80) + 7, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 80, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 80, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, 4, P>>(
        ntt::fixed_shape_v<8, 80, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 80, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 80 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<
        float_e4m3_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 80, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_dynamic_scalar_contiguous_3D) {
    auto ntt_input = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<float_e4m3_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_dynamic_scalar_non_contiguous_dim2_add5_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<float_e4m3_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_dynamic_scalar_non_contiguous_dim2_mul2_3D) {
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<float_e4m3_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_dynamic_scalar_non_contiguous_dim1_mul2_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<float_e4m3_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_dynamic_scalar_non_contiguous_dim1_add5_3D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<float_e4m3_t>(ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input = *continuous_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_dynamic_1D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_dynamic_1D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_dynamic_1D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_dynamic_1D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_dynamic_1D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_dynamic_2D_vector_contiguous_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, 80, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, 4, P>>(
        ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<
        float_e4m3_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_dynamic_2D_vector_non_contiguous_dim2_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, 80, (8) + 7));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, 4, P>>(
        ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<
        float_e4m3_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_dynamic_2D_vector_non_contiguous_dim2_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, 80, (8) * 2));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, 4, P>>(
        ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<
        float_e4m3_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_dynamic_2D_vector_non_contiguous_dim1_mul2_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, (80) * 2, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, 4, P>>(
        ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<
        float_e4m3_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_dynamic_2D_vector_non_contiguous_dim1_add5_3D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, (80) + 7, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 80, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 80, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, 4, P>>(
        ntt::make_shape(8, 80, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 80, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 80; j++) {
            for (size_t k = 0; k < 8; k++) {
                (*continuous_input)(i, j, k) = ntt_input(i, j, k);
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 80 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<1, 2>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<
        float_e4m3_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 80, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<1, 2>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_fixed_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_fixed_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<float_e4m3_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8>);

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
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_fixed_1D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_fixed_1D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_fixed_2D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    NttTest::init_tensor(ntt_input, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, 4, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8 * 4, 8 * P>);
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2, 3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<
        float_e4m3_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2, 3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_fixed_2D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::fixed_shape_v<8, (16) * 2, 8, 8>);
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<8, 16, 8, 8>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<8, 16, 8, 8>,
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, 4, P>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<8, 16, 8 * 4, 8 * P>);
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2, 3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<
        float_e4m3_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::fixed_shape_v<8, 16, 8, 8>);
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2, 3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_dynamic_scalar_contiguous_4D) {
    auto ntt_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<float_e4m3_t>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input = ntt_input;
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_dynamic_scalar_non_contiguous_dim1_mul2_4D) {
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
        ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                  big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 =
        ntt::make_tensor<float_e4m3_t>(ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8));

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
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto &ntt_golden = ntt_golden_scalar;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_dynamic_1D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 16, 8, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_dynamic_1D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, P>>(
        ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 16, 8, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<
        ntt::vector<float_e4m3_t, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 16, 8, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(CastTest_Float8e5m2,
     from_Float8e5m2_to_Float8e4m3_dynamic_2D_vector_contiguous_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    auto ntt_input = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, 16, 8, 8));
    NttTest::init_tensor(ntt_input, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, 4, P>>(
        ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8 * 4, 8 * P));
    ntt::unpack(ntt_input, ntt_scalar_input, ntt::fixed_shape_v<2, 3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<
        float_e4m3_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 16, 8, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2, 3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
TEST(
    CastTest_Float8e5m2,
    from_Float8e5m2_to_Float8e4m3_dynamic_2D_vector_non_contiguous_dim1_mul2_4D) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    // Create non-contiguous tensor (on dimension 1)
    auto big_tensor = ntt::make_tensor<ntt::vector<float_e5m2_t, 4, P>>(
        ntt::make_shape(8, (16) * 2, 8, 8));
    NttTest::init_tensor(big_tensor, float_e5m2_t(-16.0f), float_e5m2_t(16.0f),
                         true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(8, 16, 8, 8),
            ntt::canonicalize_strides(ntt::make_shape(8, 16, 8, 8),
                                      big_tensor.strides()));
    // Create output tensor
    auto ntt_output1 = ntt::make_tensor<ntt::vector<float_e4m3_t, 4, P>>(
        ntt::make_shape(8, 16, 8, 8));

    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Execute cast operation
    ntt::cast(ntt_input, ntt_output1);

    // Copy to contiguous tensor for ORT reference
    auto continuous_input = ntt::make_unique_tensor<
        ntt::vector<float_e5m2_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8)>>(
        ntt::make_shape(8, 16, 8, 8));

    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 16; j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t l = 0; l < 8; l++) {
                    (*continuous_input)(i, j, k, l) = ntt_input(i, j, k, l);
                }
            }
        }
    }

    auto ntt_scalar_input =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(8, 16, 8 * 4, 8 * P));
    ntt::unpack(*continuous_input, ntt_scalar_input, ntt::fixed_shape_v<2, 3>);
    auto ntt_golden_scalar =
        ntt::make_tensor<float_e4m3_t>(ntt_scalar_input.shape());
    ntt::apply(ntt_golden_scalar.shape(), [&](auto &index) {
        (ntt_golden_scalar)(index) =
            static_cast<float_e4m3_t>(ntt_scalar_input(index));
    });
    auto ntt_golden_vector = ntt::make_tensor<ntt::vector<
        float_e4m3_t, 4, NTT_VLEN / (sizeof(float_e5m2_t) * 8) / 1>>(
        ntt::make_shape(8, 16, 8, 8));
    ntt::pack(ntt_golden_scalar, ntt_golden_vector, ntt::fixed_shape_v<2, 3>);
    auto &ntt_golden = ntt_golden_vector;
    // Compare results
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output1, ntt_golden));
}
int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
