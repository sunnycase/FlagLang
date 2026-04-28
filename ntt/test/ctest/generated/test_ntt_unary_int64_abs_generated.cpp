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

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_fixed_scalar_raw_tensor_shape1) {
    //---init ntt_input---
    auto ntt_input =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_fixed_scalar_view_dim2_add3_shape1) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_fixed_1D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_fixed_1D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_fixed_2D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_fixed_2D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_fixed_scalar_raw_tensor_shape2) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, 16, 7>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_fixed_scalar_view_dim2_add3_shape2) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, (16) + 3, 7>);
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<2, 1, 16, 7>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 7>,
                                  big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, 16, 7>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_fixed_1D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_fixed_1D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, (16) + 3, 7>);
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<2, 1, 16, 7>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 7>,
                                      big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_fixed_2D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_fixed_2D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 1, (16) + 3, 7>);
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<2, 1, 16, 7>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 7>,
                                      big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_dynamic_scalar_raw_tensor_shape1) {
    //---init ntt_input---
    auto ntt_input =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_dynamic_scalar_view_dim2_add3_shape1) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_dynamic_1D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_dynamic_1D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_dynamic_2D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_dynamic_2D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_dynamic_scalar_raw_tensor_shape2) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, 16, 7));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_dynamic_scalar_view_dim2_add3_shape2) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, (16) + 3, 7>);
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor.elements().data(), ntt::fixed_shape_v<2, 1, 16, 7>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 7>,
                                  big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, 16, 7));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_dynamic_1D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(2, 1, 16, 7));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_dynamic_1D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, (16) + 3, 7>);
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<2, 1, 16, 7>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 7>,
                                      big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(2, 1, 16, 7));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_dynamic_2D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 1, 16, 7));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_fixed_output_dynamic_2D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 1, (16) + 3, 7>);
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, 4, P>>(
            big_tensor.elements().data(), ntt::fixed_shape_v<2, 1, 16, 7>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 7>,
                                      big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 1, 16, 7));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_fixed_scalar_raw_tensor_shape1) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_fixed_scalar_view_dim2_add3_shape1) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_fixed_1D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_fixed_1D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_fixed_2D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_fixed_2D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_fixed_scalar_raw_tensor_shape2) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, 16, 7));
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, 16, 7>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_fixed_scalar_view_dim2_add3_shape2) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, (16) + 3, 7));
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor.elements().data(), ntt::make_shape(2, 1, 16, 7),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 7),
                                  big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, 16, 7>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<int64_t>(ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_fixed_1D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    auto ntt_input =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_fixed_1D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 1, (16) + 3, 7));
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(2, 1, 16, 7),
            ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 7),
                                      big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_fixed_2D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 1, 16, 7));
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_fixed_2D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 1, (16) + 3, 7));
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(2, 1, 16, 7),
            ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 7),
                                      big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::fixed_shape_v<2, 1, 16, 7>);
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_dynamic_scalar_raw_tensor_shape1) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_dynamic_scalar_view_dim2_add3_shape1) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_dynamic_1D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_dynamic_1D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_dynamic_2D_vector_raw_tensor_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_dynamic_2D_vector_view_dim2_add3_shape1) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 3, 16, 16));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_dynamic_scalar_raw_tensor_shape2) {
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, 16, 7));
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, 16, 7));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_dynamic_scalar_view_dim2_add3_shape2) {
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor =
        ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, (16) + 3, 7));
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input = ntt::make_tensor_view_from_address<int64_t>(
        big_tensor.elements().data(), ntt::make_shape(2, 1, 16, 7),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 7),
                                  big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, 16, 7));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<int64_t>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_dynamic_1D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    auto ntt_input =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(2, 1, 16, 7));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_dynamic_1D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<int64_t, P>>(
        ntt::make_shape(2, 1, (16) + 3, 7));
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, P>>(
            big_tensor.elements().data(), ntt::make_shape(2, 1, 16, 7),
            ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 7),
                                      big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(2, 1, 16, 7));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden =
        ntt::make_tensor<ntt::vector<int64_t, P>>(ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_dynamic_2D_vector_raw_tensor_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    auto ntt_input = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 1, 16, 7));
    NttTest::init_tensor(ntt_input, -1000000, 1000000, true, false);
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 1, 16, 7));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(UnaryTestabs_Int64,
     Int64_input_dynamic_output_dynamic_2D_vector_view_dim2_add3_shape2) {
    constexpr size_t P = NTT_VLEN / (sizeof(int64_t) * 8);
    //---init ntt_input---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 1, (16) + 3, 7));
    NttTest::init_tensor(big_tensor, -1000000, 1000000, true, false);

    auto ntt_input =
        ntt::make_tensor_view_from_address<ntt::vector<int64_t, 4, P>>(
            big_tensor.elements().data(), ntt::make_shape(2, 1, 16, 7),
            ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 7),
                                      big_tensor.strides()));
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 1, 16, 7));

    // Execute unary operation
    ntt::unary<ntt::ops::abs>(ntt_input, ntt_output);

    auto ort_input_org = NttTest::ntt2ort(ntt_input);
    auto ort_input = ort_input_org;
    // Execute Ort operation
    auto ort_output = ortki_Abs(ort_input);

    auto ort_golden = ort_output;
    // ------------------------------------------------------------------
    // 3. convert ORT output back to NTT tensor (golden)
    // ------------------------------------------------------------------
    auto ntt_golden = ntt::make_tensor<ntt::vector<int64_t, 4, P>>(
        ntt::make_shape(2, 1, 16, 7));
    NttTest::ort2ntt(ort_golden, ntt_golden);
    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
