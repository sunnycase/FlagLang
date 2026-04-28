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

static ortki::OrtKITensor *
ortki_inner_product(ortki::OrtKITensor *ort_input_lhs,
                    ortki::OrtKITensor *ort_input_rhs, bool element_is_vec) {
    ortki::OrtKITensor *product_tensor =
        ortki_Mul(ort_input_lhs, ort_input_rhs);
    if (!element_is_vec)
        return product_tensor;
    int64_t axis_data[] = {-1};
    const int64_t axis_shape[] = {1};
    size_t axis_rank = 1;
    auto ort_type = nncase::NttTest::primitive_type2ort_type<int64_t>();
    ortki::OrtKITensor *axes_tensor =
        make_tensor(axis_data, // void* buffer
                    ort_type,
                    axis_shape, // const int64_t* shape
                    axis_rank   // rank
        );
    if (axes_tensor == nullptr) {
        return nullptr;
    }
    int64_t keepdims = 0;
    int64_t noop_with_empty_axes = 0;
    ortki::OrtKITensor *result_tensor = ortki_ReduceSum(
        product_tensor, axes_tensor, keepdims, noop_with_empty_axes);
    return result_tensor;
}
TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_view_dim2_add3_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_view_2_add3_rhs_fixed_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_view_2_add3_rhs_fixed_scalar_view_dim2_add3_no_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim2_add3_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_view_2_add3_rhs_fixed_1D_vector_view_dim2_add3_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_lhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<1>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_view_dim2_add3_lhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<1>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_lhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<1, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim2_add3_lhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<1, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<1>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_view_2_add3_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<1>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<1, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<1, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_view_dim2_add3_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_view_0_add3_rhs_fixed_scalar_raw_tensor_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_view_0_add3_rhs_fixed_scalar_view_dim2_add3_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim2_add3_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_view_0_add3_rhs_fixed_1D_vector_raw_tensor_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_view_0_add3_rhs_fixed_1D_vector_view_dim2_add3_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_view_dim0_add3_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_view_2_add3_rhs_fixed_scalar_raw_tensor_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_view_2_add3_rhs_fixed_scalar_view_dim0_add3_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim0_add3_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_view_2_add3_rhs_fixed_1D_vector_view_dim0_add3_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_multi_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 1>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<1, 3, 1, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 1>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<1, 3, 1, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_fixed_scalar_view_dim2_add3_multi_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 1>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<1, 3, (1) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<1, 3, 1, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<1, 3, 1, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 1>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<1, 3, 1, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_view_2_add3_rhs_fixed_scalar_raw_tensor_multi_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, (16) + 3, 1>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 1, 16, 1>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 1>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<1, 3, 1, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 1>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<1, 3, 1, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_view_2_add3_rhs_fixed_scalar_view_dim2_add3_multi_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, (16) + 3, 1>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 1, 16, 1>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 1>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<1, 3, (1) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<1, 3, 1, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<1, 3, 1, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 1>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<1, 3, 1, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 1, 16, 1>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<1, 3, 1, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 1, 16, 1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, 3, 1, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 1, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<1, 3, 1, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim2_add3_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 1, 16, 1>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<1, 3, (1) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<1, 3, 1, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<1, 3, 1, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 1, 16, 1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, 3, 1, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 1, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<1, 3, 1, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 1, (16) + 3, 1>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 1, 16, 1>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 1>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<1, 3, 1, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 1, 16, 1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, 3, 1, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 1, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<1, 3, 1, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_view_2_add3_rhs_fixed_1D_vector_view_dim2_add3_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 1, (16) + 3, 1>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 1, 16, 1>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 1>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<1, 3, (1) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<1, 3, 1, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<1, 3, 1, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 1, 16, 1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, 3, 1, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 1, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<1, 3, 1, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_view_dim2_add3_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_view_2_add3_rhs_dynamic_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_view_2_add3_rhs_dynamic_scalar_view_dim2_add3_no_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim2_add3_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_view_2_add3_rhs_dynamic_1D_vector_view_dim2_add3_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_lhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<1>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_view_dim2_add3_lhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<1>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_lhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<1, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim2_add3_lhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<1, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(1));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_view_2_add3_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(1));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(1, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(1, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_view_dim2_add3_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_view_0_add3_rhs_dynamic_scalar_raw_tensor_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_view_0_add3_rhs_dynamic_scalar_view_dim2_add3_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim2_add3_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_view_0_add3_rhs_dynamic_1D_vector_raw_tensor_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_view_0_add3_rhs_dynamic_1D_vector_view_dim2_add3_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_view_dim0_add3_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(16),
        ntt::canonicalize_strides(ntt::make_shape(16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_view_2_add3_rhs_dynamic_scalar_raw_tensor_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_view_2_add3_rhs_dynamic_scalar_view_dim0_add3_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(16),
        ntt::canonicalize_strides(ntt::make_shape(16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim0_add3_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(16),
            ntt::canonicalize_strides(ntt::make_shape(16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_view_2_add3_rhs_dynamic_1D_vector_view_dim0_add3_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(16),
            ntt::canonicalize_strides(ntt::make_shape(16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_multi_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 1>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(1, 3, 1, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 1>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(1, 3, 1, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_raw_tensor_rhs_dynamic_scalar_view_dim2_add3_multi_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, 16, 1>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(1, 3, (1) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(1, 3, 1, 16),
        ntt::canonicalize_strides(ntt::make_shape(1, 3, 1, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 1>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(1, 3, 1, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_view_2_add3_rhs_dynamic_scalar_raw_tensor_multi_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, (16) + 3, 1>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 1, 16, 1>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 1>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(1, 3, 1, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 1>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(1, 3, 1, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_scalar_view_2_add3_rhs_dynamic_scalar_view_dim2_add3_multi_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 1, (16) + 3, 1>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 1, 16, 1>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 1>,
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(1, 3, (1) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(1, 3, 1, 16),
        ntt::canonicalize_strides(ntt::make_shape(1, 3, 1, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 1>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(1, 3, 1, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 1, 16, 1>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(1, 3, 1, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 1, 16, 1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, 3, 1, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 1, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(1, 3, 1, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim2_add3_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 1, 16, 1>);
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(1, 3, (1) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(1, 3, 1, 16),
            ntt::canonicalize_strides(ntt::make_shape(1, 3, 1, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 1, 16, 1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, 3, 1, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 1, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(1, 3, 1, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 1, (16) + 3, 1>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 1, 16, 1>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 1>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(1, 3, 1, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 1, 16, 1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, 3, 1, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 1, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(1, 3, 1, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_fixed_1D_vector_view_2_add3_rhs_dynamic_1D_vector_view_dim2_add3_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 1, (16) + 3, 1>);
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::fixed_shape_v<2, 1, 16, 1>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 1, 16, 1>,
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(1, 3, (1) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(1, 3, 1, 16),
            ntt::canonicalize_strides(ntt::make_shape(1, 3, 1, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 1, 16, 1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, 3, 1, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 1, 16, 1, P>);
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(1, 3, 1, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_view_dim2_add3_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_view_2_add3_rhs_fixed_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_view_2_add3_rhs_fixed_scalar_view_dim2_add3_no_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim2_add3_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_view_2_add3_rhs_fixed_1D_vector_view_dim2_add3_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_lhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(1));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_view_dim2_add3_lhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(1));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_lhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(1, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim2_add3_lhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(1, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<1>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_view_2_add3_rhs_fixed_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<1>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<1, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<1>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<1, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(16));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_view_dim2_add3_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(16));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_view_0_add3_rhs_fixed_scalar_raw_tensor_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(16),
        ntt::canonicalize_strides(ntt::make_shape(16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(16));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_view_0_add3_rhs_fixed_scalar_view_dim2_add3_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(16),
        ntt::canonicalize_strides(ntt::make_shape(16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(16));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(16, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim2_add3_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(16, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_view_0_add3_rhs_fixed_1D_vector_raw_tensor_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(16),
            ntt::canonicalize_strides(ntt::make_shape(16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, 16, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(16, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_view_0_add3_rhs_fixed_1D_vector_view_dim2_add3_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(16),
            ntt::canonicalize_strides(ntt::make_shape(16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<2, 3, (16) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<2, 3, 16, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<2, 3, 16, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(16, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<2, 3, 16, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_view_dim0_add3_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_view_2_add3_rhs_fixed_scalar_raw_tensor_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_view_2_add3_rhs_fixed_scalar_view_dim0_add3_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim0_add3_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::fixed_shape_v<16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_view_2_add3_rhs_fixed_1D_vector_view_dim0_add3_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<(16) + 3>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::fixed_shape_v<16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_raw_tensor_multi_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 1));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<1, 3, 1, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 1));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<1, 3, 1, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_fixed_scalar_view_dim2_add3_multi_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 1));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<1, 3, (1) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<1, 3, 1, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<1, 3, 1, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 1));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<1, 3, 1, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_view_2_add3_rhs_fixed_scalar_raw_tensor_multi_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 1, (16) + 3, 1));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 1, 16, 1),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 1),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<1, 3, 1, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 1));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<1, 3, 1, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_view_2_add3_rhs_fixed_scalar_view_dim2_add3_multi_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 1, (16) + 3, 1));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 1, 16, 1),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 1),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::fixed_shape_v<1, 3, (1) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::fixed_shape_v<1, 3, 1, 16>,
        ntt::canonicalize_strides(ntt::fixed_shape_v<1, 3, 1, 16>,
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 1));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<1, 3, 1, 16>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_raw_tensor_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 1, 16, 1));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<1, 3, 1, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 1, 16, 1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, 3, 1, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 1, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<1, 3, 1, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_fixed_1D_vector_view_dim2_add3_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 1, 16, 1));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<1, 3, (1) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<1, 3, 1, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<1, 3, 1, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 1, 16, 1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, 3, 1, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 1, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<1, 3, 1, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_view_2_add3_rhs_fixed_1D_vector_raw_tensor_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 1, (16) + 3, 1));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 1, 16, 1),
            ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 1),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<1, 3, 1, 16>);
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 1, 16, 1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, 3, 1, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 1, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<1, 3, 1, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_view_2_add3_rhs_fixed_1D_vector_view_dim2_add3_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 1, (16) + 3, 1));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 1, 16, 1),
            ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 1),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::fixed_shape_v<1, 3, (1) + 3, 16>);
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::fixed_shape_v<1, 3, 1, 16>,
            ntt::canonicalize_strides(ntt::fixed_shape_v<1, 3, 1, 16>,
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 1, 16, 1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, 3, 1, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 1, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::fixed_shape_v<1, 3, 1, 16, P>);

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_view_dim2_add3_no_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_view_2_add3_rhs_dynamic_scalar_raw_tensor_no_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_view_2_add3_rhs_dynamic_scalar_view_dim2_add3_no_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim2_add3_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_view_2_add3_rhs_dynamic_1D_vector_view_dim2_add3_no_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_lhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(1));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_view_dim2_add3_lhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(1));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_lhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(1, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim2_add3_lhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(1, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(1));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_view_2_add3_rhs_dynamic_scalar_raw_tensor_rhs_singleton_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(1));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(1, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_rhs_singleton_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(1));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(1, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(16));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_view_dim2_add3_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(16));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_view_0_add3_rhs_dynamic_scalar_raw_tensor_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(16),
        ntt::canonicalize_strides(ntt::make_shape(16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(16));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_view_0_add3_rhs_dynamic_scalar_view_dim2_add3_lhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(16),
        ntt::canonicalize_strides(ntt::make_shape(16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(16));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(16, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim2_add3_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(16, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_view_0_add3_rhs_dynamic_1D_vector_raw_tensor_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(16),
            ntt::canonicalize_strides(ntt::make_shape(16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(16, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_view_0_add3_rhs_dynamic_1D_vector_view_dim2_add3_lhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(16),
            ntt::canonicalize_strides(ntt::make_shape(16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<1>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double = ntt::make_tensor<double>(ntt::make_shape(16, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_view_dim0_add3_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(16),
        ntt::canonicalize_strides(ntt::make_shape(16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_view_2_add3_rhs_dynamic_scalar_raw_tensor_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<float_e5m2_t>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_view_2_add3_rhs_dynamic_scalar_view_dim0_add3_rhs_1d_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
        ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(16),
        ntt::canonicalize_strides(ntt::make_shape(16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim0_add3_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, 16, 16));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(16),
            ntt::canonicalize_strides(ntt::make_shape(16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(ntt::make_shape(16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_view_2_add3_rhs_dynamic_1D_vector_view_dim0_add3_rhs_1d_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 3, (16) + 3, 16));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 3, 16, 16),
            ntt::canonicalize_strides(ntt::make_shape(2, 3, 16, 16),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 0)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape((16) + 3));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(16),
            ntt::canonicalize_strides(ntt::make_shape(16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 3, 16, 16, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<1>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<1>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 3, 16, 16, P));
    auto ntt_rhs_double = ntt::make_tensor<double>(ntt::make_shape(16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_raw_tensor_multi_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 1));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(1, 3, 1, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 1));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(1, 3, 1, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_raw_tensor_rhs_dynamic_scalar_view_dim2_add3_multi_broadcast) {
    //---init ntt_input_lhs---
    auto ntt_input_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 1, 16, 1));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(1, 3, (1) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(1, 3, 1, 16),
        ntt::canonicalize_strides(ntt::make_shape(1, 3, 1, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 1));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(1, 3, 1, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_view_2_add3_rhs_dynamic_scalar_raw_tensor_multi_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 1, (16) + 3, 1));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 1, 16, 1),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 1),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(1, 3, 1, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 1));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(1, 3, 1, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_scalar_view_2_add3_rhs_dynamic_scalar_view_dim2_add3_multi_broadcast) {
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 1, (16) + 3, 1));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_lhs.elements().data(), ntt::make_shape(2, 1, 16, 1),
        ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 1),
                                  big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(1, 3, (1) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs = ntt::make_tensor_view_from_address<float_e5m2_t>(
        big_tensor_rhs.elements().data(), ntt::make_shape(1, 3, 1, 16),
        ntt::canonicalize_strides(ntt::make_shape(1, 3, 1, 16),
                                  big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    // 1.1.b for tensors pair that are all tensor of scalar
    auto ntt_input_lhs_aligned = (ntt_input_lhs).view();
    auto ntt_input_rhs_aligned = (ntt_input_rhs).view();
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 1));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(1, 3, 1, 16));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_raw_tensor_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 1, 16, 1));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(1, 3, 1, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 1, 16, 1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, 3, 1, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 1, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(1, 3, 1, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_raw_tensor_rhs_dynamic_1D_vector_view_dim2_add3_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    auto ntt_input_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 1, 16, 1));
    NttTest::init_tensor(ntt_input_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(1, 3, (1) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(1, 3, 1, 16),
            ntt::canonicalize_strides(ntt::make_shape(1, 3, 1, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 1, 16, 1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, 3, 1, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 1, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(1, 3, 1, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_view_2_add3_rhs_dynamic_1D_vector_raw_tensor_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 1, (16) + 3, 1));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 1, 16, 1),
            ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 1),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    auto ntt_input_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(1, 3, 1, 16));
    NttTest::init_tensor(ntt_input_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 1, 16, 1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, 3, 1, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 1, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(1, 3, 1, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

TEST(
    BinaryTestinner_product_Float8e5m2,
    Float8e5m2_lhs_dynamic_1D_vector_view_2_add3_rhs_dynamic_1D_vector_view_dim2_add3_multi_broadcast) {
    constexpr size_t P = NTT_VLEN / (sizeof(float_e5m2_t) * 8);
    //---init ntt_input_lhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_lhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(2, 1, (16) + 3, 1));
    NttTest::init_tensor(big_tensor_lhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_lhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_lhs.elements().data(), ntt::make_shape(2, 1, 16, 1),
            ntt::canonicalize_strides(ntt::make_shape(2, 1, 16, 1),
                                      big_tensor_lhs.strides()));
    //---init ntt_input_rhs---
    // Create non-contiguous tensor (on dimension 2)
    auto big_tensor_rhs = ntt::make_tensor<ntt::vector<float_e5m2_t, P>>(
        ntt::make_shape(1, 3, (1) + 3, 16));
    NttTest::init_tensor(big_tensor_rhs, float_e5m2_t(-32.0f),
                         float_e5m2_t(32.0f), true, false);

    auto ntt_input_rhs =
        ntt::make_tensor_view_from_address<ntt::vector<float_e5m2_t, P>>(
            big_tensor_rhs.elements().data(), ntt::make_shape(1, 3, 1, 16),
            ntt::canonicalize_strides(ntt::make_shape(1, 3, 1, 16),
                                      big_tensor_rhs.strides()));
    //---generate output tensor---
    // ------------------------------------------------------------------
    // 2. call NTT operation to get NTT output (under test)
    // ------------------------------------------------------------------
    // Create output tensor
    auto ntt_output =
        ntt::make_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));

    // Execute binary operation
    ntt::binary<ntt::ops::inner_product>(ntt_input_lhs, ntt_input_rhs,
                                         ntt_output);

    // align in NTT, then cast to double, then process in ORT
    auto ntt_input_lhs_unsqueezed = (ntt_input_lhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_lhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<2, 1, 16, 1, P>);
    ntt::unpack(ntt_input_lhs_unsqueezed, ntt_input_lhs_aligned,
                fixed_shape_v<4>);
    auto ntt_input_rhs_unsqueezed = (ntt_input_rhs).unsqueeze(fixed_shape_v<4>);
    auto ntt_input_rhs_aligned =
        ntt::make_tensor<float_e5m2_t>(fixed_shape_v<1, 3, 1, 16, P>);
    ntt::unpack(ntt_input_rhs_unsqueezed, ntt_input_rhs_aligned,
                fixed_shape_v<4>);
    // 1.2 get ntt_lhs/rhs_double
    auto ntt_lhs_double =
        ntt::make_tensor<double>(ntt::make_shape(2, 1, 16, 1, P));
    auto ntt_rhs_double =
        ntt::make_tensor<double>(ntt::make_shape(1, 3, 1, 16, P));

    ntt::cast(ntt_input_lhs_aligned, ntt_lhs_double);
    ntt::cast(ntt_input_rhs_aligned, ntt_rhs_double);

    // 2. calculated ort_output
    auto [ort_input_lhs, ort_input_rhs] = NttTest::convert_and_align_to_ort(
        ntt_lhs_double, ntt_rhs_double, false, false);
    // Execute Ort operation
    static bool element_is_vec =
        ntt::Vector<typename decltype(ntt_input_lhs)::element_type>;
    auto ort_output =
        ortki_inner_product(ort_input_lhs, ort_input_rhs, element_is_vec);

    auto ort_golden_double = ort_output;
    //  transform ort_golden_double to ntt_goldenfloat_e5m2_t_scalar
    auto ntt_golden_double_scalar =
        ntt::make_unique_tensor<double>(ntt::make_shape(2, 3, 16, 16));
    NttTest::ort2ntt(ort_golden_double, *ntt_golden_double_scalar);

    auto ntt_golden_float_e5m2_t_scalar =
        ntt::make_unique_tensor<float_e5m2_t>(ntt::make_shape(2, 3, 16, 16));
    ntt::cast(*ntt_golden_double_scalar, *ntt_golden_float_e5m2_t_scalar);

    // 4.a if ntt_output is not tensor of vector
    auto ntt_golden = *ntt_golden_float_e5m2_t_scalar;

    EXPECT_TRUE(NttTest::compare_tensor(ntt_output, ntt_golden, 1));
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
