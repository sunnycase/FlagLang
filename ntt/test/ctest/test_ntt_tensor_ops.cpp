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

#include <gtest/gtest.h>
#include <nncase/ntt/ntt.h>

using namespace nncase;

TEST(TensorOpsTest, AsTensorCreatesOwnedRankZeroTensor) {
    int64_t source = 42;
    auto input = ntt::as_tensor(source);
    source = 7;

    static_assert(decltype(input)::rank() == 0);
    EXPECT_EQ(input(ntt::fixed_shape_v<>), 42);

    auto output = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<>);
    ntt::cast(input, output, ntt::fixed_shape_v<>);

    EXPECT_EQ(output(ntt::fixed_shape_v<>), 42);
}

TEST(TensorOpsTest, AsTensorTemporaryCanFeedBroadcastBinary) {
    auto rhs = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1>);
    rhs(0) = 8;
    auto output = ntt::make_tensor<int64_t>(ntt::fixed_shape_v<1>);

    ntt::binary<ntt::ops::mul>(ntt::as_tensor(static_cast<int64_t>(3)), rhs,
                               output);

    EXPECT_EQ(output(0), 24);
}

TEST(TensorOpsTest, AsTensorPointerCanFeedBroadcastPointerArithmetic) {
    float storage[4] = {};
    auto offsets = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<2>);
    offsets(0) = 1;
    offsets(1) = 3;
    auto output = ntt::make_tensor<float *>(ntt::fixed_shape_v<2>);

    ntt::binary<ntt::ops::add>(ntt::as_tensor(storage), offsets, output);

    EXPECT_EQ(output(0), storage + 1);
    EXPECT_EQ(output(1), storage + 3);
}
