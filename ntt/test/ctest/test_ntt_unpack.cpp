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

TEST(UnpackTest, ContiguousUnpackSkipsPaddedTailLane) {
    using TVec = ntt::vector<uint32_t, 4>;
    constexpr uint32_t Sentinel = 0xdeadbeefU;

    auto input = ntt::make_tensor<TVec>(ntt::fixed_shape_v<1>);
    input(0)(0) = 10;
    input(0)(1) = 11;
    input(0)(2) = 12;
    input(0)(3) = 99;

    auto storage = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<4>);
    for (auto &item : storage.elements()) {
        item = Sentinel;
    }

    auto output = ntt::make_tensor_view_from_address<uint32_t>(
        storage.elements().data(), ntt::fixed_shape_v<3>,
        ntt::fixed_strides_v<1>);

    ntt::unpack(input, output, ntt::fixed_shape_v<0>);

    EXPECT_EQ(storage(0), 10);
    EXPECT_EQ(storage(1), 11);
    EXPECT_EQ(storage(2), 12);
    EXPECT_EQ(storage(3), Sentinel);
}

TEST(UnpackTest, FallbackUnpackSkipsPaddedTailLane) {
    using TVec = ntt::vector<uint32_t, 4>;
    constexpr uint32_t Sentinel = 0xdeadbeefU;

    auto input = ntt::make_tensor<TVec>(ntt::fixed_shape_v<1>);
    input(0)(0) = 20;
    input(0)(1) = 21;
    input(0)(2) = 22;
    input(0)(3) = 88;

    auto storage = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<8>);
    for (auto &item : storage.elements()) {
        item = Sentinel;
    }

    auto output = ntt::make_tensor_view_from_address<uint32_t>(
        storage.elements().data(), ntt::fixed_shape_v<3>,
        ntt::fixed_strides_v<2>);

    ntt::unpack(input, output, ntt::fixed_shape_v<0>);

    EXPECT_EQ(storage(0), 20);
    EXPECT_EQ(storage(2), 21);
    EXPECT_EQ(storage(4), 22);
    EXPECT_EQ(storage(6), Sentinel);
}
