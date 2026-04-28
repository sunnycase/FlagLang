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

TEST(PackTest, PackAxis1HonorsStridedOutput) {
    constexpr size_t P = NTT_VLEN / (sizeof(uint32_t) * 8);
    constexpr size_t C = 2;
    constexpr size_t H = P * 3;
    constexpr size_t W = 5;
    constexpr uint32_t Sentinel = 0xdeadbeefU;

    auto input = ntt::make_tensor<uint32_t>(ntt::fixed_shape_v<C, H, W>);
    uint32_t value = 1;
    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                input(c, h, w) = value++;
            }
        }
    }

    auto storage = ntt::make_tensor<ntt::vector<uint32_t, P>>(
        ntt::fixed_shape_v<C, H / P, W * 2>);
    for (auto &item : storage.elements()) {
        item = ntt::vector<uint32_t, P>(Sentinel);
    }

    auto output = ntt::make_tensor_view_from_address<ntt::vector<uint32_t, P>>(
        storage.elements().data(), ntt::fixed_shape_v<C, H / P, W>,
        ntt::fixed_strides_v<(H / P) * (W * 2), W * 2, 2>);

    ntt::pack(input, output, ntt::fixed_shape_v<1>);

    for (size_t c = 0; c < C; c++) {
        for (size_t hb = 0; hb < H / P; hb++) {
            for (size_t w = 0; w < W; w++) {
                for (size_t lane = 0; lane < P; lane++) {
                    EXPECT_EQ(storage(c, hb, w * 2)(lane),
                              input(c, hb * P + lane, w));
                    EXPECT_EQ(storage(c, hb, w * 2 + 1)(lane), Sentinel);
                }
            }
        }
    }
}

TEST(PackTest, PackLastAxisPadsEachRowTail) {
    using TVec = ntt::vector<uint64_t, 2>;

    auto input = ntt::make_tensor<uint64_t>(ntt::fixed_shape_v<2, 3>);
    uint64_t value = 1;
    for (size_t row = 0; row < 2; row++) {
        for (size_t col = 0; col < 3; col++) {
            input(row, col) = value++;
        }
    }

    auto output = ntt::make_tensor<TVec>(ntt::fixed_shape_v<2, 2>);
    for (auto &item : output.elements()) {
        item = TVec(UINT64_MAX);
    }

    ntt::pack(input, output, ntt::fixed_shape_v<1>);

    EXPECT_EQ(output(0, 0)(0), 1);
    EXPECT_EQ(output(0, 0)(1), 2);
    EXPECT_EQ(output(0, 1)(0), 3);
    EXPECT_EQ(output(0, 1)(1), 0);
    EXPECT_EQ(output(1, 0)(0), 4);
    EXPECT_EQ(output(1, 0)(1), 5);
    EXPECT_EQ(output(1, 1)(0), 6);
    EXPECT_EQ(output(1, 1)(1), 0);
}
