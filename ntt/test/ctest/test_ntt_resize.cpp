/* Copyright 2019-2025 Canaan Inc.
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

TEST(ResizeTest, BilinearInt32KeepsFractionalWeights) {
    auto input = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<1, 1, 2, 2>);
    input(0, 0, 0, 0) = 0;
    input(0, 0, 0, 1) = 10;
    input(0, 0, 1, 0) = 20;
    input(0, 0, 1, 1) = 30;

    auto output = ntt::make_tensor<int32_t>(ntt::fixed_shape_v<1, 1, 1, 1>);
    ntt::resize(input, output, ntt::fixed_shape_v<>, ntt::fixed_shape_v<>,
                ntt::fixed_shape_v<1, 1, 1, 1>,
                ntt::image_resize_mode_t::bilinear,
                ntt::image_resize_transformation_mode_t::half_pixel,
                ntt::image_resize_nearest_mode_t::round_prefer_floor);

    EXPECT_EQ(output(0, 0, 0, 0), 15);
}

TEST(ResizeTest, BilinearUsesPytorchHalfPixelCoordinates) {
    auto input = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 1, 2, 2>);
    input(0, 0, 0, 0) = 0.f;
    input(0, 0, 0, 1) = 10.f;
    input(0, 0, 1, 0) = 20.f;
    input(0, 0, 1, 1) = 30.f;

    auto output = ntt::make_tensor<float>(ntt::fixed_shape_v<1, 1, 4, 4>);
    ntt::resize(input, output, ntt::fixed_shape_v<>, ntt::fixed_shape_v<>,
                ntt::fixed_shape_v<1, 1, 4, 4>,
                ntt::image_resize_mode_t::bilinear,
                ntt::image_resize_transformation_mode_t::pytorch_half_pixel,
                ntt::image_resize_nearest_mode_t::round_prefer_floor);

    EXPECT_NEAR(output(0, 0, 1, 1), 7.5f, 1e-6f);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
