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
#include <cstdint>
#include <gtest/gtest.h>
#include <nncase/ntt/ntt.h>

TEST(SliceNegativeStepTest, CopiesLastAxisWithSignedInputStride) {
    auto input =
        nncase::ntt::make_tensor<int32_t>(nncase::ntt::fixed_shape_v<2, 5>);
    auto output =
        nncase::ntt::make_tensor<int32_t>(nncase::ntt::fixed_shape_v<2, 5>);

    for (nncase::ntt::dim_t row = 0; row < 2; row++) {
        for (nncase::ntt::dim_t col = 0; col < 5; col++) {
            input(row, col) = static_cast<int32_t>(row * 10 + col);
            output(row, col) = -1;
        }
    }

    nncase::ntt::slice(input, output, nncase::ntt::fixed_shape_v<4>,
                       nncase::ntt::fixed_shape_v<-6>,
                       nncase::ntt::fixed_shape_v<1>,
                       nncase::ntt::fixed_shape_v<-1>);

    for (nncase::ntt::dim_t row = 0; row < 2; row++) {
        for (nncase::ntt::dim_t col = 0; col < 5; col++) {
            EXPECT_EQ(output(row, col), input(row, 4 - col));
        }
    }
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
