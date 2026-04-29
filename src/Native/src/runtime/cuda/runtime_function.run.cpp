/* Copyright 2019-2021 Canaan Inc.
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
#include "runtime_function.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <nncase/ntt/arch/cuda/runtime.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/type_serializer.h>
#include <utility>
#include <vector>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cuda;
using namespace nncase::ntt::runtime;

#define WARP_SIZE 32

namespace {
class cuda_device_params {
  public:
    cuda_device_params() noexcept = default;
    cuda_device_params(cuda_device_params &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)) {}
    cuda_device_params(const cuda_device_params &) = delete;

    ~cuda_device_params() { reset(); }

    cuda_device_params &operator=(cuda_device_params &&other) noexcept {
        if (this != &other) {
            reset();
            data_ = std::exchange(other.data_, nullptr);
        }

        return *this;
    }

    cuda_device_params &operator=(const cuda_device_params &) = delete;

    result<void> copy_from(const cuda_block_entry_params_t &params) noexcept {
        reset();
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&data_),
                              sizeof(cuda_block_entry_params_t)));
        CHECK_CUDA(cudaMemcpy(data_, &params, sizeof(cuda_block_entry_params_t),
                              cudaMemcpyHostToDevice));
        return ok();
    }

    cuda_block_entry_params_t *data() const noexcept { return data_; }

  private:
    void reset() noexcept {
        if (!data_) {
            return;
        }

        auto status = cudaFree(data_);
        if (status != cudaSuccess) {
            std::cerr << "CUDA error during cudaFree - "
                      << cudaGetErrorString(status) << std::endl;
        }

        data_ = nullptr;
    }

  private:
    cuda_block_entry_params_t *data_ = nullptr;
};
} // namespace

result<void> cuda_runtime_function::run(const thread_inout_desc *input_descs,
                                        thread_inout_desc *output_descs,
                                        std::byte *output_data) noexcept {
    CHECK_WITH_ERR(module().cdim() == 1, std::errc::not_supported);

    auto enable_profiling = module()
                                .interp()
                                .options()
                                .get_scalar_opt<uint8_t>("enable_profiling")
                                .or_(false);
    CHECK_WITH_ERR(!enable_profiling, std::errc::not_supported);

    std::vector<cuda_device_params> params_blocks(module().cdim());
    for (size_t cid = 0; cid < module().cdim(); cid++) {
        CHECK_CUDA(cudaSetDevice(cid));

        cuda_block_entry_params_t src_params{
            .tdim = module().tdim(),
            .bdim = module().bdim(),
            .cdim = module().cdim(),
            .cid = cid,
            .enable_profiling = enable_profiling,
            .input_descs = input_descs,
            .output_descs = output_descs,
            .rdata = module().rdata(),
            .output = output_data,
            .thread_local_rdata_header = module().thread_local_rdata_header(
                cid * module().bdim() * module().wdim() * module().tdim()),
            .thread_local_rdata = module().thread_local_rdata_content(),
            .warp_local_rdata_header = module().warp_local_rdata_header(
                cid * module().bdim() * module().wdim()),
            .warp_local_rdata = module().warp_local_rdata_content(),
            .block_local_rdata_header =
                module().block_local_rdata_header(cid * module().bdim()),
            .block_local_rdata = module().block_local_rdata_content(),
            .thread_local_data = thread_local_data(cid),
            .warp_local_data = warp_local_data(cid),
            .block_local_data = block_local_data(cid),
            .profile_records = enable_profiling
                                   ? thread_local_profile_records(cid)
                                   : std::span<ntt::runtime::profile_record>{},
            .profile_record_counts =
                enable_profiling
                    ? thread_local_profile_record_counts(cid).data()
                    : nullptr,
        };
        try_(params_blocks[cid].copy_from(src_params));
    }

    std::vector<size_t> launched_cids;
    for (size_t cid = 0; cid < module().cdim(); cid++) {
        CHECK_CUDA(cudaSetDevice(cid));
        auto params = params_blocks[cid].data();
        void *args[] = {&params};
        auto status = cudaLaunchKernel(
            (const void *)block_entry_, dim3(module().bdim()),
            dim3(module().wdim() * module().tdim()), args, 0, nullptr);
        if (status != cudaSuccess) {
            for (auto launched_cid : launched_cids) {
                CHECK_CUDA(cudaSetDevice(launched_cid));
                CHECK_CUDA(cudaDeviceSynchronize());
            }

            CHECK_CUDA(status);
        }

        launched_cids.push_back(cid);
    }

    for (auto cid : launched_cids) {
        CHECK_CUDA(cudaSetDevice(cid));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    return ok();
}
