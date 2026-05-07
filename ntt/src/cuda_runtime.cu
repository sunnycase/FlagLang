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
#include <cstddef>
#include <cstring>
#include <nncase/ntt/arch/cuda/distributed.h>
#include <nncase/ntt/arch/cuda/runtime.h>
#include <nncase/ntt/distributed.h>
#include <nncase/ntt/profiling.h>
#include <nncase/ntt/shape.h>
#include <nncase/ntt/vector.h>

using namespace nncase;
using namespace nncase::ntt;
using namespace nncase::ntt::distributed;
using namespace nncase::ntt::runtime;

namespace nncase::ntt::distributed::detail {
__device__ uintptr_t global_local_data_ptr[topology_element_count * 2];
__device__ uintptr_t global_thread_local_rdata_ptr[topology_element_count * 2];
__device__ uintptr_t global_thread_local_cache_ptr[topology_element_count * 3];
__device__ uintptr_t global_block_local_rdata_ptr[topology_element_count * 2];
} // namespace nncase::ntt::distributed::detail

namespace nncase::ntt::runtime {
alignas(cuda_thread_context_t) __shared__ std::byte
    cuda_thread_contexts_storage[sizeof(cuda_thread_context_t) * tdim() *
                                 wdim()];

__device__ bool is_profiling_enabled() noexcept {
    return cuda_thread_context_t::current().enable_profiling;
}

__device__ uint64_t get_profile_time() noexcept { return clock64(); }

__device__ void record_profile(profile_level level,
                               const profile_record &record) noexcept {
    // Other levels are not supported yet.
    if (level == profile_level::kernel) {
        auto &ctx = cuda_thread_context_t::current();
        auto idx = ctx.profile_record_counts[0]++;
        ctx.profile_records[idx] = record;
    }
}
} // namespace nncase::ntt::runtime

__device__ cuda_thread_context_t &cuda_thread_context_t::current() noexcept {
    auto &cuda_thread_contexts =
        *reinterpret_cast<cuda_thread_context_t(*)[tdim() * wdim()]>(
            cuda_thread_contexts_storage);
    return cuda_thread_contexts[wid() * tdim() + tid()];
}

extern "C" __global__ void
block_entry(const cuda_block_entry_params_t &params) {
    const auto linear_wid = bid() * wdim() + wid();
    const auto linear_tid = linear_wid * tdim() + tid();

    // Get thread local rdata
    auto thread_local_rdata_offset =
        (size_t)params.thread_local_rdata_header[linear_tid * 2];
    auto thread_local_rdata_size =
        (size_t)params.thread_local_rdata_header[linear_tid * 2 + 1];
    auto thread_local_rdata = params.thread_local_rdata.subspan(
        thread_local_rdata_offset, thread_local_rdata_size);

    // Get thread local cache
    auto thread_local_cache_offset =
        (size_t)params.thread_local_cache_header[linear_tid * 2];
    auto thread_local_cache_size =
        (size_t)params.thread_local_cache_header[linear_tid * 2 + 1];
    auto thread_local_cache = params.thread_local_cache.subspan(
        thread_local_cache_offset, thread_local_cache_size);

    // Get thread local data
    auto thread_local_chip_data = params.thread_local_data;
    const auto thread_local_data_size =
        thread_local_chip_data.size_bytes() / (bdim() * wdim() * tdim());
    auto thread_local_data = thread_local_chip_data.subspan(
        thread_local_data_size * linear_tid, thread_local_data_size);

    // Get warp local rdata
    auto warp_local_rdata_offset =
        (size_t)params.warp_local_rdata_header[linear_wid * 2];
    auto warp_local_rdata_size =
        (size_t)params.warp_local_rdata_header[linear_wid * 2 + 1];
    auto warp_local_rdata = params.warp_local_rdata.subspan(
        warp_local_rdata_offset, warp_local_rdata_size);

    // Get warp local data
    auto warp_local_chip_data = params.warp_local_data;
    const auto warp_local_data_size =
        warp_local_chip_data.size_bytes() / (bdim() * wdim());
    auto warp_local_data = warp_local_chip_data.subspan(
        warp_local_data_size * linear_wid, warp_local_data_size);

    // Get block local rdata
    auto block_local_rdata_offset =
        (size_t)params.block_local_rdata_header[bid() * 2];
    auto block_local_rdata_size =
        (size_t)params.block_local_rdata_header[bid() * 2 + 1];
    auto block_local_rdata = params.block_local_rdata.subspan(
        block_local_rdata_offset, block_local_rdata_size);

    // Get block local data
    auto block_local_chip_data = params.block_local_data;
    const auto block_local_data_size =
        block_local_chip_data.size_bytes() / bdim();
    auto block_local_data = block_local_chip_data.subspan(
        block_local_data_size * bid(), block_local_data_size);

    // Get thread local profile records
    auto block_profile_records = params.profile_records;
    const auto profile_records_size =
        block_profile_records.size() / (bdim() * wdim() * tdim());
    auto profile_records = block_profile_records.subspan(
        profile_records_size * linear_tid, profile_records_size);

    const auto program_ids = make_shape(params.cid, bid(), wid(), tid());

    // Set distributed pointers
    auto thread_local_rdata_desc =
        ntt::distributed::detail::global_thread_local_rdata_desc(program_ids);
    thread_local_rdata_desc[0] = (uintptr_t)thread_local_rdata.data();
    thread_local_rdata_desc[1] = (uintptr_t)(thread_local_rdata.data() +
                                             thread_local_rdata.size_bytes());
    auto local_data_desc =
        ntt::distributed::detail::global_local_data_desc(program_ids);
    local_data_desc[0] = (uintptr_t)thread_local_data.data();
    local_data_desc[1] =
        (uintptr_t)(thread_local_data.data() + thread_local_data.size_bytes());
    auto thread_local_cache_desc =
        ntt::distributed::detail::global_thread_local_cache_desc(program_ids);
    std::array<uintptr_t, 3> thread_local_cache_ptrs{};
    for (size_t i = 0; i < params.thread_local_cache_starts.size(); i++) {
        if (params.thread_local_cache_starts[i] >= 0) {
            thread_local_cache_ptrs[i] =
                (uintptr_t)(thread_local_cache.data() +
                            params.thread_local_cache_starts[i]);
            thread_local_cache_desc[i] = thread_local_cache_ptrs[i];
        } else {
            thread_local_cache_desc[i] = 0;
        }
    }
    auto block_local_rdata_desc =
        ntt::distributed::detail::global_block_local_rdata_desc(program_ids);
    block_local_rdata_desc[0] = (uintptr_t)params.block_local_rdata.data();
    block_local_rdata_desc[1] =
        (uintptr_t)(params.block_local_rdata.data() +
                    params.block_local_rdata.size_bytes());

    cuda_thread_context_t::current() = {
        .cid = params.cid,
        .enable_profiling = params.enable_profiling,
        .profile_records = profile_records,
        .profile_record_counts = params.profile_record_counts + linear_tid,
        .thread_local_cache_ptrs = thread_local_cache_ptrs};

    distributed::topology_synchronize();
    thread_main(params.input_descs, params.output_descs, params.rdata.data(),
                thread_local_rdata.data(), warp_local_rdata.data(),
                block_local_rdata.data(), thread_local_data.data(),
                warp_local_data.data(), block_local_data.data(), params.output);
}

int main() {
    cuda_block_entry_params_t params = {};
    block_entry<<<1, 1>>>(params);
}
