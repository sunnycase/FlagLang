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
#include "nncase/runtime/buffer.h"
#include <algorithm>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <nncase/llm/paged_attention_kv_cache.h>
#include <nncase/ntt/arch/cuda/runtime.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>
#include <nncase/type.h>
#include <span>
#include <utility>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cuda;
using namespace nncase::ntt::runtime;

typedef struct {
    uint32_t output_align;
    uint32_t local_data_align;
    uint64_t output_pool_size;
    uint64_t thread_local_data_pool_size;
    uint64_t warp_local_data_pool_size;
    uint64_t block_local_data_pool_size;
} kernel_desc_header;

namespace {
result<void> check_cuda_status(cudaError_t status,
                               const char *operation) noexcept {
    if (status == cudaSuccess) {
        return ok();
    }

    std::cerr << "CUDA error during " << operation << " - "
              << cudaGetErrorString(status) << std::endl;
    return err(std::errc::io_error);
}

class cuda_device_allocation {
  public:
    cuda_device_allocation() noexcept = default;

    cuda_device_allocation(cuda_device_allocation &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0)) {}

    cuda_device_allocation(const cuda_device_allocation &) = delete;

    ~cuda_device_allocation() { reset(); }

    cuda_device_allocation &operator=(cuda_device_allocation &&other) noexcept {
        if (this != &other) {
            reset();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }

        return *this;
    }

    cuda_device_allocation &operator=(const cuda_device_allocation &) = delete;

    result<void> allocate(size_t size) noexcept {
        reset();
        size_ = size;
        if (!size_) {
            return ok();
        }

        void *data = nullptr;
        try_(check_cuda_status(cudaMalloc(&data, size_), "cudaMalloc"));
        data_ = static_cast<std::byte *>(data);
        return ok();
    }

    result<void> copy_from(std::span<const std::byte> source) noexcept {
        CHECK_WITH_ERR(source.size_bytes() == size_,
                       std::errc::invalid_argument);
        if (source.empty()) {
            return ok();
        }

        CHECK_WITH_ERR(data_ != nullptr, std::errc::invalid_argument);
        return check_cuda_status(cudaMemcpy(data_, source.data(),
                                            source.size_bytes(),
                                            cudaMemcpyHostToDevice),
                                 "cudaMemcpyHostToDevice");
    }

    result<void> copy_to(std::span<std::byte> destination) const noexcept {
        CHECK_WITH_ERR(destination.size_bytes() == size_,
                       std::errc::invalid_argument);
        if (destination.empty()) {
            return ok();
        }

        CHECK_WITH_ERR(data_ != nullptr, std::errc::invalid_argument);
        return check_cuda_status(cudaMemcpy(destination.data(), data_,
                                            destination.size_bytes(),
                                            cudaMemcpyDeviceToHost),
                                 "cudaMemcpyDeviceToHost");
    }

    std::byte *data() const noexcept { return data_; }
    size_t size() const noexcept { return size_; }

  private:
    void reset() noexcept {
        if (!data_) {
            size_ = 0;
            return;
        }

        auto status = cudaFree(data_);
        if (status != cudaSuccess) {
            std::cerr << "CUDA error during cudaFree - "
                      << cudaGetErrorString(status) << std::endl;
        }

        data_ = nullptr;
        size_ = 0;
    }

  private:
    std::byte *data_ = nullptr;
    size_t size_ = 0;
};

struct staged_mapped_buffer {
    staged_mapped_buffer(mapped_buffer host_map,
                         cuda_device_allocation device_data,
                         bool copy_back) noexcept
        : host_map(std::move(host_map)),
          device_data(std::move(device_data)),
          copy_back(copy_back) {}

    staged_mapped_buffer(staged_mapped_buffer &&) noexcept = default;
    staged_mapped_buffer(const staged_mapped_buffer &) = delete;
    staged_mapped_buffer &operator=(staged_mapped_buffer &&) noexcept = default;
    staged_mapped_buffer &operator=(const staged_mapped_buffer &) = delete;

    mapped_buffer host_map;
    cuda_device_allocation device_data;
    bool copy_back;
};

template <class T>
result<void> stage_array(cuda_device_allocation &allocation,
                         std::span<const T> source) noexcept {
    try_(allocation.allocate(source.size_bytes()));
    return allocation.copy_from(std::as_bytes(source));
}

template <class T>
result<void> copy_array_from_device(const cuda_device_allocation &allocation,
                                    std::span<T> destination) noexcept {
    return allocation.copy_to(std::as_writable_bytes(destination));
}

bool contains_address_range(std::byte *base, size_t size, std::byte *data,
                            size_t data_size) noexcept {
    auto base_addr = reinterpret_cast<uintptr_t>(base);
    auto data_addr = reinterpret_cast<uintptr_t>(data);
    return data_addr >= base_addr && data_size <= size &&
           data_addr - base_addr <= size - data_size;
}
} // namespace

cuda_runtime_function::cuda_runtime_function(runtime_module &rt_module)
    : runtime_function(rt_module), block_entry_(nullptr) {}

cuda_runtime_function::~cuda_runtime_function() {
    for (size_t cid = 0; cid < thread_local_datas_.size(); cid++) {
        auto status = cudaSetDevice(cid);
        if (status != cudaSuccess) {
            std::cerr << "CUDA error during cudaSetDevice - "
                      << cudaGetErrorString(status) << std::endl;
            continue;
        }

        auto free_device_span = [](std::span<std::byte> span) noexcept {
            if (span.data()) {
                auto free_status = cudaFree(span.data());
                if (free_status != cudaSuccess) {
                    std::cerr << "CUDA error during cudaFree - "
                              << cudaGetErrorString(free_status) << std::endl;
                }
            }
        };

        free_device_span(thread_local_datas_[cid]);
        free_device_span(warp_local_datas_[cid]);
        free_device_span(block_local_datas_[cid]);
    }
}

cuda_runtime_module &cuda_runtime_function::module() const noexcept {
    return static_cast<cuda_runtime_module &>(runtime_function::module());
}

result<void> cuda_runtime_function::initialize_core(
    runtime_function_init_context &context) noexcept {
    try_(context.read_section(
        ".desc", [this](auto reader, size_t) -> result<void> {
            auto header = reader.template read<kernel_desc_header>();

            // Allocate output buffer
            buffer_allocate_options options{};
            options.flags = HOST_BUFFER_ALLOCATE_CPU_ONLY;
            options.alignment = header.output_align;
            try_var(output_buffer, buffer_allocator::host().allocate(
                                       header.output_pool_size, options));
            try_set(this->output_buffer_,
                    output_buffer.template as<host_buffer_t>());

            const size_t thread_local_data_size =
                header.thread_local_data_pool_size * module().tdim() *
                module().wdim() * module().bdim();
            const size_t warp_local_data_size =
                header.warp_local_data_pool_size * module().wdim() *
                module().bdim();
            const size_t block_local_data_size =
                header.block_local_data_pool_size * module().bdim();
            for (size_t cid = 0; cid < module().cdim(); cid++) {
                CHECK_CUDA(cudaSetDevice(cid));

                // Allocate thread local datas
                std::byte *thread_local_data_dev_ptr;
                CHECK_CUDA(cudaMalloc((void **)&thread_local_data_dev_ptr,
                                      thread_local_data_size));
                thread_local_datas_.emplace_back(thread_local_data_dev_ptr,
                                                 thread_local_data_size);

                // Allocate warp local datas
                std::byte *warp_local_data_dev_ptr;
                CHECK_CUDA(cudaMalloc((void **)&warp_local_data_dev_ptr,
                                      warp_local_data_size));
                warp_local_datas_.emplace_back(warp_local_data_dev_ptr,
                                               warp_local_data_size);

                // Allocate block local datas
                std::byte *block_local_data_dev_ptr;
                CHECK_CUDA(cudaMalloc((void **)&block_local_data_dev_ptr,
                                      block_local_data_size));
                block_local_datas_.emplace_back(block_local_data_dev_ptr,
                                                block_local_data_size);
            }
            return ok();
        }));
    try_set(block_entry_, module().block_entry());

    // Allocate input descs
    auto input_size = parameters_size();
    input_descs_.resize(input_size);

    // Allocate output descs
    auto output_size = return_size();
    output_descs_.resize(output_size);
    output_shapes_.resize(output_size);
    output_strides_.resize(output_size);
    for (size_t i = 0; i < output_size; i++) {
        try_var(type, return_type(i));
        try_var(ttype, type.as<tensor_type>());
        auto rank = ttype->shape().rank();
        CHECK_WITH_ERR(rank.has_value(), std::errc::invalid_argument);
        output_shapes_[i].resize(*rank);
        output_strides_[i].resize(*rank);
        output_descs_[i] = thread_inout_desc{
            .data = nullptr,
            .size = 0,
            .shape = output_shapes_[i].data(),
            .strides = output_strides_[i].data(),
            .rank = output_shapes_[i].size(),
        };
    }

    // Allocate profiling records
    if (module()
            .interp()
            .options()
            .get_scalar_opt<uint8_t>("enable_profiling")
            .or_(false)) {
        // profile_records_.resize(blocks_count);
        // profile_record_counts_.resize(blocks_count);
        // for (size_t i = 0; i < blocks_count; i++) {
        //     profile_records_[i].resize(module().tdim() *
        //                                default_profile_record_count);
        //     profile_record_counts_[i].resize(module().tdim());
        // }
    }

    return ok();
}

result<value_t> cuda_runtime_function::invoke_core(
    std::span<value_t> parameters,
    [[maybe_unused]] value_t return_value) noexcept {
    CHECK_WITH_ERR(module().cdim() == 1, std::errc::not_supported);
    CHECK_CUDA(cudaSetDevice(0));

    size_t input_id = 0;
    std::vector<thread_inout_desc> input_descs(input_descs_.size());
    std::vector<thread_inout_desc> output_descs(output_descs_.size());
    std::vector<staged_mapped_buffer> staged_buffers;
    std::vector<cuda_device_allocation> input_shape_buffers;
    std::vector<cuda_device_allocation> input_stride_buffers;
    std::vector<cuda_device_allocation> output_shape_buffers;
    std::vector<cuda_device_allocation> output_stride_buffers;
    std::vector<cuda_device_allocation> paged_kv_cache_desc_buffers;

    auto stage_tensor_buffer = [&](const tensor &source, map_access_t access,
                                   bool copy_back, std::byte **device_data,
                                   size_t *device_size) -> result<void> {
        try_var(host_buffer, source->buffer().as_host());
        try_var(mapped, host_buffer.map(access));
        cuda_device_allocation device_buffer;
        try_(device_buffer.allocate(mapped.buffer().size_bytes()));
        try_(device_buffer.copy_from(mapped.buffer()));
        *device_data = device_buffer.data();
        *device_size = device_buffer.size();
        staged_buffers.emplace_back(std::move(mapped), std::move(device_buffer),
                                    copy_back);
        return ok();
    };

    auto stage_tensor_shape =
        [](const tensor &source, cuda_device_allocation &shape_buffer,
           cuda_device_allocation &stride_buffer, size_t **device_shape,
           size_t **device_strides) -> result<void> {
        auto shape = source->shape();
        auto strides = source->strides();
        try_(stage_array(shape_buffer,
                         std::span<const size_t>(shape.data(), shape.size())));
        try_(stage_array(stride_buffer, std::span<const size_t>(
                                            strides.data(), strides.size())));
        *device_shape = reinterpret_cast<size_t *>(shape_buffer.data());
        *device_strides = reinterpret_cast<size_t *>(stride_buffer.data());
        return ok();
    };

    for (auto arg : parameters) {
        try_var(t, arg.as<tensor>());
        cuda_device_allocation shape_buffer;
        cuda_device_allocation stride_buffer;
        size_t *device_shape;
        size_t *device_strides;
        try_(stage_tensor_shape(t, shape_buffer, stride_buffer, &device_shape,
                                &device_strides));
        input_shape_buffers.emplace_back(std::move(shape_buffer));
        input_stride_buffers.emplace_back(std::move(stride_buffer));

        if (t->dtype().is_a<reference_type_t>()) {
            auto rt = t->dtype().as<reference_type_t>().expect(
                "now only support reference value type!");
            auto vt = rt->elemtype().as<value_type_t>().expect(
                "now only support reference value type!");
            if (vt->uuid() == datatype_t::paged_attention_kv_cache->uuid()) {
                try_var(hb, t->buffer().as_host());
                try_var(m, hb.map(map_read));
                auto refspan =
                    as_span<llm::paged_attention_kv_cache_node *>(m.buffer());
                std::vector<thread_paged_attention_kv_cache_desc> descs(
                    refspan.size());
                for (size_t i = 0; i < refspan.size(); i++) {
                    auto &node = refspan[i];
                    auto &desc = descs[i];
                    {
                        desc.num_seqs = node->num_seqs();
                        desc.num_tokens = node->num_tokens();
                        {
                            std::byte *device_data;
                            size_t device_size;
                            try_(stage_tensor_buffer(
                                node->context_lens(), map_read, false,
                                &device_data, &device_size));
                            desc.context_lens =
                                reinterpret_cast<int64_t *>(device_data);
                            desc.context_lens_size =
                                device_size / sizeof(int64_t);
                        }
                        {
                            std::byte *device_data;
                            size_t device_size;
                            try_(stage_tensor_buffer(node->seq_lens(), map_read,
                                                     false, &device_data,
                                                     &device_size));
                            desc.seq_lens =
                                reinterpret_cast<int64_t *>(device_data);
                            desc.seq_lens_size = device_size / sizeof(int64_t);
                        }

                        // Paged attention specific parameters
                        {
                            std::byte *device_data;
                            size_t device_size;
                            try_(stage_tensor_buffer(
                                node->block_tables(), map_read, false,
                                &device_data, &device_size));
                            desc.block_table =
                                reinterpret_cast<int64_t *>(device_data);
                            desc.block_table_shape[0] =
                                node->block_tables()->shape()[0];
                            desc.block_table_shape[1] =
                                node->block_tables()->shape()[1];
                            desc.block_table_shape[2] =
                                node->block_tables()->shape()[2];
                        }
                        {
                            std::byte *device_data;
                            size_t device_size;
                            try_(stage_tensor_buffer(
                                node->slot_mapping(), map_read, false,
                                &device_data, &device_size));
                            desc.slot_mapping =
                                reinterpret_cast<int64_t *>(device_data);
                            desc.slot_mapping_shape[0] =
                                node->slot_mapping()->shape()[0];
                            desc.slot_mapping_shape[1] =
                                node->slot_mapping()->shape()[1];
                        }

                        {
                            CHECK_WITH_ERR(!node->kv_caches().empty(),
                                           std::errc::invalid_argument);
                            auto &kv_cache = node->kv_caches()[0];
                            if (kv_cache->dtype().equals(datatype_t::int64)) {
                                // Host address tables do not carry the storage
                                // extents needed to recreate device buffers.
                                return err(std::errc::not_supported);
                            } else {
                                // 2. kv_cache is kv cache buffers
                                size_t kv_cache_id = 0;
                                for (auto kv_cache_tensor : node->kv_caches()) {
                                    CHECK_WITH_ERR(
                                        kv_cache_id <
                                            desc.kv_cache_addrs.size(),
                                        std::errc::invalid_argument);
                                    std::byte *device_data;
                                    size_t device_size;
                                    try_(stage_tensor_buffer(
                                        kv_cache_tensor, map_read_write, true,
                                        &device_data, &device_size));
                                    desc.kv_cache_addrs[kv_cache_id++] =
                                        reinterpret_cast<intptr_t>(device_data);
                                }
                            }
                        }
                    }
                }
                cuda_device_allocation desc_buffer;
                try_(stage_array(
                    desc_buffer,
                    std::span<const thread_paged_attention_kv_cache_desc>(
                        descs.data(), descs.size())));
                input_descs[input_id++] = thread_inout_desc{
                    .data = desc_buffer.data(),
                    .size = sizeof(thread_paged_attention_kv_cache_desc) *
                            refspan.size(),
                    .shape = device_shape,
                    .strides = device_strides,
                    .rank = t->shape().size(),
                };
                paged_kv_cache_desc_buffers.emplace_back(
                    std::move(desc_buffer));
            } else {
                return err(std::errc::not_supported);
            }
        } else {
            std::byte *device_data;
            size_t device_size;
            try_(stage_tensor_buffer(t, map_read_write, true, &device_data,
                                     &device_size));
            input_descs[input_id++] = thread_inout_desc{
                .data = device_data,
                .size = device_size,
                .shape = device_shape,
                .strides = device_strides,
                .rank = t->shape().size(),
            };
        }
    }
    CHECK_WITH_ERR(input_id == input_descs.size(), std::errc::invalid_argument);

    try_var(mapped_output, output_buffer_->map(map_read_write));
    cuda_device_allocation output_data_buffer;
    try_(output_data_buffer.allocate(mapped_output.buffer().size_bytes()));

    for (size_t i = 0; i < output_descs.size(); i++) {
        std::fill(output_shapes_[i].begin(), output_shapes_[i].end(), 0);
        std::fill(output_strides_[i].begin(), output_strides_[i].end(), 0);

        cuda_device_allocation shape_buffer;
        cuda_device_allocation stride_buffer;
        try_(stage_array(shape_buffer,
                         std::span<const size_t>(output_shapes_[i].data(),
                                                 output_shapes_[i].size())));
        try_(stage_array(stride_buffer,
                         std::span<const size_t>(output_strides_[i].data(),
                                                 output_strides_[i].size())));

        output_descs[i] = thread_inout_desc{
            .data = nullptr,
            .size = 0,
            .shape = reinterpret_cast<size_t *>(shape_buffer.data()),
            .strides = reinterpret_cast<size_t *>(stride_buffer.data()),
            .rank = output_shapes_[i].size(),
        };
        output_shape_buffers.emplace_back(std::move(shape_buffer));
        output_stride_buffers.emplace_back(std::move(stride_buffer));
    }

    cuda_device_allocation input_descs_buffer;
    cuda_device_allocation output_descs_buffer;
    try_(stage_array(input_descs_buffer,
                     std::span<const thread_inout_desc>(input_descs.data(),
                                                        input_descs.size())));
    try_(stage_array(output_descs_buffer,
                     std::span<const thread_inout_desc>(output_descs.data(),
                                                        output_descs.size())));

    auto input_descs_device =
        reinterpret_cast<const thread_inout_desc *>(input_descs_buffer.data());
    auto output_descs_device =
        reinterpret_cast<thread_inout_desc *>(output_descs_buffer.data());
    auto output_data = output_data_buffer.data();
    try_(run(input_descs_device, output_descs_device, output_data));

    try_(copy_array_from_device(output_descs_buffer,
                                std::span<thread_inout_desc>(
                                    output_descs.data(), output_descs.size())));
    for (size_t i = 0; i < output_descs.size(); i++) {
        try_(copy_array_from_device(
            output_shape_buffers[i],
            std::span<size_t>(output_shapes_[i].data(),
                              output_shapes_[i].size())));
        try_(copy_array_from_device(
            output_stride_buffers[i],
            std::span<size_t>(output_strides_[i].data(),
                              output_strides_[i].size())));
    }
    try_(output_data_buffer.copy_to(mapped_output.buffer()));
    for (auto &staged_buffer : staged_buffers) {
        if (staged_buffer.copy_back) {
            try_(staged_buffer.device_data.copy_to(
                staged_buffer.host_map.buffer()));
        }
    }

    input_descs_ = std::move(input_descs);
    output_descs_ = std::move(output_descs);

    std::vector<value_t> outputs(return_size());
    for (size_t i = 0; i < outputs.size(); i++) {
        try_set(outputs[i], create_output_tensor(i, parameters, output_data));
    }

    auto output_value = outputs.size() == 1
                            ? outputs[0]
                            : tuple(std::in_place, std::move(outputs));
    return ok(output_value);
}

result<tensor>
cuda_runtime_function::create_output_tensor(size_t output_id,
                                            std::span<value_t> parameters,
                                            std::byte *output_data) noexcept {
    auto &output_desc = output_descs_[output_id];
    buffer_slice buffer;
    size_t offset = 0;
    // 1. Find in inputs
    for (size_t i = 0; i < input_descs_.size(); i++) {
        auto &candidate_desc = input_descs_[i];
        if (contains_address_range(candidate_desc.data, candidate_desc.size,
                                   output_desc.data, output_desc.size)) {
            try_var(t, parameters[i].as<tensor>());
            buffer = t->buffer();
            offset = reinterpret_cast<uintptr_t>(output_desc.data) -
                     reinterpret_cast<uintptr_t>(candidate_desc.data);
            break;
        }
    }

    // 2. Find in output buffer
    if (buffer.buffer().empty()) {
        if (contains_address_range(output_data, output_buffer_->size_bytes(),
                                   output_desc.data, output_desc.size)) {
            buffer = buffer_slice(output_buffer_);
            offset = reinterpret_cast<uintptr_t>(output_desc.data) -
                     reinterpret_cast<uintptr_t>(output_data);
        }
    }

    if (buffer.buffer().empty()) {
        return err(std::errc::invalid_argument);
    }

    // 2. Fix offset & size
    buffer = buffer_slice(buffer.buffer(), buffer.start() + offset,
                          output_desc.size);
    try_var(output_type, return_type(output_id));
    try_var(ttype, output_type.as<tensor_type>());
    return ok(tensor(std::in_place, ttype->dtype(), output_shapes_[output_id],
                     output_strides_[output_id], buffer));
}
