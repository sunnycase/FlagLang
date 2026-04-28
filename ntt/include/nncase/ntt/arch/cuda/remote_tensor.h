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
#pragma once
#include "../../distributed/remote_tensor.h"
#include "../../tensor.h"
#include "../../vector.h"

namespace nncase::ntt::distributed {
namespace detail {
inline constexpr size_t topology_element_count = (size_t)nncase::ntt::dim_value(
    program_dim<topology::chip>() * program_dim<topology::block>() *
    program_dim<topology::warp>() * program_dim<topology::thread>());

extern __device__ uintptr_t global_local_data_ptr[topology_element_count * 2];
extern __device__ uintptr_t
    global_thread_local_rdata_ptr[topology_element_count * 2];
extern __device__ uintptr_t
    global_thread_local_cache_ptr[topology_element_count * 3];
extern __device__ uintptr_t
    global_block_local_rdata_ptr[topology_element_count * 2];

template <class TProgramIds>
__device__ size_t program_linear_offset(const TProgramIds &program_ids) {
    return (size_t)((((program_ids.template at<0>() *
                       program_dim<topology::block>()) +
                      program_ids.template at<1>()) *
                         program_dim<topology::warp>() +
                     program_ids.template at<2>()) *
                        program_dim<topology::thread>() +
                    program_ids.template at<3>());
}

template <class TProgramIds>
__device__ uintptr_t *global_local_data_desc(const TProgramIds &program_ids) {
    return global_local_data_ptr + program_linear_offset(program_ids) * 2;
}

template <class TProgramIds>
__device__ uintptr_t *
global_thread_local_rdata_desc(const TProgramIds &program_ids) {
    return global_thread_local_rdata_ptr +
           program_linear_offset(program_ids) * 2;
}

template <class TProgramIds>
__device__ uintptr_t *
global_thread_local_cache_desc(const TProgramIds &program_ids) {
    return global_thread_local_cache_ptr +
           program_linear_offset(program_ids) * 3;
}

template <class TProgramIds>
__device__ uintptr_t *
global_block_local_rdata_desc(const TProgramIds &program_ids) {
    return global_block_local_rdata_ptr +
           program_linear_offset(program_ids) * 2;
}

template <class T, topology RemoteScope, topology TensorScope,
          ScopedProgramIds<TensorScope> TLocalProgramIds,
          ScopedProgramIds<TensorScope> TRemoteProgramIds>
__device__ auto get_remote_address(const TLocalProgramIds &local_program_ids,
                                   const TRemoteProgramIds &remote_program_ids,
                                   T *local_address) {
    auto local_data = global_local_data_desc(local_program_ids);
    auto remote_data = global_local_data_desc(remote_program_ids);
    auto start = (size_t)local_data[0];
    auto end = (size_t)local_data[1];
    auto remote_address = (size_t)remote_data[0];
    if ((uintptr_t)local_address < start || (uintptr_t)local_address >= end) {
        auto local_rdata = global_thread_local_rdata_desc(local_program_ids);
        auto remote_rdata = global_thread_local_rdata_desc(remote_program_ids);
        start = (size_t)local_rdata[0];
        end = (size_t)local_rdata[1];
        remote_address = (size_t)remote_rdata[0];
        if ((uintptr_t)local_address < start ||
            (uintptr_t)local_address >= end) {
            auto local_block_rdata =
                global_block_local_rdata_desc(local_program_ids);
            auto remote_block_rdata =
                global_block_local_rdata_desc(remote_program_ids);
            start = (size_t)local_block_rdata[0];
            remote_address = (size_t)remote_block_rdata[0];
        }
    }

    return local_address - (T *)start + (T *)remote_address;
}
} // namespace detail

template <topology RemoteScope, topology TensorScope>
struct remote_tensor_constructor {
    template <class T, Shape TShape, Strides TStrides,
              ScopedProgramIds<TensorScope> TLocalProgramIds,
              ScopedProgramIds<TensorScope> TRemoteProgramIds>
    __device__ auto operator()(T *data, const TShape &shape,
                               const TStrides &strides,
                               const TLocalProgramIds &local_program_ids,
                               const TRemoteProgramIds &remote_program_ids) {
        auto remote_address =
            detail::get_remote_address<T, RemoteScope, TensorScope>(
                local_program_ids, remote_program_ids, data);
        return make_tensor_view_from_address(remote_address, shape, strides);
    }
};
} // namespace nncase::ntt::distributed
