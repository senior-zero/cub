/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * cub::DeviceBatchMemcpy provides device-wide, parallel operations for copying data from a number of given source
 * buffers to their corresponding destination buffer.
 */

#pragma once

#include "../config.cuh"
#include "dispatch/dispatch_batch_memcpy.cuh"

#include <cstdint>
#include <type_traits>

CUB_NAMESPACE_BEGIN

/**
 * @brief Copies data from a batch of given source buffers to their corresponding destination buffer.
 * @note If any input buffer aliases memory from any output buffer the behavior is undefined. If any output buffer
 * aliases memory of another output buffer the behavior is undefined. Input buffers can alias one another.
 *
 * @tparam InputBufferIt <b>[inferred]</b> Device-accessible random-access input iterator type providing the pointers to
 * the source memory buffers
 * @tparam OutputBufferIt <b>[inferred]</b> Device-accessible random-access input iterator type providing the pointers
 * to the destination memory buffers
 * @tparam BufferSizeIteratorT <b>[inferred]</b> Device-accessible random-access input iterator type providing the
 * number of bytes to be copied for each pair of buffers
 * @param d_temp_storage [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation
 * size is written to \p temp_storage_bytes and no work is done.
 * @param temp_storage_bytes [in,out] Reference to size in bytes of \p d_temp_storage allocation
 * @param input_buffer_it [in] Iterator providing the pointers to the source memory buffers
 * @param output_buffer_it [in] Iterator providing the pointers to the destination memory buffers
 * @param buffer_sizes [in] Iterator providing the number of bytes to be copied for each pair of buffers
 * @param num_buffers [in] The total number of buffer pairs
 * @param stream [in] <b>[optional]</b> CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
 * @param debug_synchronous [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to
 * check for errors. May cause significant slowdown.  Default is \p false.
 */
template <typename InputBufferIt, typename OutputBufferIt, typename BufferSizeIteratorT>
cudaError_t DeviceBatchMemcpy(void *d_temp_storage,
                              size_t &temp_storage_bytes,
                              InputBufferIt input_buffer_it,
                              OutputBufferIt output_buffer_it,
                              BufferSizeIteratorT buffer_sizes,
                              uint32_t num_buffers,
                              cudaStream_t stream    = 0,
                              bool debug_synchronous = false)
{
  // Integer type large enough to hold any offset in [0, num_buffers)
  using BufferOffsetT = uint32_t;

  // Integer type large enough to hold any offset in [0, num_thread_blocks_launched), where a safe uppper bound on
  // num_thread_blocks_launched can be assumed to be given by IDIV_CEIL(num_buffers, 64)
  using BlockOffsetT = uint32_t;

  return DispatchBatchMemcpy<InputBufferIt, OutputBufferIt, BufferSizeIteratorT, BufferOffsetT, BlockOffsetT>::Dispatch(
    d_temp_storage,
    temp_storage_bytes,
    input_buffer_it,
    output_buffer_it,
    buffer_sizes,
    num_buffers,
    stream,
    debug_synchronous);
}

CUB_NAMESPACE_END
