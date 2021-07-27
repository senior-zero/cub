/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "../../util_math.cuh"
#include "../../util_device.cuh"
#include "../../util_namespace.cuh"
#include "../../block/block_scan.cuh"
#include "../../block/block_load.cuh"
#include "../../block/block_radix_rank.cuh"
#include "../../agent/agent_radix_sort_downsweep.cuh"

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <type_traits>

CUB_NAMESPACE_BEGIN

template <typename KeyT,
          typename ValueT>
struct DeviceSegmentedSortPolicy
{
  using DominantT = typename std::conditional<(sizeof(ValueT) > sizeof(KeyT)), ValueT, KeyT>::type;

  constexpr static int KEYS_ONLY = std::is_same<ValueT, cub::NullType>::value;

  //------------------------------------------------------------------------------
  // Architecture-specific tuning policies
  //------------------------------------------------------------------------------

  struct Policy300 : ChainedPolicy<300, Policy300, Policy300>
  {
    using LargeSegmentPolicy =
      cub::AgentRadixSortDownsweepPolicy<256,
                                         23,
                                         DominantT,
                                         cub::BLOCK_LOAD_TRANSPOSE,
                                         cub::LOAD_DEFAULT,
                                         cub::RADIX_RANK_MEMOIZE,
                                         cub::BLOCK_SCAN_WARP_SCANS,
                                         6>;

    constexpr static int ITEMS_PER_THREAD = KEYS_ONLY ? 9 : 7; // TODO 4B ITEMS
    constexpr static int THREADS_PER_MEDIUM_SEGMENT = 32;
    constexpr static int MEDIUM_SEGMENT_MAX_SIZE = ITEMS_PER_THREAD *
                                                   THREADS_PER_MEDIUM_SEGMENT;

    constexpr static int THREADS_PER_SMALL_SEGMENT = 4;
    constexpr static int SMALL_SEGMENT_MAX_SIZE = ITEMS_PER_THREAD *
                                                  THREADS_PER_SMALL_SEGMENT;
  };

  /// MaxPolicy
  using MaxPolicy = Policy300;
};

template <typename KeyT,
          typename ValueT,
          typename OffsetT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename SelectedPolicy = DeviceSegmentedSortPolicy<KeyT, ValueT>>
struct DispatchSegmentedSort : SelectedPolicy
{
  void *d_temp_storage;
  std::size_t &temp_storage_bytes;
  const KeyT *d_keys_in;
  KeyT *d_keys_out;
  const ValueT *d_values_in;
  ValueT *d_values_out;
  OffsetT num_items;
  OffsetT num_segments;
  BeginOffsetIteratorT d_begin_offsets;
  EndOffsetIteratorT d_end_offsets;
  cudaStream_t stream;
  bool debug_synchronous;

  /// Constructor
  CUB_RUNTIME_FUNCTION __forceinline__
  DispatchSegmentedSort(void *d_temp_storage,
                        std::size_t &temp_storage_bytes,
                        const KeyT *d_keys_in,
                        KeyT *d_keys_out,
                        const ValueT *d_values_in,
                        ValueT *d_values_out,
                        OffsetT num_items,
                        OffsetT num_segments,
                        BeginOffsetIteratorT d_begin_offsets,
                        EndOffsetIteratorT d_end_offsets,
                        cudaStream_t stream,
                        bool debug_synchronous)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_keys_in(d_keys_in)
      , d_keys_out(d_keys_out)
      , d_values_in(d_values_in)
      , d_values_out(d_values_out)
      , num_items(num_items)
      , num_segments(num_segments)
      , d_begin_offsets(d_begin_offsets)
      , d_end_offsets(d_end_offsets)
      , stream(stream)
      , debug_synchronous(debug_synchronous)
  {}

  /// Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t Invoke()
  {
    cudaError error = cudaSuccess;

    do
    {
    } while (false);

    return error;
  }

  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
           std::size_t &temp_storage_bytes,
           const KeyT *d_keys_in,
           KeyT *d_keys_out,
           const ValueT *d_values_in,
           ValueT *d_values_out,
           OffsetT num_items,
           OffsetT num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           cudaStream_t stream,
           bool debug_synchronous)
  {
    using MaxPolicyT = typename DispatchSegmentedSort::MaxPolicy;

    cudaError error = cudaSuccess;

    do
    {
      // Get PTX version
      int ptx_version = 0;
      if (CubDebug(error = PtxVersion(ptx_version)))
      {
        break;
      }

      // Create dispatch functor
      DispatchSegmentedSort dispatch(d_temp_storage,
                                     temp_storage_bytes,
                                     d_keys_in,
                                     d_keys_out,
                                     d_values_in,
                                     d_values_out,
                                     num_items,
                                     num_segments,
                                     d_begin_offsets,
                                     d_end_offsets,
                                     stream,
                                     debug_synchronous);

      // Dispatch to chained policy
      if (CubDebug(error = MaxPolicyT::Invoke(ptx_version, dispatch)))
      {
        break;
      }
    } while (false);

    return error;
  }
};

CUB_NAMESPACE_END
