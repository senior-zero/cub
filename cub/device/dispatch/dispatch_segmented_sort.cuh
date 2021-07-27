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
#include "../../agent/agent_segmented_radix_sort.cuh"

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <type_traits>

CUB_NAMESPACE_BEGIN

template <bool IS_DESCENDING,
          typename SegmentedPolicyT,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT>
__launch_bounds__ (SegmentedPolicyT::BLOCK_THREADS)
__global__ void DeviceSegmentedRadixSortFallbackKernel(
  const KeyT                      *d_keys_in_origin,              ///< [in] Input keys buffer
  KeyT                            *d_keys_out_orig,               ///< [out] Output keys buffer
  cub::DeviceDoubleBuffer<KeyT>    d_keys_remaining_passes,       ///< [in,out] Double keys buffer
  const ValueT                    *d_values_in_origin,            ///< [in] Input values buffer
  ValueT                          *d_values_out_origin,           ///< [out] Output values buffer
  cub::DeviceDoubleBuffer<ValueT>  d_values_remaining_passes,     ///< [in,out] Double values buffer
  BeginOffsetIteratorT             d_begin_offsets,               ///< [in] Random-access input iterator to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
  EndOffsetIteratorT               d_end_offsets)                 ///< [in] Random-access input iterator to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
{
  const unsigned int segment_id = blockIdx.x;
  OffsetT segment_begin         = d_begin_offsets[segment_id];
  OffsetT segment_end           = d_end_offsets[segment_id];
  OffsetT num_items             = segment_end - segment_begin;

  if (num_items <= 0)
  {
    return;
  }

  using AgentSegmentedRadixSortT =
    cub::AgentSegmentedRadixSort<SegmentedPolicyT,
                                 IS_DESCENDING,
                                 KeyT,
                                 ValueT,
                                 OffsetT>;

  using WarpReduceT = cub::WarpReduce<KeyT>;

  constexpr int items_per_medium_segment = 9;
  constexpr int threads_per_medium_segment = 32;
  constexpr int sub_warp_sort_threshold = items_per_medium_segment *
                                          threads_per_medium_segment;

  __shared__ union
  {
    typename AgentSegmentedRadixSortT::TempStorage block_sort;
    typename WarpReduceT::TempStorage warp_reduce;

    KeyT medium_tile_keys_cache[sub_warp_sort_threshold + 1];
    ValueT medium_tile_values_cache[sub_warp_sort_threshold + 1];
  } temp_storage;

  AgentSegmentedRadixSortT agent(segment_begin,
                                 segment_end,
                                 num_items,
                                 temp_storage.block_sort);

  constexpr int begin_bit = 0;
  constexpr int end_bit = sizeof(KeyT) * 8;

  constexpr int small_tile_size = SegmentedPolicyT::BLOCK_THREADS *
                                  SegmentedPolicyT::ITEMS_PER_THREAD;

  // TODO
  /*
  constexpr int single_thread_sort_threshold = 4;
  if (num_items <= sub_warp_sort_threshold)
  {
    if (threadIdx.x < threads_per_medium_segment)
    {
      sub_warp_merge_sort<items_per_medium_segment,
        threads_per_medium_segment,
        KeyT,
        ValueT>(
        d_keys_in_origin + segment_begin,
        d_keys_out_orig + segment_begin,
        d_values_in_origin + segment_begin,
        d_values_out_origin + segment_begin,
        num_items,
        temp_storage.medium_tile_keys_cache,
        temp_storage.medium_tile_values_cache);
    }
  }
  else
   */

  if (num_items < small_tile_size)
  {
    agent.ProcessSmallSegment(begin_bit,
                              end_bit,
                              d_keys_in_origin,
                              d_values_in_origin,
                              d_keys_out_orig,
                              d_values_out_origin);
  }
  else
  {
    int current_bit = begin_bit;
    int pass_bits = CUB_MIN(SegmentedPolicyT::RADIX_BITS, (end_bit - current_bit));

    agent.ProcessLargeSegment(current_bit,
                              pass_bits,
                              d_keys_in_origin,
                              d_values_in_origin,
                              d_keys_remaining_passes.Current(),
                              d_values_remaining_passes.Current());
    current_bit += pass_bits;

    #pragma unroll 1
    while (current_bit < end_bit)
    {
      pass_bits = CUB_MIN(SegmentedPolicyT::RADIX_BITS, (end_bit - current_bit));

      __syncthreads();
      agent.ProcessLargeSegment(current_bit,
                                pass_bits,
                                d_keys_remaining_passes.Current(),
                                d_values_remaining_passes.Current(),
                                d_keys_remaining_passes.Alternate(),
                                d_values_remaining_passes.Alternate());

      d_keys_remaining_passes.Swap();
      d_values_remaining_passes.Swap();
      current_bit += pass_bits;
    }
  }
}


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

template <bool IS_DESCENDING,
          typename KeyT,
          typename ValueT,
          typename OffsetT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename SelectedPolicy = DeviceSegmentedSortPolicy<KeyT, ValueT>>
struct DispatchSegmentedSort : SelectedPolicy
{
  static constexpr int KEYS_ONLY = Equals<ValueT, NullType>::VALUE;

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
    using LargeSegmentPolicyT = typename ActivePolicyT::LargeSegmentPolicy;

    cudaError error = cudaSuccess;

    do
    {
      // TODO Add DoubleBuffer option - is_override_enabled

      std::size_t tmp_keys_storage_bytes = num_items * sizeof(KeyT);
      std::size_t tmp_values_storage_bytes = KEYS_ONLY
                                           ? std::size_t{}
                                           : num_items * sizeof(KeyT);

      void *allocations[2]            = {nullptr, nullptr};
      std::size_t allocation_sizes[2] = {tmp_keys_storage_bytes,
                                         tmp_values_storage_bytes};

      if (CubDebug(error = AliasTemporaries(d_temp_storage,
                                            temp_storage_bytes,
                                            allocations,
                                            allocation_sizes)))
      {
        break;
      }

      if (d_temp_storage == nullptr)
      {
        if (temp_storage_bytes == 0)
        {
          temp_storage_bytes = 1;
        }

        // Return if the caller is simply requesting the size of the storage allocation
        break;
      }

      KeyT *d_keys_allocation = reinterpret_cast<KeyT*>(allocations[0]);
      ValueT *d_values_allocation = reinterpret_cast<ValueT*>(allocations[1]);

      const int radix_bits          = LargeSegmentPolicyT::RADIX_BITS;
      const int num_bits            = sizeof(KeyT) * 8;
      const int num_passes          = (num_bits + radix_bits - 1) / radix_bits;
      const bool is_num_passes_odd  = num_passes & 1;

      cub::DeviceDoubleBuffer<KeyT> d_keys_remaining_passes(
        is_num_passes_odd ? d_keys_out: d_keys_allocation,
        is_num_passes_odd ? d_keys_allocation : d_keys_out);

      cub::DeviceDoubleBuffer<ValueT> d_values_remaining_passes(
        is_num_passes_odd ? d_values_out: d_values_allocation,
        is_num_passes_odd ? d_values_allocation : d_values_out);

      const unsigned int blocks_in_grid = num_segments;
      const unsigned int threads_in_block = LargeSegmentPolicyT::BLOCK_THREADS;

      // Log single_tile_kernel configuration
      if (debug_synchronous)
      {
        _CubLog("Invoking DeviceSegmentedRadixSortFallbackKernel<<<%d, %d, 0, "
                "%lld>>>(), %d items per thread, bit_grain %d\n",
                blocks_in_grid,
                threads_in_block,
                (long long)stream,
                LargeSegmentPolicyT::ITEMS_PER_THREAD,
                LargeSegmentPolicyT::RADIX_BITS);
      }

      // Invoke fallback kernel
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(blocks_in_grid,
                                                              threads_in_block,
                                                              0,
                                                              stream)
        .doit(DeviceSegmentedRadixSortFallbackKernel<IS_DESCENDING,
                                                     LargeSegmentPolicyT,
                                                     KeyT,
                                                     ValueT,
                                                     BeginOffsetIteratorT,
                                                     EndOffsetIteratorT,
                                                     OffsetT>,
              d_keys_in,
              d_keys_out,
              d_keys_remaining_passes,
              d_values_in,
              d_values_out,
              d_values_remaining_passes,
              d_begin_offsets,
              d_end_offsets);

      // Check for failure to launch
      if (CubDebug(error = cudaPeekAtLastError()))
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      if (debug_synchronous)
      {
        if (CubDebug(error = SyncStream(stream)))
        {
          break;
        }
      }

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
      if (num_items == 0)
      {
        break;
      }

      if (num_segments == 0)
      {
        break;
      }

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
