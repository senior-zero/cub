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
#include "../../device/device_partition.cuh"
#include "../../agent/agent_segmented_sort.cuh"
#include "../../agent/agent_segmented_radix_sort.cuh"
#include "../../block/block_merge_sort.cuh"
#include "../../thread/thread_sort.cuh"

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>
#include <thrust/iterator/counting_iterator.h>

#include <type_traits>

CUB_NAMESPACE_BEGIN


// TODO There should be function like that in CUB already, look it up
template <int threads_per_segment>
__device__ __forceinline__ unsigned int gen_mask()
{
  const unsigned int group_id = LaneId() / threads_per_segment;
  const unsigned int group_mask = (1 << threads_per_segment) - 1;
  const unsigned int final_mask = group_mask << (group_id * threads_per_segment);

  return final_mask;
}

template <>
__device__ __forceinline__ unsigned int gen_mask<32>()
{
  return 0xFFFFFFFF;
}

template <int items_per_thread,
  int threads_per_segment,
  typename T>
__device__ void sub_warp_load(int lane_id,
                              int segment_size,
                              int group_mask,
                              const T *input,
                              T (&keys)[items_per_thread],
                              T *cache)
{
  if (items_per_thread > 10)
  {
    // COALESCED_LOAD

#pragma unroll
    for (int item = 0; item < items_per_thread; item++)
    {
      const int idx = threads_per_segment * item + lane_id;

      if (idx < segment_size)
      {
        keys[item] = input[idx];
      }
    }

// store in shared
    for (int item = 0; item < items_per_thread; item++)
    {
      const int idx = threads_per_segment * item + lane_id;
      cache[idx]    = keys[item];
    }
    __syncwarp(group_mask);

// load blocked
    for (int item = 0; item < items_per_thread; item++)
    {
      const int idx = items_per_thread * lane_id + item;
      keys[item]    = cache[idx];
    }
    __syncwarp(group_mask);
  }
  else
  {
    // Naive load

    for (int item = 0; item < items_per_thread; item++)
    {
      const unsigned int idx = lane_id * items_per_thread + item;

      if (idx < segment_size)
      {
        keys[item] = input[idx];
      }
    }
  }
}

template <int items_per_thread,
  int threads_per_segment,
  typename T>
__device__ void sub_warp_store(int lane_id,
                               int segment_size,
                               int group_mask,
                               T *output,
                               T (&keys)[items_per_thread],
                               T *cache)
{
  if (items_per_thread > 10)
  {
    // Coalesced store
    __syncwarp(group_mask);

    // load blocked
    for (int item = 0; item < items_per_thread; item++)
    {
      const int idx = items_per_thread * lane_id + item;
      cache[idx] = keys[item];
    }
    __syncwarp(group_mask);

    // store in shared
    for (int item = 0; item < items_per_thread; item++)
    {
      const int idx = threads_per_segment * item + lane_id;
      keys[item] = cache[idx];
    }

    for (int item = 0; item < items_per_thread; item++)
    {
      const int idx = threads_per_segment * item + lane_id;

      if (idx < segment_size)
      {
        output[idx] = keys[item];
      }
    }
  }
  else
  {
    // Naive store
    for (int item = 0; item < items_per_thread; item++)
    {
      const int idx = lane_id * items_per_thread + item;

      if (idx < segment_size)
      {
        output[idx] = keys[item];
      }
    }
  }
}

template <bool IS_DESCENDING,
          int items_per_thread,
          int threads_per_segment,
          typename KeyT,
          typename ValueT>
__device__ void
sub_warp_merge_sort(const KeyT *keys_input,
                    KeyT *keys_output,
                    const ValueT *values_input,
                    ValueT *values_output,
                    int segment_size,
                    KeyT *keys_cache,
                    ValueT *values_cache) // T *cache = block_cache + (tid /
                                          // threads_per_segment) *
                                          // threads_per_segment *
                                          // items_per_thread;
{
  static constexpr bool KEYS_ONLY = cub::Equals<ValueT, cub::NullType>::VALUE;

  auto binary_op = [] (KeyT lhs, KeyT rhs) -> bool
  {
    if (IS_DESCENDING)
    {
      return lhs > rhs;
    }
    else
    {
      return lhs < rhs;
    }
  };

  int tid = static_cast<int>(threadIdx.x);

  if (segment_size == 0)
  {
    return;
  }

  int lane_id = tid % threads_per_segment;

  if (segment_size == 1)
  {
    if (lane_id == 0)
    {
      keys_output[0] = keys_input[0];

      if (!KEYS_ONLY)
      {
        values_output[0] = values_input[1];
      }
    }

    return;
  }
  else if (segment_size == 2)
  {
    if (lane_id == 0)
    {
      KeyT lhs = keys_input[0];
      KeyT rhs = keys_input[1];

      if (lhs < rhs)
      {
        keys_output[0] = lhs;
        keys_output[1] = rhs;

        if (!KEYS_ONLY)
        {
          values_output[0] = values_input[0];
          values_output[1] = values_input[1];
        }
      }
      else
      {
        keys_output[0] = rhs;
        keys_output[1] = lhs;

        if (!KEYS_ONLY)
        {
          values_output[0] = values_input[1];
          values_output[1] = values_input[0];
        }
      }
    }

    return;
  }

  KeyT keys[items_per_thread];
  ValueT values[items_per_thread];

  int indices[items_per_thread];

  int group_mask = static_cast<int>(gen_mask<threads_per_segment>());

  sub_warp_load<items_per_thread, threads_per_segment>(lane_id,
                                                       segment_size,
                                                       group_mask,
                                                       keys_input,
                                                       keys,
                                                       keys_cache);

  if (!KEYS_ONLY)
  {
    sub_warp_load<items_per_thread, threads_per_segment>(lane_id,
                                                         segment_size,
                                                         group_mask,
                                                         values_input,
                                                         values,
                                                         values_cache);
  }

  // 2) Sort

  KeyT max_key = keys_input[0];

#pragma unroll
  for (int item = 1; item < items_per_thread; ++item)
  {
    if (items_per_thread * lane_id + item < segment_size)
    {
      max_key = max_key < keys[item] ? keys[item] : max_key;
    }
    else
    {
      keys[item] = max_key;
    }
  }

  if (lane_id * items_per_thread < segment_size)
  {
    StableOddEvenSort(keys, values, binary_op);
  }

  if (segment_size > items_per_thread)
  {
#pragma unroll
    for (int target_merged_threads_number = 2;
         target_merged_threads_number <= threads_per_segment;
         target_merged_threads_number *= 2)
    {
      int merged_threads_number = target_merged_threads_number / 2;
      int mask                  = target_merged_threads_number - 1;

      WARP_SYNC(group_mask);

      for (int item = 0; item < items_per_thread; item++)
      {
        keys_cache[lane_id * items_per_thread + item] = keys[item];
      }
      WARP_SYNC(group_mask);

      int first_thread_idx_in_thread_group_being_merged = ~mask & lane_id;
      int start                                         = items_per_thread *
                  first_thread_idx_in_thread_group_being_merged;
      int size = items_per_thread * merged_threads_number;

      int thread_idx_in_thread_group_being_merged = mask & lane_id;

      int diag =
        min(segment_size,
            items_per_thread * thread_idx_in_thread_group_being_merged);

      int keys1_beg = min(segment_size, start);
      int keys1_end = min(segment_size, keys1_beg + size);
      int keys2_beg = keys1_end;
      int keys2_end = min(segment_size, keys2_beg + size);

      int keys1_count = keys1_end - keys1_beg;
      int keys2_count = keys2_end - keys2_beg;

      int partition_diag = MergePath<KeyT>(&keys_cache[keys1_beg],
                                            &keys_cache[keys2_beg],
                                            keys1_count,
                                            keys2_count,
                                            diag,
                                            binary_op);

      int keys1_beg_loc   = keys1_beg + partition_diag;
      int keys1_end_loc   = keys1_end;
      int keys2_beg_loc   = keys2_beg + diag - partition_diag;
      int keys2_end_loc   = keys2_end;
      int keys1_count_loc = keys1_end_loc - keys1_beg_loc;
      int keys2_count_loc = keys2_end_loc - keys2_beg_loc;

      SerialMerge(&keys_cache[0],
                  keys1_beg_loc,
                  keys2_beg_loc,
                  keys1_count_loc,
                  keys2_count_loc,
                  keys,
                  indices,
                  binary_op);

      if (!KEYS_ONLY)
      {
        WARP_SYNC(group_mask);

#pragma unroll
        for (int item = 0; item < items_per_thread; item++)
        {
          int idx           = items_per_thread * lane_id + item;
          values_cache[idx] = values[item];
        }
        WARP_SYNC(group_mask);

        // gather items from shmem
        //
#pragma unroll
        for (int item = 0; item < items_per_thread; item++)
        {
          values[item] = values_cache[indices[item]];
        }
      }
    }
  }

  sub_warp_store<items_per_thread, threads_per_segment>(lane_id,
                                                        segment_size,
                                                        group_mask,
                                                        keys_output,
                                                        keys,
                                                        keys_cache);

  if (!KEYS_ONLY)
  {
    sub_warp_store<items_per_thread, threads_per_segment>(lane_id,
                                                          segment_size,
                                                          group_mask,
                                                          values_output,
                                                          values,
                                                          values_cache);
  }
}

template <bool IS_DESCENDING,
          typename SegmentedPolicyT,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT>
__launch_bounds__ (SegmentedPolicyT::BLOCK_THREADS)
__global__ void DeviceSegmentedSortFallbackKernel(
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
    cub::AgentSegmentedRadixSort<IS_DESCENDING,
                                 SegmentedPolicyT,
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

  if (num_items <= sub_warp_sort_threshold)
  {
    if (threadIdx.x < threads_per_medium_segment)
    {
      sub_warp_merge_sort<IS_DESCENDING,
                          items_per_medium_segment,
                          threads_per_medium_segment,
                          KeyT,
                          ValueT>(d_keys_in_origin + segment_begin,
                                  d_keys_out_orig + segment_begin,
                                  d_values_in_origin + segment_begin,
                                  d_values_out_origin + segment_begin,
                                  num_items,
                                  temp_storage.medium_tile_keys_cache,
                                  temp_storage.medium_tile_values_cache);
    }
  }
  else if (num_items < small_tile_size)
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

      CTA_SYNC();
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


template <
  bool                    IS_DESCENDING,
  typename                SegmentedPolicyT,               ///< Active tuning policy
  typename                KeyT,                           ///< Key type
  typename                ValueT,                         ///< Value type
  typename                BeginOffsetIteratorT,           ///< Random-access input iterator type for reading segment beginning offsets \iterator
  typename                EndOffsetIteratorT,             ///< Random-access input iterator type for reading segment ending offsets \iterator
  typename                OffsetT>                        ///< Signed integer type for global offsets
__launch_bounds__ (SegmentedPolicyT::BLOCK_THREADS)
__global__ void DeviceSegmentedSortKernelWithReorderingSmall(
  OffsetT                          small_segments,
  OffsetT                          medium_segments,
  OffsetT                          medium_blocks,
  const OffsetT                   *d_small_segments_reordering,
  const OffsetT                   *d_medium_segments_reordering,
  const KeyT                      *d_keys_in_origin,              ///< [in] Input keys buffer
  KeyT                            *d_keys_out_orig,               ///< [out] Output keys buffer
  const ValueT                    *d_values_in_origin,            ///< [in] Input values buffer
  ValueT                          *d_values_out_origin,           ///< [out] Output values buffer
  BeginOffsetIteratorT             d_begin_offsets,               ///< [in] Random-access input iterator to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
  EndOffsetIteratorT               d_end_offsets)                 ///< [in] Random-access input iterator to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
{
  const unsigned int tid = threadIdx.x;
  const unsigned int bid = blockIdx.x;

  constexpr int items_per_thread = SegmentedPolicyT::ITEMS_PER_THREAD;
  constexpr int threads_per_medium_segment = SegmentedPolicyT::THREADS_PER_MEDIUM_SEGMENT;
  constexpr int threads_per_small_segment = SegmentedPolicyT::THREADS_PER_SMALL_SEGMENT;
  constexpr int medium_segment_max_size = SegmentedPolicyT::MEDIUM_SEGMENT_MAX_SIZE;
  constexpr int small_segment_max_size = SegmentedPolicyT::SMALL_SEGMENT_MAX_SIZE;

  __shared__ union
  {
    KeyT block_keys_cache[items_per_thread * SegmentedPolicyT::BLOCK_THREADS + 1];
    ValueT block_values_cache[items_per_thread * SegmentedPolicyT::BLOCK_THREADS + 1];
  } temp_storage;

  if (bid < medium_blocks)
  {
    constexpr OffsetT segments_per_medium_block =
      static_cast<OffsetT>(SegmentedPolicyT::SEGMENTS_PER_MEDIUM_BLOCK);

    const OffsetT sid_within_block = tid / threads_per_medium_segment;
    const OffsetT reordered_segment_id = bid * segments_per_medium_block + sid_within_block;

    if (reordered_segment_id < medium_segments)
    {
      const OffsetT segment_id =
        d_medium_segments_reordering[reordered_segment_id];

      OffsetT segment_begin = d_begin_offsets[segment_id];
      OffsetT segment_end   = d_end_offsets[segment_id];
      OffsetT num_items     = segment_end - segment_begin;

      const int group_offset = sid_within_block * medium_segment_max_size;

      sub_warp_merge_sort<IS_DESCENDING, items_per_thread, threads_per_medium_segment, KeyT, ValueT>(
        d_keys_in_origin + segment_begin,
        d_keys_out_orig + segment_begin,
        d_values_in_origin + segment_begin,
        d_values_out_origin + segment_begin,
        num_items,
        temp_storage.block_keys_cache + group_offset,
        temp_storage.block_values_cache + group_offset);
    }
  }
  else
  {
    constexpr OffsetT segments_per_small_block =
      static_cast<OffsetT>(SegmentedPolicyT::SEGMENTS_PER_SMALL_BLOCK);

    const OffsetT sid_within_block = tid / threads_per_small_segment;
    const OffsetT reordered_segment_id = (bid - medium_blocks) * segments_per_small_block + sid_within_block;

    if (reordered_segment_id < small_segments)
    {
      const OffsetT segment_id =
        d_small_segments_reordering[reordered_segment_id];

      OffsetT segment_begin = d_begin_offsets[segment_id];
      OffsetT segment_end   = d_end_offsets[segment_id];
      OffsetT num_items     = segment_end - segment_begin;

      const int group_offset = sid_within_block * small_segment_max_size;

      sub_warp_merge_sort<IS_DESCENDING, items_per_thread, threads_per_small_segment, KeyT, ValueT>(
        d_keys_in_origin + segment_begin,
        d_keys_out_orig + segment_begin,
        d_values_in_origin + segment_begin,
        d_values_out_origin + segment_begin,
        num_items,
        temp_storage.block_keys_cache + group_offset,
        temp_storage.block_values_cache + group_offset);
    }
  }
}

template <
  bool                    IS_DESCENDING,
  typename                SegmentedPolicyT,               ///< Active tuning policy
  typename                KeyT,                           ///< Key type
  typename                ValueT,                         ///< Value type
  typename                BeginOffsetIteratorT,           ///< Random-access input iterator type for reading segment beginning offsets \iterator
  typename                EndOffsetIteratorT,             ///< Random-access input iterator type for reading segment ending offsets \iterator
  typename                OffsetT>                        ///< Signed integer type for global offsets
__launch_bounds__ (SegmentedPolicyT::BLOCK_THREADS)
__global__ void DeviceSegmentedSortKernelWithReorderingLarge(
  const OffsetT                   *d_segments_reordering,
  const KeyT                      *d_keys_in_origin,              ///< [in] Input keys buffer
  KeyT                            *d_keys_out_orig,               ///< [out] Output keys buffer
  cub::DeviceDoubleBuffer<KeyT>    d_keys_remaining_passes,       ///< [in,out] Double keys buffer
  const ValueT                    *d_values_in_origin,            ///< [in] Input values buffer
  ValueT                          *d_values_out_origin,           ///< [out] Output values buffer
  cub::DeviceDoubleBuffer<ValueT>  d_values_remaining_passes,     ///< [in,out] Double values buffer
  BeginOffsetIteratorT             d_begin_offsets,               ///< [in] Random-access input iterator to the sequence of beginning offsets of length \p num_segments, such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
  EndOffsetIteratorT               d_end_offsets)                 ///< [in] Random-access input iterator to the sequence of ending offsets of length \p num_segments, such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup> data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.  If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>, the <em>i</em><sup>th</sup> is considered empty.
{
  constexpr int small_tile_size = SegmentedPolicyT::BLOCK_THREADS *
                                  SegmentedPolicyT::ITEMS_PER_THREAD;

  using AgentSegmentedRadixSortT =
    cub::AgentSegmentedRadixSort<IS_DESCENDING,
                                 SegmentedPolicyT,
                                 KeyT,
                                 ValueT,
                                 OffsetT>;

  __shared__ typename AgentSegmentedRadixSortT::TempStorage block_sort;

  const unsigned int bid = blockIdx.x;

  constexpr int begin_bit = 0;
  constexpr int end_bit   = sizeof(KeyT) * 8;

  const OffsetT segment_id    = d_segments_reordering[bid];
  const OffsetT segment_begin = d_begin_offsets[segment_id];
  const OffsetT segment_end   = d_end_offsets[segment_id];
  const OffsetT num_items     = segment_end - segment_begin;

  AgentSegmentedRadixSortT agent(segment_begin,
                                 segment_end,
                                 num_items,
                                 block_sort);

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
    int pass_bits   = CUB_MIN(SegmentedPolicyT::RADIX_BITS,
                              (end_bit - current_bit));

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
      pass_bits = CUB_MIN(SegmentedPolicyT::RADIX_BITS,
                          (end_bit - current_bit));

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

template <typename T,
  typename BeginOffsetIteratorT,
  typename EndOffsetIteratorT>
struct SegmentSizeGreaterThan
{
  T value {};
  BeginOffsetIteratorT d_offset_begin {};
  EndOffsetIteratorT d_offset_end {};

  explicit SegmentSizeGreaterThan(
    T value,
    BeginOffsetIteratorT d_offset_begin,
    EndOffsetIteratorT d_offset_end)
    : value(value)
    , d_offset_begin(d_offset_begin)
    , d_offset_end(d_offset_end)
  {}

  __device__ bool operator()(unsigned int segment_id) const
  {
    const T segment_size = d_offset_end[segment_id] - d_offset_begin[segment_id];
    return segment_size > value;
  }
};

template <typename T,
  typename BeginOffsetIteratorT,
  typename EndOffsetIteratorT>
struct SegmentSizeLessThan
{
  T value {};
  BeginOffsetIteratorT d_offset_begin {};
  EndOffsetIteratorT d_offset_end {};

  explicit SegmentSizeLessThan(
    T value,
    BeginOffsetIteratorT d_offset_begin,
    EndOffsetIteratorT d_offset_end)
    : value(value)
    , d_offset_begin(d_offset_begin)
    , d_offset_end(d_offset_end)
  {}

  __device__ bool operator()(unsigned int segment_id) const
  {
    const T segment_size = d_offset_end[segment_id] - d_offset_begin[segment_id];
    return segment_size < value;
  }
};


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

    constexpr static int ITEMS_PER_SMALL_AND_MEDIUM_THREAD =
      Nominal4BItemsToItems<DominantT>(KEYS_ONLY ? 9 : 7);

    using SmallAndMediumPolicy = cub::AgentSmallAndMediumSegmentedSortPolicy<
      256,                               // Block size
      ITEMS_PER_SMALL_AND_MEDIUM_THREAD, // Items per thread in small and medium segments
      32,                                // Threads per medium segment
      4>;                                // Threads per small segment
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
    using SmallAndMediumPolicyT = typename ActivePolicyT::SmallAndMediumPolicy;

    cudaError error = cudaSuccess;

    do
    {
      // TODO Add DoubleBuffer option - is_override_enabled
      const bool reorder_segments = num_segments > 500; // TODO Magick number

      std::size_t tmp_keys_storage_bytes = num_items * sizeof(KeyT);
      std::size_t tmp_values_storage_bytes = KEYS_ONLY
                                           ? std::size_t{}
                                           : num_items * sizeof(KeyT);

      OffsetT *d_large_and_medium_segments_reordering = nullptr;
      OffsetT *d_small_segments_reordering = nullptr;
      OffsetT *d_group_sizes = nullptr;

      std::size_t three_way_partition_temp_storage_bytes {};

      SegmentSizeGreaterThan<OffsetT, BeginOffsetIteratorT, EndOffsetIteratorT>
        large_segments_selector(
          SmallAndMediumPolicyT::MEDIUM_SEGMENT_MAX_SIZE,
          d_begin_offsets,
          d_end_offsets);

      SegmentSizeLessThan<OffsetT, BeginOffsetIteratorT, EndOffsetIteratorT>
        small_segments_selector(
          SmallAndMediumPolicyT::SMALL_SEGMENT_MAX_SIZE + 1,
          d_begin_offsets,
          d_end_offsets);

      if (reorder_segments)
      {
        cub::DevicePartition::If(nullptr,
                                 three_way_partition_temp_storage_bytes,
                                 THRUST_NS_QUALIFIER::counting_iterator<OffsetT>(0),
                                 d_large_and_medium_segments_reordering,
                                 d_small_segments_reordering,
                                 d_group_sizes,
                                 num_segments,
                                 large_segments_selector,
                                 small_segments_selector,
                                 stream,
                                 debug_synchronous);
      }

      std::size_t large_and_medium_segments_reordering_bytes =
        reorder_segments ? sizeof(OffsetT) * num_segments : 0;

      std::size_t small_segments_reordering_bytes =
        reorder_segments ? sizeof(OffsetT) * num_segments : 0;

      std::size_t keys_and_partitioning_union =
        std::max(tmp_keys_storage_bytes,
                 three_way_partition_temp_storage_bytes);

      // Partition selects large and small groups.
      // The middle group is not selected.
      constexpr std::size_t num_selected_groups = 2;
      std::size_t group_sizes_bytes = reorder_segments
                                    ? sizeof(OffsetT) * num_selected_groups
                                    : 0ul;

      std::size_t allocation_sizes[5] = {
        keys_and_partitioning_union,
        tmp_values_storage_bytes,
        large_and_medium_segments_reordering_bytes,
        small_segments_reordering_bytes,
        group_sizes_bytes
      };

      void *allocations[5] =
        {nullptr, nullptr, nullptr, nullptr, nullptr};

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

      d_large_and_medium_segments_reordering = reinterpret_cast<OffsetT*>(allocations[2]);
      d_small_segments_reordering = reinterpret_cast<OffsetT*>(allocations[3]);
      d_group_sizes = reinterpret_cast<OffsetT*>(allocations[4]);

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

      if (reorder_segments)
      {
        void *d_partition_temp_storage = allocations[0];

        if (CubDebug(error = cub::DevicePartition::If(
          d_partition_temp_storage,
          three_way_partition_temp_storage_bytes,
          THRUST_NS_QUALIFIER::counting_iterator<OffsetT>(0),
          d_large_and_medium_segments_reordering,
          d_small_segments_reordering,
          d_group_sizes,
          num_segments,
          large_segments_selector,
          small_segments_selector,
          stream,
          debug_synchronous)))
        {
          break;
        }

        OffsetT h_group_sizes[num_selected_groups];
        if (CubDebug(error = cudaMemcpy(h_group_sizes,
                                        d_group_sizes,
                                        group_sizes_bytes,
                                        cudaMemcpyDeviceToHost)))
        {
          break;
        }

        const OffsetT large_segments = h_group_sizes[0];

        if (large_segments > 0)
        {
          const OffsetT blocks_in_grid = large_segments; // One CTA per segment

          // TODO cudaGraph
          if (debug_synchronous)
          {
            _CubLog("Invoking "
                    "DeviceSegmentedSortKernelWithReorderingLarge<<<"
                    "%d, %d, 0, %lld>>>(), %d items per thread\n",
                    static_cast<int>(blocks_in_grid),
                    SmallAndMediumPolicyT::BLOCK_THREADS,
                    (long long)stream,
                    SmallAndMediumPolicyT::ITEMS_PER_THREAD);
          }

          THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
            blocks_in_grid,
            LargeSegmentPolicyT::BLOCK_THREADS,
            0,
            stream)
            .doit(DeviceSegmentedSortKernelWithReorderingLarge<
                    IS_DESCENDING,
                    LargeSegmentPolicyT,
                    KeyT,
                    ValueT,
                    BeginOffsetIteratorT,
                    EndOffsetIteratorT,
                    OffsetT>,
                  d_large_and_medium_segments_reordering,
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
        }

        const OffsetT small_segments = h_group_sizes[1];
        const OffsetT medium_segments = num_segments -
                                        (large_segments + small_segments);

        const OffsetT small_blocks =
          (small_segments +
           ActivePolicyT::SmallAndMediumPolicy::SEGMENTS_PER_SMALL_BLOCK - 1) /
          ActivePolicyT::SmallAndMediumPolicy::SEGMENTS_PER_SMALL_BLOCK;

        const OffsetT medium_blocks =
          (medium_segments +
           ActivePolicyT::SmallAndMediumPolicy::SEGMENTS_PER_MEDIUM_BLOCK - 1) /
          ActivePolicyT::SmallAndMediumPolicy::SEGMENTS_PER_MEDIUM_BLOCK;

        const OffsetT small_and_medium_blocks_in_grid = small_blocks +
                                                        medium_blocks;

        if (small_and_medium_blocks_in_grid)
        {
          if (debug_synchronous)
          {
            _CubLog("Invoking "
                    "DeviceSegmentedSortKernelWithReorderingSmall<<<"
                    "%d, %d, 0, %lld>>>(), %d items per thread\n",
                    static_cast<int>(small_and_medium_blocks_in_grid),
                    SmallAndMediumPolicyT::BLOCK_THREADS,
                    (long long)stream,
                    SmallAndMediumPolicyT::ITEMS_PER_THREAD);
          }

          // TODO cudaGraph
          THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
            small_and_medium_blocks_in_grid,
            SmallAndMediumPolicyT::BLOCK_THREADS,
            0,
            stream)
            .doit(DeviceSegmentedSortKernelWithReorderingSmall<
                    IS_DESCENDING,
                    SmallAndMediumPolicyT,
                    KeyT,
                    ValueT,
                    BeginOffsetIteratorT,
                    EndOffsetIteratorT,
                    OffsetT>,
                  small_segments,
                  medium_segments,
                  medium_blocks,
                  d_small_segments_reordering,
                  d_large_and_medium_segments_reordering + num_segments -
                    medium_segments,
                  d_keys_in,
                  d_keys_out,
                  d_values_in,
                  d_values_out,
                  d_begin_offsets,
                  d_end_offsets);
        }
      }
      else
      {
        const unsigned int blocks_in_grid = num_segments;
        const unsigned int threads_in_block =
          LargeSegmentPolicyT::BLOCK_THREADS;

        // Log kernel configuration
        if (debug_synchronous)
        {
          _CubLog("Invoking DeviceSegmentedSortFallbackKernel<<<%d, %d, "
                  "0, %lld>>>(), %d items per thread, bit_grain %d\n",
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
          .doit(DeviceSegmentedSortFallbackKernel<IS_DESCENDING,
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
      }

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
