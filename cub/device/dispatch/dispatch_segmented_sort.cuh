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
#include "../../agent/agent_segmented_radix_sort.cuh"
#include "../../agent/agent_sub_warp_merge_sort.cuh"
#include "../../block/block_merge_sort.cuh"
#include "../../warp/warp_merge_sort.cuh"
#include "../../thread/thread_sort.cuh"

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/reverse_iterator.h>

#include <type_traits>

#ifndef ITEMS_PER_SMALL_SEG
#define ITEMS_PER_SMALL_SEG 9
#endif

#ifndef ITEMS_PER_MEDIUM_SEG
#define ITEMS_PER_MEDIUM_SEG 9
#endif

#ifndef THREADS_PER_SMALL_SEG
#define THREADS_PER_SMALL_SEG 4
#endif

#ifndef THREADS_PER_MEDIUM_SEG
#define THREADS_PER_MEDIUM_SEG 32
#endif

#ifndef SMALL_LOAD_ALG
#define SMALL_LOAD_ALG WarpLoadAlgorithm::WARP_LOAD_DIRECT
#endif

#ifndef MEDIUM_LOAD_ALG
#define MEDIUM_LOAD_ALG WarpLoadAlgorithm::WARP_LOAD_DIRECT
#endif

#ifndef LOAD_MOD
#define LOAD_MOD CacheLoadModifier::LOAD_DEFAULT
#endif


CUB_NAMESPACE_BEGIN


template <bool IS_DESCENDING,
          typename SegmentedPolicyT,
          typename MediumPolicyT,
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

  using AgentWarpMergeSortT =
    AgentSubWarpSort<IS_DESCENDING,
                     MediumPolicyT,
                     KeyT,
                     ValueT,
                     OffsetT>;

  __shared__ union
  {
    typename AgentSegmentedRadixSortT::TempStorage block_sort;
    typename WarpReduceT::TempStorage warp_reduce;
    typename AgentWarpMergeSortT::TempStorage medium_warp_sort;
  } temp_storage;

  AgentSegmentedRadixSortT agent(segment_begin,
                                 segment_end,
                                 num_items,
                                 temp_storage.block_sort);

  constexpr int begin_bit = 0;
  constexpr int end_bit = sizeof(KeyT) * 8;

  constexpr int cacheable_tile_size = SegmentedPolicyT::BLOCK_THREADS *
                                      SegmentedPolicyT::ITEMS_PER_THREAD;

  if (num_items <= MediumPolicyT::ITEMS_PER_TILE)
  {
    // Sort by a single warp
    if (threadIdx.x < MediumPolicyT::WARP_THREADS)
    {
      AgentWarpMergeSortT(temp_storage.medium_warp_sort)
        .ProcessSegment(num_items,
                        d_keys_in_origin + segment_begin,
                        d_keys_out_orig + segment_begin,
                        d_values_in_origin + segment_begin,
                        d_values_out_origin + segment_begin);
    }
  }
  else if (num_items < cacheable_tile_size)
  {
    // Sort by a CTA if data fits into shared memory
    agent.ProcessSmallSegment(begin_bit,
                              end_bit,
                              d_keys_in_origin,
                              d_values_in_origin,
                              d_keys_out_orig,
                              d_values_out_origin);
  }
  else
  {
    // Sort by a CTA with multiple reads from global memory
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

  using SmallPolicyT = typename SegmentedPolicyT::SmallPolicyT;
  using MediumPolicyT = typename SegmentedPolicyT::MediumPolicyT;

  constexpr int threads_per_medium_segment = MediumPolicyT::WARP_THREADS;
  constexpr int threads_per_small_segment = SmallPolicyT::WARP_THREADS;

  using MediumAgentWarpMergeSortT =
    AgentSubWarpSort<IS_DESCENDING, MediumPolicyT, KeyT, ValueT, OffsetT>;

  using SmallAgentWarpMergeSortT =
    AgentSubWarpSort<IS_DESCENDING, SmallPolicyT, KeyT, ValueT, OffsetT>;

  constexpr auto segments_per_medium_block =
    static_cast<OffsetT>(SegmentedPolicyT::SEGMENTS_PER_MEDIUM_BLOCK);

  constexpr auto segments_per_small_block =
    static_cast<OffsetT>(SegmentedPolicyT::SEGMENTS_PER_SMALL_BLOCK);

  __shared__ union
  {
    typename MediumAgentWarpMergeSortT::TempStorage medium[segments_per_medium_block];
    typename SmallAgentWarpMergeSortT::TempStorage small[segments_per_small_block];
  } temp_storage;

  if (bid < medium_blocks)
  {
    const OffsetT sid_within_block = tid / threads_per_medium_segment;
    const OffsetT reordered_segment_id = bid * segments_per_medium_block
                                       + sid_within_block;

    if (reordered_segment_id < medium_segments)
    {
      const OffsetT segment_id =
        d_medium_segments_reordering[reordered_segment_id];

      const OffsetT segment_begin = d_begin_offsets[segment_id];
      const OffsetT segment_end   = d_end_offsets[segment_id];
      const OffsetT num_items     = segment_end - segment_begin;

      MediumAgentWarpMergeSortT(temp_storage.medium[sid_within_block])
        .ProcessSegment(num_items,
                        d_keys_in_origin + segment_begin,
                        d_keys_out_orig + segment_begin,
                        d_values_in_origin + segment_begin,
                        d_values_out_origin + segment_begin);
    }
  }
  else
  {
    const OffsetT sid_within_block = tid / threads_per_small_segment;
    const OffsetT reordered_segment_id = (bid - medium_blocks) * segments_per_small_block
                                       + sid_within_block;

    if (reordered_segment_id < small_segments)
    {
      const OffsetT segment_id =
        d_small_segments_reordering[reordered_segment_id];

      const OffsetT segment_begin = d_begin_offsets[segment_id];
      const OffsetT segment_end   = d_end_offsets[segment_id];
      const OffsetT num_items     = segment_end - segment_begin;

      SmallAgentWarpMergeSortT(temp_storage.small[sid_within_block])
        .ProcessSegment(num_items,
                        d_keys_in_origin + segment_begin,
                        d_keys_out_orig + segment_begin,
                        d_values_in_origin + segment_begin,
                        d_values_out_origin + segment_begin);
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

  __host__ __device__ __forceinline__
  SegmentSizeGreaterThan(
    T value,
    BeginOffsetIteratorT d_offset_begin,
    EndOffsetIteratorT d_offset_end)
    : value(value)
    , d_offset_begin(d_offset_begin)
    , d_offset_end(d_offset_end)
  {}

  __host__ __device__ __forceinline__
  bool operator()(unsigned int segment_id) const
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

  __host__ __device__ __forceinline__
  SegmentSizeLessThan(
    T value,
    BeginOffsetIteratorT d_offset_begin,
    EndOffsetIteratorT d_offset_end)
    : value(value)
    , d_offset_begin(d_offset_begin)
    , d_offset_end(d_offset_end)
  {}

  __host__ __device__ __forceinline__
  bool operator()(unsigned int segment_id) const
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

  // TODO uint32, uint64, pairs
  struct Policy350 : ChainedPolicy<350, Policy350, Policy350>
  {
    constexpr static int BLOCK_THREADS = 128;
    constexpr static int RADIX_BITS = sizeof(KeyT) > 1 ? 6 : 5;

    using LargeSegmentPolicy =
      AgentRadixSortDownsweepPolicy<BLOCK_THREADS,
                                    9,
                                    DominantT,
                                    BLOCK_LOAD_WARP_TRANSPOSE,
                                    LOAD_LDG,
                                    RADIX_RANK_MATCH,
                                    BLOCK_SCAN_WARP_SCANS,
                                    RADIX_BITS>;

    constexpr static int ITEMS_PER_SMALL_THREAD =
      Nominal4BItemsToItems<DominantT>(ITEMS_PER_SMALL_SEG);

    constexpr static int ITEMS_PER_MEDIUM_THREAD =
      Nominal4BItemsToItems<DominantT>(ITEMS_PER_MEDIUM_SEG);

    using SmallAndMediumSegmentedSortPolicyT =
    AgentSmallAndMediumSegmentedSortPolicy<

      BLOCK_THREADS,

      // Small policy
      cub::AgentSubWarpMergeSortPolicy<THREADS_PER_SMALL_SEG, // Threads per
        // segment
        ITEMS_PER_SMALL_THREAD,
        SMALL_LOAD_ALG,
        LOAD_MOD>,

      // Medium policy
      cub::AgentSubWarpMergeSortPolicy<THREADS_PER_MEDIUM_SEG, // Threads per
        // segment
        ITEMS_PER_MEDIUM_THREAD,
        MEDIUM_LOAD_ALG,
        LOAD_MOD>>;
  };

  // TODO uint32, uint64, pairs
  struct Policy500 : ChainedPolicy<500, Policy500, Policy350>
  {
    constexpr static int BLOCK_THREADS = 256;
    constexpr static int RADIX_BITS = sizeof(KeyT) > 1 ? 6 : 4;

    using LargeSegmentPolicy =
    AgentRadixSortDownsweepPolicy<BLOCK_THREADS,
      16,
      DominantT,
                                    BLOCK_LOAD_DIRECT,
                                    LOAD_LDG,
                                    RADIX_RANK_MEMOIZE,
                                    BLOCK_SCAN_RAKING_MEMOIZE,
                                    RADIX_BITS>;

    constexpr static int ITEMS_PER_SMALL_THREAD =
      Nominal4BItemsToItems<DominantT>(ITEMS_PER_SMALL_SEG);

    constexpr static int ITEMS_PER_MEDIUM_THREAD =
      Nominal4BItemsToItems<DominantT>(ITEMS_PER_MEDIUM_SEG);

    using SmallAndMediumSegmentedSortPolicyT =
      AgentSmallAndMediumSegmentedSortPolicy<

        BLOCK_THREADS,

        // Small policy
        cub::AgentSubWarpMergeSortPolicy<THREADS_PER_SMALL_SEG, // Threads per segment
                                         ITEMS_PER_SMALL_THREAD,
                                         SMALL_LOAD_ALG,
                                         LOAD_MOD>,

        // Medium policy
        cub::AgentSubWarpMergeSortPolicy<THREADS_PER_MEDIUM_SEG, // Threads per segment
                                         ITEMS_PER_MEDIUM_THREAD,
                                         MEDIUM_LOAD_ALG,
                                         LOAD_MOD>>;
  };

  // TODO uint32, uint64, pairs
  struct Policy600 : ChainedPolicy<600, Policy600, Policy500>
  {
    constexpr static int BLOCK_THREADS = 256;
    constexpr static int RADIX_BITS = sizeof(KeyT) > 1 ? 6 : 5;

    using LargeSegmentPolicy =
    AgentRadixSortDownsweepPolicy<BLOCK_THREADS,
      19,
      DominantT,
                                    BLOCK_LOAD_TRANSPOSE,
                                    LOAD_DEFAULT,
                                    RADIX_RANK_MATCH,
                                    BLOCK_SCAN_WARP_SCANS,
                                    RADIX_BITS>;

    constexpr static int ITEMS_PER_SMALL_THREAD =
      Nominal4BItemsToItems<DominantT>(ITEMS_PER_SMALL_SEG);

    constexpr static int ITEMS_PER_MEDIUM_THREAD =
      Nominal4BItemsToItems<DominantT>(ITEMS_PER_MEDIUM_SEG);

    using SmallAndMediumSegmentedSortPolicyT =
      AgentSmallAndMediumSegmentedSortPolicy<

        BLOCK_THREADS,

        // Small policy
        cub::AgentSubWarpMergeSortPolicy<THREADS_PER_SMALL_SEG, // Threads per segment
                                         ITEMS_PER_SMALL_THREAD,
                                         SMALL_LOAD_ALG,
                                         LOAD_MOD>,

        // Medium policy
        cub::AgentSubWarpMergeSortPolicy<THREADS_PER_MEDIUM_SEG, // Threads per segment
                                         ITEMS_PER_MEDIUM_THREAD,
                                         MEDIUM_LOAD_ALG,
                                         LOAD_MOD>>;
  };

  // TODO uint32, uint64, pairs
  struct Policy610 : ChainedPolicy<610, Policy610, Policy600>
  {
    constexpr static int BLOCK_THREADS = 256;
    constexpr static int RADIX_BITS = sizeof(KeyT) > 1 ? 6 : 5;

    using LargeSegmentPolicy =
    AgentRadixSortDownsweepPolicy<BLOCK_THREADS,
      19,
      DominantT,
                                    BLOCK_LOAD_DIRECT,
                                    LOAD_LDG,
                                    RADIX_RANK_MEMOIZE,
                                    BLOCK_SCAN_WARP_SCANS,
                                    RADIX_BITS>;

    constexpr static int ITEMS_PER_SMALL_THREAD =
      Nominal4BItemsToItems<DominantT>(ITEMS_PER_SMALL_SEG);

    constexpr static int ITEMS_PER_MEDIUM_THREAD =
      Nominal4BItemsToItems<DominantT>(ITEMS_PER_MEDIUM_SEG);

    using SmallAndMediumSegmentedSortPolicyT =
      AgentSmallAndMediumSegmentedSortPolicy<

        BLOCK_THREADS,

        // Small policy
        cub::AgentSubWarpMergeSortPolicy<THREADS_PER_SMALL_SEG, // Threads per segment
                                         ITEMS_PER_SMALL_THREAD,
                                         SMALL_LOAD_ALG,
                                         LOAD_MOD>,

        // Medium policy
        cub::AgentSubWarpMergeSortPolicy<THREADS_PER_MEDIUM_SEG, // Threads per segment
                                         ITEMS_PER_MEDIUM_THREAD,
                                         MEDIUM_LOAD_ALG,
                                         LOAD_MOD>>;
  };

  // TODO uint32, uint64, pairs
  struct Policy620 : ChainedPolicy<620, Policy620, Policy610>
  {
    constexpr static int BLOCK_THREADS = 256;
    constexpr static int RADIX_BITS = 5;

    using LargeSegmentPolicy =
      AgentRadixSortDownsweepPolicy<BLOCK_THREADS,
                                    16,
                                    DominantT,
                                    BLOCK_LOAD_TRANSPOSE,
                                    LOAD_DEFAULT,
                                    RADIX_RANK_MEMOIZE,
                                    BLOCK_SCAN_RAKING_MEMOIZE,
                                    RADIX_BITS>;

    constexpr static int ITEMS_PER_SMALL_THREAD =
      Nominal4BItemsToItems<DominantT>(ITEMS_PER_SMALL_SEG);

    constexpr static int ITEMS_PER_MEDIUM_THREAD =
      Nominal4BItemsToItems<DominantT>(ITEMS_PER_MEDIUM_SEG);

    using SmallAndMediumSegmentedSortPolicyT =
      AgentSmallAndMediumSegmentedSortPolicy<

        BLOCK_THREADS,

        // Small policy
        cub::AgentSubWarpMergeSortPolicy<THREADS_PER_SMALL_SEG, // Threads per segment
                                         ITEMS_PER_SMALL_THREAD,
                                         SMALL_LOAD_ALG,
                                         LOAD_MOD>,

        // Medium policy
        cub::AgentSubWarpMergeSortPolicy<THREADS_PER_MEDIUM_SEG, // Threads per segment
                                         ITEMS_PER_MEDIUM_THREAD,
                                         MEDIUM_LOAD_ALG,
                                         LOAD_MOD>>;
  };

  // TODO uint32, uint64, pairs
  struct Policy700 : ChainedPolicy<700, Policy700, Policy620>
  {
    constexpr static int BLOCK_THREADS = 256;
    constexpr static int RADIX_BITS = sizeof(KeyT) > 1 ? 6 : 5;

    using LargeSegmentPolicy =
    AgentRadixSortDownsweepPolicy<BLOCK_THREADS,
      19,
      DominantT,
                                    BLOCK_LOAD_DIRECT,
                                    LOAD_LDG,
                                    RADIX_RANK_MEMOIZE,
                                    BLOCK_SCAN_WARP_SCANS,
                                    RADIX_BITS>;

    constexpr static int ITEMS_PER_SMALL_THREAD =
      Nominal4BItemsToItems<DominantT>(ITEMS_PER_SMALL_SEG);

    constexpr static int ITEMS_PER_MEDIUM_THREAD =
      Nominal4BItemsToItems<DominantT>(ITEMS_PER_MEDIUM_SEG);

    using SmallAndMediumSegmentedSortPolicyT =
      AgentSmallAndMediumSegmentedSortPolicy<

        BLOCK_THREADS,

        // Small policy
        cub::AgentSubWarpMergeSortPolicy<THREADS_PER_SMALL_SEG, // Threads per segment
                                         ITEMS_PER_SMALL_THREAD,
                                         SMALL_LOAD_ALG,
                                         LOAD_MOD>,

        // Medium policy
        cub::AgentSubWarpMergeSortPolicy<THREADS_PER_MEDIUM_SEG, // Threads per segment
                                         ITEMS_PER_MEDIUM_THREAD,
                                         MEDIUM_LOAD_ALG,
                                         LOAD_MOD>>;
  };

  // TODO uint64, pairs
  struct Policy750 : ChainedPolicy<750, Policy750, Policy700>
  {
    constexpr static int BLOCK_THREADS = 256;
    constexpr static int RADIX_BITS    = sizeof(KeyT) > 1 ? 6 : 5;

    using LargeSegmentPolicy =
      AgentRadixSortDownsweepPolicy<BLOCK_THREADS,
                                    19,
                                    DominantT,
                                    BLOCK_LOAD_DIRECT,
                                    LOAD_LDG,
                                    RADIX_RANK_MEMOIZE,
                                    BLOCK_SCAN_WARP_SCANS,
                                    RADIX_BITS>;

    constexpr static int ITEMS_PER_SMALL_THREAD =
      Nominal4BItemsToItems<DominantT>(11);

    constexpr static int ITEMS_PER_MEDIUM_THREAD =
      Nominal4BItemsToItems<DominantT>(11);

    using SmallAndMediumSegmentedSortPolicyT =
      AgentSmallAndMediumSegmentedSortPolicy<

        BLOCK_THREADS,

        // Small policy
        cub::AgentSubWarpMergeSortPolicy<4, // Threads per segment
                                         ITEMS_PER_SMALL_THREAD,
                                         WarpLoadAlgorithm::WARP_LOAD_DIRECT,
                                         CacheLoadModifier::LOAD_DEFAULT>,

        // Medium policy
        cub::AgentSubWarpMergeSortPolicy<32, // Threads per segment
                                         ITEMS_PER_MEDIUM_THREAD,
                                         WarpLoadAlgorithm::WARP_LOAD_TRANSPOSE,
                                         CacheLoadModifier::LOAD_DEFAULT>>;
  };

  // TODO uint64, pairs
  struct Policy800 : ChainedPolicy<800, Policy800, Policy750>
  {
    constexpr static int BLOCK_THREADS = 256;

    using LargeSegmentPolicy =
      cub::AgentRadixSortDownsweepPolicy<BLOCK_THREADS,
                                         23,
                                         DominantT,
                                         cub::BLOCK_LOAD_TRANSPOSE,
                                         cub::LOAD_DEFAULT,
                                         cub::RADIX_RANK_MEMOIZE,
                                         cub::BLOCK_SCAN_WARP_SCANS,
                                         6>;

    constexpr static int ITEMS_PER_SMALL_THREAD =
      Nominal4BItemsToItems<DominantT>(9);

    constexpr static int ITEMS_PER_MEDIUM_THREAD =
      Nominal4BItemsToItems<DominantT>(7);

    using SmallAndMediumSegmentedSortPolicyT =
      AgentSmallAndMediumSegmentedSortPolicy<

        BLOCK_THREADS,

        // Small policy
        cub::AgentSubWarpMergeSortPolicy<4, // Threads per segment
                                         ITEMS_PER_SMALL_THREAD,
                                         WarpLoadAlgorithm::WARP_LOAD_TRANSPOSE,
                                         CacheLoadModifier::LOAD_LDG>,

        // Medium policy
        cub::AgentSubWarpMergeSortPolicy<32, // Threads per segment
                                         ITEMS_PER_MEDIUM_THREAD,
                                         WarpLoadAlgorithm::WARP_LOAD_TRANSPOSE,
                                         CacheLoadModifier::LOAD_LDG>>;
  };

  // TODO pairs
  struct Policy860 : ChainedPolicy<860, Policy860, Policy800>
  {
    constexpr static int BLOCK_THREADS = 256;

    using LargeSegmentPolicy =
      cub::AgentRadixSortDownsweepPolicy<BLOCK_THREADS,
                                         23,
                                         DominantT,
                                         cub::BLOCK_LOAD_TRANSPOSE,
                                         cub::LOAD_DEFAULT,
                                         cub::RADIX_RANK_MEMOIZE,
                                         cub::BLOCK_SCAN_WARP_SCANS,
                                         6>;

    constexpr static bool LARGE_ITEMS = sizeof(DominantT) > 4;

    constexpr static int ITEMS_PER_SMALL_THREAD =
      Nominal4BItemsToItems<DominantT>(LARGE_ITEMS ? 7 : 9);

    constexpr static int ITEMS_PER_MEDIUM_THREAD =
      Nominal4BItemsToItems<DominantT>(LARGE_ITEMS ? 9 : 7);

    using SmallAndMediumSegmentedSortPolicyT =
      AgentSmallAndMediumSegmentedSortPolicy<

        BLOCK_THREADS,

        // Small policy
        cub::AgentSubWarpMergeSortPolicy<(LARGE_ITEMS ? 8 : 2), // Threads per segment
                                         ITEMS_PER_SMALL_THREAD,
                                         WarpLoadAlgorithm::WARP_LOAD_TRANSPOSE,
                                         CacheLoadModifier::LOAD_LDG>,

        // Medium policy
        cub::AgentSubWarpMergeSortPolicy<16, // Threads per segment
                                         ITEMS_PER_MEDIUM_THREAD,
                                         WarpLoadAlgorithm::WARP_LOAD_TRANSPOSE,
                                         CacheLoadModifier::LOAD_LDG>>;
  };

  /// MaxPolicy
  using MaxPolicy = Policy860;
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

  using LargeSegmentsSelectorT =
    SegmentSizeGreaterThan<OffsetT, BeginOffsetIteratorT, EndOffsetIteratorT>;

  using SmallSegmentsSelectorT =
    SegmentSizeLessThan<OffsetT, BeginOffsetIteratorT, EndOffsetIteratorT>;

  // Partition selects large and small groups.
  // The middle group is unselected.
  constexpr static std::size_t num_selected_groups = 2;

  void *d_temp_storage;
  std::size_t &temp_storage_bytes;
  DoubleBuffer<KeyT> &d_keys;
  DoubleBuffer<ValueT> &d_values;
  OffsetT num_items;
  OffsetT num_segments;
  BeginOffsetIteratorT d_begin_offsets;
  EndOffsetIteratorT d_end_offsets;
  bool is_overwrite_okay;
  cudaStream_t stream;
  bool debug_synchronous;

  /// Constructor
  CUB_RUNTIME_FUNCTION __forceinline__
  DispatchSegmentedSort(void *d_temp_storage,
                        std::size_t &temp_storage_bytes,
                        DoubleBuffer<KeyT> &d_keys,
                        DoubleBuffer<ValueT> &d_values,
                        OffsetT num_items,
                        OffsetT num_segments,
                        BeginOffsetIteratorT d_begin_offsets,
                        EndOffsetIteratorT d_end_offsets,
                        bool is_overwrite_okay,
                        cudaStream_t stream,
                        bool debug_synchronous)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_keys(d_keys)
      , d_values(d_values)
      , num_items(num_items)
      , num_segments(num_segments)
      , d_begin_offsets(d_begin_offsets)
      , d_end_offsets(d_end_offsets)
      , is_overwrite_okay(is_overwrite_okay)
      , stream(stream)
      , debug_synchronous(debug_synchronous)
  {}

  /// Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t Invoke()
  {
    using LargeSegmentPolicyT = typename ActivePolicyT::LargeSegmentPolicy;
    using SmallAndMediumPolicyT =
      typename ActivePolicyT::SmallAndMediumSegmentedSortPolicyT;

    constexpr int radix_bits = LargeSegmentPolicyT::RADIX_BITS;

    cudaError error = cudaSuccess;

    do
    {
      if (num_items == 0 || num_segments == 0)
      {
        temp_storage_bytes = 0;
        break;
      }

      //------------------------------------------------------------------------
      // Prepare temporary storage layout
      //------------------------------------------------------------------------

      const bool reorder_segments = num_segments > 500; // TODO Magick number

      TemporaryStorage::Layout<5> temporary_storage_layout;

      auto keys_slot = temporary_storage_layout.GetSlot(0);
      auto values_slot = temporary_storage_layout.GetSlot(1);
      auto large_and_medium_reordering_slot = temporary_storage_layout.GetSlot(2);
      auto small_reordering_slot = temporary_storage_layout.GetSlot(3);
      auto group_sizes_slot = temporary_storage_layout.GetSlot(4);

      auto keys_allocation = keys_slot->GetAlias<KeyT>();
      auto values_allocation = values_slot->GetAlias<ValueT>();

      if (!is_overwrite_okay)
      {
        keys_allocation.Grow(num_items);

        if (!KEYS_ONLY)
        {
          values_allocation.Grow(num_items);
        }
      }

      auto large_and_medium_segments_reordering = large_and_medium_reordering_slot->GetAlias<OffsetT>();
      auto small_segments_reordering = small_reordering_slot->GetAlias<OffsetT>();
      auto group_sizes = group_sizes_slot->GetAlias<OffsetT>();

      std::size_t three_way_partition_temp_storage_bytes {};

      LargeSegmentsSelectorT large_segments_selector(
        SmallAndMediumPolicyT::MediumPolicyT::ITEMS_PER_TILE,
        d_begin_offsets,
        d_end_offsets);

      SmallSegmentsSelectorT small_segments_selector(
        SmallAndMediumPolicyT::SmallPolicyT::ITEMS_PER_TILE + 1,
        d_begin_offsets,
        d_end_offsets);

      auto device_partition_temp_storage = keys_slot->GetAlias<std::uint8_t>();
      if (reorder_segments)
      {
        large_and_medium_segments_reordering.Grow(num_segments);
        small_segments_reordering.Grow(num_segments);
        group_sizes.Grow(num_selected_groups);

        auto medium_reordering_iterator =
          THRUST_NS_QUALIFIER::make_reverse_iterator(
            large_and_medium_segments_reordering.Get());

        cub::DevicePartition::If(
          nullptr,
          three_way_partition_temp_storage_bytes,
          THRUST_NS_QUALIFIER::counting_iterator<OffsetT>(0),
          large_and_medium_segments_reordering.Get(),
          small_segments_reordering.Get(),
          medium_reordering_iterator,
          group_sizes.Get(),
          num_segments,
          large_segments_selector,
          small_segments_selector,
          stream,
          debug_synchronous);

        device_partition_temp_storage.Grow(
          three_way_partition_temp_storage_bytes);
      }

      if (d_temp_storage == nullptr)
      {
        temp_storage_bytes = temporary_storage_layout.GetSize();

        // Return if the caller is simply requesting the size of the storage allocation
        break;
      }

      if (CubDebug(
            error = temporary_storage_layout.MapToBuffer(d_temp_storage,
                                                         temp_storage_bytes)))
      {
        break;
      }

      //------------------------------------------------------------------------
      // Sort
      //------------------------------------------------------------------------

      const bool is_num_passes_odd = GetNumPasses(radix_bits) & 1;

      DeviceDoubleBuffer<KeyT> d_keys_remaining_passes(
        (is_overwrite_okay || is_num_passes_odd) ? d_keys.Alternate() : keys_allocation.Get(),
        (is_overwrite_okay) ? d_keys.Current() : (is_num_passes_odd) ? keys_allocation.Get() : d_keys.Alternate());

      DeviceDoubleBuffer<ValueT> d_values_remaining_passes(
        (is_overwrite_okay || is_num_passes_odd) ? d_values.Alternate() : values_allocation.Get(),
        (is_overwrite_okay) ? d_values.Current() : (is_num_passes_odd) ? values_allocation.Get() : d_values.Alternate());

      if (reorder_segments)
      {
        error =
          SortWithReordering<LargeSegmentPolicyT, SmallAndMediumPolicyT>(
            three_way_partition_temp_storage_bytes,
            d_keys_remaining_passes,
            d_values_remaining_passes,
            large_segments_selector,
            small_segments_selector,
            device_partition_temp_storage,
            large_and_medium_segments_reordering,
            small_segments_reordering,
            group_sizes);
      }
      else
      {
        using MediumPolicyT = typename SmallAndMediumPolicyT::MediumPolicyT;
        error = SortWithoutReordering<LargeSegmentPolicyT, MediumPolicyT>(
          d_keys_remaining_passes,
          d_values_remaining_passes);
      }

      d_keys.selector = GetFinalSelector(d_keys.selector, radix_bits);
      d_values.selector = GetFinalSelector(d_values.selector, radix_bits);

    } while (false);

    return error;
  }

  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(void *d_temp_storage,
           std::size_t &temp_storage_bytes,
           DoubleBuffer<KeyT> &d_keys,
           DoubleBuffer<ValueT> &d_values,
           OffsetT num_items,
           OffsetT num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           bool is_overwrite_okay,
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
                                     d_keys,
                                     d_values,
                                     num_items,
                                     num_segments,
                                     d_begin_offsets,
                                     d_end_offsets,
                                     is_overwrite_okay,
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

private:
  CUB_RUNTIME_FUNCTION __forceinline__
  int GetNumPasses(int radix_bits)
  {
    const int byte_size  = 8;
    const int num_bits   = sizeof(KeyT) * byte_size;
    const int num_passes = DivideAndRoundUp(num_bits, radix_bits);
    return num_passes;
  }

  CUB_RUNTIME_FUNCTION __forceinline__
  int GetFinalSelector(int selector, int radix_bits)
  {
    // Sorted data always ends up in the other vector
    if (!is_overwrite_okay)
    {
      return (selector + 1) & 1;
    }

    return (selector + GetNumPasses(radix_bits)) & 1;
  }

  template <typename T>
  CUB_RUNTIME_FUNCTION __forceinline__
  T* GetFinalOutput(int radix_bits,
                    DoubleBuffer<T> &buffer)
  {
    const int final_selector = GetFinalSelector(buffer.selector, radix_bits);
    return buffer.d_buffers[final_selector];
  }

  template <typename LargeSegmentPolicyT,
            typename SmallAndMediumPolicyT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t
  SortWithReordering(
    std::size_t three_way_partition_temp_storage_bytes,
    cub::DeviceDoubleBuffer<KeyT> &d_keys_remaining_passes,
    cub::DeviceDoubleBuffer<ValueT> &d_values_remaining_passes,
    LargeSegmentsSelectorT &large_segments_selector,
    SmallSegmentsSelectorT &small_segments_selector,
    TemporaryStorage::Array<std::uint8_t> &device_partition_temp_storage,
    TemporaryStorage::Array<OffsetT> &large_and_medium_segments_reordering,
    TemporaryStorage::Array<OffsetT> &small_segments_reordering,
    TemporaryStorage::Array<OffsetT> &group_sizes)
  {
    cudaError_t error = cudaSuccess;

    auto medium_reordering_iterator =
      THRUST_NS_QUALIFIER::make_reverse_iterator(
        large_and_medium_segments_reordering.Get() + num_segments);

    if (CubDebug(error = cub::DevicePartition::If(
      device_partition_temp_storage.Get(),
      three_way_partition_temp_storage_bytes,
      THRUST_NS_QUALIFIER::counting_iterator<OffsetT>(0),
      large_and_medium_segments_reordering.Get(),
      small_segments_reordering.Get(),
      medium_reordering_iterator,
      group_sizes.Get(),
      num_segments,
      large_segments_selector,
      small_segments_selector,
      stream,
      debug_synchronous)))
    {
      return error;
    }

#ifdef __CUDA_ARCH__
    OffsetT *h_group_sizes = group_sizes.Get();
#else
    OffsetT h_group_sizes[num_selected_groups];
    if (CubDebug(error = cudaMemcpy(h_group_sizes,
                                    group_sizes.Get(),
                                    num_selected_groups * sizeof(OffsetT),
                                    cudaMemcpyDeviceToHost)))
    {
      return error;
    }
#endif

    const OffsetT large_segments = h_group_sizes[0];

    if (large_segments > 0)
    {
      const OffsetT blocks_in_grid = large_segments; // One CTA per segment

      // TODO cudaGraph
      if (debug_synchronous)
      {
        _CubLog("Invoking "
                "DeviceSegmentedSortKernelWithReorderingLarge<<<"
                "%d, %d, 0, %lld>>>()\n",
                static_cast<int>(blocks_in_grid),
                SmallAndMediumPolicyT::BLOCK_THREADS,
                (long long)stream);
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
              large_and_medium_segments_reordering.Get(),
              d_keys.Current(),
              GetFinalOutput(LargeSegmentPolicyT::RADIX_BITS, d_keys),
              d_keys_remaining_passes,
              d_values.Current(),
              GetFinalOutput(LargeSegmentPolicyT::RADIX_BITS, d_values),
              d_values_remaining_passes,
              d_begin_offsets,
              d_end_offsets);

      // Check for failure to launch
      if (CubDebug(error = cudaPeekAtLastError()))
      {
        return error;
      }

      // Sync the stream if specified to flush runtime errors
      if (debug_synchronous)
      {
        if (CubDebug(error = SyncStream(stream)))
        {
          return error;
        }
      }
    }

    const OffsetT small_segments = h_group_sizes[1];
    const OffsetT medium_segments = num_segments -
                                    (large_segments + small_segments);

    const OffsetT small_blocks =
      DivideAndRoundUp(small_segments,
                       SmallAndMediumPolicyT::SEGMENTS_PER_SMALL_BLOCK);

    const OffsetT medium_blocks =
      DivideAndRoundUp(medium_segments,
                       SmallAndMediumPolicyT::SEGMENTS_PER_MEDIUM_BLOCK);


    const OffsetT small_and_medium_blocks_in_grid = small_blocks +
                                                    medium_blocks;

    if (small_and_medium_blocks_in_grid)
    {
      if (debug_synchronous)
      {
        _CubLog("Invoking "
                "DeviceSegmentedSortKernelWithReorderingSmall<<<"
                "%d, %d, 0, %lld>>>()\n",
                static_cast<int>(small_and_medium_blocks_in_grid),
                SmallAndMediumPolicyT::BLOCK_THREADS,
                (long long)stream);
      }

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
              small_segments_reordering.Get(),
              large_and_medium_segments_reordering.Get() + num_segments -
              medium_segments,
              d_keys.Current(),
              GetFinalOutput(LargeSegmentPolicyT::RADIX_BITS, d_keys),
              d_values.Current(),
              GetFinalOutput(LargeSegmentPolicyT::RADIX_BITS, d_values),
              d_begin_offsets,
              d_end_offsets);

      // Check for failure to launch
      if (CubDebug(error = cudaPeekAtLastError()))
      {
        return error;
      }

      // Sync the stream if specified to flush runtime errors
      if (debug_synchronous)
      {
        if (CubDebug(error = SyncStream(stream)))
        {
          return error;
        }
      }
    }

    return error;
  }

  template <typename LargeSegmentPolicyT,
            typename MediumPolicyT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t SortWithoutReordering(
    cub::DeviceDoubleBuffer<KeyT> &d_keys_remaining_passes,
    cub::DeviceDoubleBuffer<ValueT> &d_values_remaining_passes)
  {
    cudaError_t error = cudaSuccess;

    const unsigned int blocks_in_grid   = num_segments;
    const unsigned int threads_in_block = LargeSegmentPolicyT::BLOCK_THREADS;

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
                                              MediumPolicyT,
                                              KeyT,
                                              ValueT,
                                              BeginOffsetIteratorT,
                                              EndOffsetIteratorT,
                                              OffsetT>,
            d_keys.Current(),
            GetFinalOutput(LargeSegmentPolicyT::RADIX_BITS, d_keys),
            d_keys_remaining_passes,
            d_values.Current(),
            GetFinalOutput(LargeSegmentPolicyT::RADIX_BITS, d_values),
            d_values_remaining_passes,
            d_begin_offsets,
            d_end_offsets);

    // Check for failure to launch
    if (CubDebug(error = cudaPeekAtLastError()))
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    if (debug_synchronous)
    {
      if (CubDebug(error = SyncStream(stream)))
      {
        return error;
      }
    }

    return error;
  }
};


CUB_NAMESPACE_END
