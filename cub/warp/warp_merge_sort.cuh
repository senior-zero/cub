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

#include "../config.cuh"
#include "../util_ptx.cuh"
#include "../util_type.cuh"
#include "../block/block_merge_sort.cuh"

CUB_NAMESPACE_BEGIN

/**
 * \addtogroup WarpModule
 * @{
 */

/**
 *
 * @tparam KeyT
 * @tparam ValueT
 * @tparam LOGICAL_WARP_THREADS must be a power-of-two (less or equal to 32)
 * @tparam PTX_ARCH
 */
template <
  typename    KeyT,
  typename    ValueT,
  int         ITEMS_PER_THREAD,
  int         LOGICAL_WARP_THREADS    = CUB_PTX_WARP_THREADS,
  int         PTX_ARCH                = CUB_PTX_ARCH>
class WarpMergeSort
{
private:
  static_assert(PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE,
                "LOGICAL_WARP_THREADS must be a power of two");

  constexpr static bool IS_ARCH_WARP = LOGICAL_WARP_THREADS == CUB_WARP_THREADS(PTX_ARCH);
  constexpr static bool KEYS_ONLY = cub::Equals<ValueT, cub::NullType>::VALUE;
  constexpr static int TILE_SIZE = ITEMS_PER_THREAD * LOGICAL_WARP_THREADS;

  union _TempStorage
  {
    KeyT keys[TILE_SIZE + 1];
    ValueT items[TILE_SIZE + 1];
  };

  _TempStorage &temp_storage;

public:
  const unsigned int lane_id;
  const unsigned int warp_id;
  const unsigned int member_mask;

  struct TempStorage : Uninitialized<_TempStorage> {};

  WarpMergeSort() = delete;

  __device__ __forceinline__ WarpMergeSort(TempStorage &temp_storage)
      : temp_storage(temp_storage.Alias())
      , lane_id(IS_ARCH_WARP ? LaneId() : LaneId() % LOGICAL_WARP_THREADS)
      , warp_id(IS_ARCH_WARP ? 0 : (LaneId() / LOGICAL_WARP_THREADS))
      , member_mask(WarpMask<LOGICAL_WARP_THREADS, PTX_ARCH>(warp_id))
  {
  }

  template <typename CompareOp>
  __device__ __forceinline__ void
  Sort(KeyT (&keys)[ITEMS_PER_THREAD],     ///< [in-out] Keys to sort
       ValueT (&items)[ITEMS_PER_THREAD],  ///< [in-out] Values to sort
       CompareOp compare_op,               ///< [in] Comparison function object which returns true if the first argument is ordered before the second
       int valid_items,                    ///< [in] Number of valid items to sort
       KeyT oob_default)                   ///< [in] Default value to assign out-of-bound items
  {
    if (valid_items < 2)
    {
      return;
    }

    int indices[ITEMS_PER_THREAD];

    KeyT max_key = oob_default;

    #pragma unroll
    for (int item = 1; item < ITEMS_PER_THREAD; ++item)
    {
      if (ITEMS_PER_THREAD * lane_id + item < valid_items)
      {
        max_key = compare_op(max_key, keys[item]) ? keys[item] : max_key;
      }
      else
      {
        keys[item] = max_key;
      }
    }

    if (lane_id * ITEMS_PER_THREAD < valid_items)
    {
      StableOddEvenSort(keys, items, compare_op);
    }

    if (valid_items > ITEMS_PER_THREAD)
    {
      #pragma unroll
      for (int target_merged_threads_number = 2;
           target_merged_threads_number <= LOGICAL_WARP_THREADS;
           target_merged_threads_number *= 2)
      {
        int merged_threads_number = target_merged_threads_number / 2;
        int mask                  = target_merged_threads_number - 1;

        WARP_SYNC(member_mask);

        for (int item = 0; item < ITEMS_PER_THREAD; item++)
        {
          temp_storage.keys[lane_id * ITEMS_PER_THREAD + item] = keys[item];
        }
        WARP_SYNC(member_mask);

        int first_thread_idx_in_thread_group_being_merged = ~mask & lane_id;
        int size = ITEMS_PER_THREAD * merged_threads_number;
        int start = ITEMS_PER_THREAD *
                    first_thread_idx_in_thread_group_being_merged;

        int thread_idx_in_thread_group_being_merged = mask & lane_id;

        int diag = (cub::min)(valid_items,
                              ITEMS_PER_THREAD *
                                thread_idx_in_thread_group_being_merged);

        int keys1_beg = (cub::min)(valid_items, start);
        int keys1_end = (cub::min)(valid_items, keys1_beg + size);
        int keys2_beg = keys1_end;
        int keys2_end = (cub::min)(valid_items, keys2_beg + size);

        int keys1_count = keys1_end - keys1_beg;
        int keys2_count = keys2_end - keys2_beg;

        int partition_diag = MergePath<KeyT>(&temp_storage.keys[keys1_beg],
                                             &temp_storage.keys[keys2_beg],
                                             keys1_count,
                                             keys2_count,
                                             diag,
                                             compare_op);

        int keys1_beg_loc   = keys1_beg + partition_diag;
        int keys1_end_loc   = keys1_end;
        int keys2_beg_loc   = keys2_beg + diag - partition_diag;
        int keys2_end_loc   = keys2_end;
        int keys1_count_loc = keys1_end_loc - keys1_beg_loc;
        int keys2_count_loc = keys2_end_loc - keys2_beg_loc;

        SerialMerge(&temp_storage.keys[0],
                    keys1_beg_loc,
                    keys2_beg_loc,
                    keys1_count_loc,
                    keys2_count_loc,
                    keys,
                    indices,
                    compare_op);

        if (!KEYS_ONLY)
        {
          WARP_SYNC(member_mask);

          #pragma unroll
          for (int item = 0; item < ITEMS_PER_THREAD; item++)
          {
            int idx                  = ITEMS_PER_THREAD * lane_id + item;
            temp_storage.items[idx] = items[item];
          }
          WARP_SYNC(member_mask);

          // gather items from shmem
          //
          #pragma unroll
          for (int item = 0; item < ITEMS_PER_THREAD; item++)
          {
            items[item] = temp_storage.items[indices[item]];
          }
        }
      }
    }
  }
};

/** @} */       // end group WarpModule

CUB_NAMESPACE_END
