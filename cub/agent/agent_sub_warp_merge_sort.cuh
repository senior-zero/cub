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
#include "../util_type.cuh"
#include "../warp/warp_merge_sort.cuh"

CUB_NAMESPACE_BEGIN

template <bool IS_DESCENDING,
          int ITEMS_PER_THREAD,
          int THREADS_PER_SEGMENT,
          typename KeyT,
          typename ValueT,
          typename OffsetT>
class AgentSubWarpSort
{
public:
  static constexpr bool KEYS_ONLY = cub::Equals<ValueT, cub::NullType>::VALUE;

  using WarpMergeSortT =
    WarpMergeSort<KeyT, ITEMS_PER_THREAD, THREADS_PER_SEGMENT, ValueT>;

  using TempStorage = typename WarpMergeSortT::TempStorage;

  __device__ __forceinline__
  void ProcessSegment(int segment_size,
                      const KeyT *keys_input,
                      KeyT *keys_output,
                      const ValueT *values_input,
                      ValueT *values_output,
                      TempStorage& temp_storage)
  {
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

    WarpMergeSortT warp_merge_sort(temp_storage);

    if (segment_size < 3)
    {
      ShortCircuit(warp_merge_sort.linear_tid,
                   segment_size,
                   keys_input,
                   keys_output,
                   values_input,
                   values_output,
                   binary_op);
    }
    else
    {
      KeyT keys[ITEMS_PER_THREAD];
      ValueT values[ITEMS_PER_THREAD];

      // For FP64 the difference is:
      // Lowest() -> -1.79769e+308 = 00...00b -> TwiddleIn -> -0 = 10...00b
      // LOWEST   -> -nan          = 11...11b -> TwiddleIn ->  0 = 00...00b

      using UnsignedBitsT = typename Traits<KeyT>::UnsignedBits;
      UnsignedBitsT default_key_bits = IS_DESCENDING ? Traits<KeyT>::LOWEST_KEY
                                                     : Traits<KeyT>::MAX_KEY;
      KeyT oob_default = reinterpret_cast<KeyT &>(default_key_bits);

      Load(warp_merge_sort.linear_tid,
           warp_merge_sort.member_mask,
           segment_size,
           keys_input,
           oob_default,
           keys,
           temp_storage.Alias().keys_shared);

      if (!KEYS_ONLY)
      {
        Load(warp_merge_sort.linear_tid,
             warp_merge_sort.member_mask,
             segment_size,
             values_input,
             values,
             temp_storage.Alias().items_shared);
      }

      warp_merge_sort.Sort(keys, values, binary_op, segment_size, oob_default);

      Store(warp_merge_sort.linear_tid,
            warp_merge_sort.member_mask,
            segment_size,
            keys_output,
            keys,
            temp_storage.Alias().keys_shared);

      if (!KEYS_ONLY)
      {
        Store(warp_merge_sort.linear_tid,
              warp_merge_sort.member_mask,
              segment_size,
              values_output,
              values,
              temp_storage.Alias().items_shared);
      }
    }
  }

private:
  template <typename CompareOpT>
  __device__ __forceinline__
  void ShortCircuit(
    unsigned int linear_tid,
    OffsetT segment_size,
    const KeyT *keys_input,
    KeyT *keys_output,
    const ValueT *values_input,
    ValueT *values_output,
    CompareOpT binary_op)
  {
    if (segment_size == 0)
    {
      return;
    }
    else if (segment_size == 1)
    {
      if (linear_tid == 0)
      {
        keys_output[0] = keys_input[0];

        if (!KEYS_ONLY)
        {
          values_output[0] = values_input[0];
        }
      }
    }
    else if (segment_size == 2)
    {
      if (linear_tid == 0)
      {
        KeyT lhs = keys_input[0];
        KeyT rhs = keys_input[1];

        if (binary_op(lhs, rhs))
        {
          keys_output[0] = lhs;
          keys_output[1] = rhs;

          if (!KEYS_ONLY)
          {
            if (values_output != values_input)
            {
              values_output[0] = values_input[0];
              values_output[1] = values_input[1];
            }
          }
        }
        else
        {
          keys_output[0] = rhs;
          keys_output[1] = lhs;

          if (!KEYS_ONLY)
          {
            // values_output might be an alias for values_input, so
            // we have to use registers here

            const ValueT lhs_val = values_input[0];
            const ValueT rhs_val = values_input[1];

            values_output[0] = rhs_val;
            values_output[1] = lhs_val;
          }
        }
      }
    }
  }

  template <typename T>
  __device__ __forceinline__
  void Load(unsigned int lane_id,
            unsigned int group_mask,
            OffsetT segment_size,
            const T *input,
            T oob_default,
            T (&keys)[ITEMS_PER_THREAD],
            T *cache)
  {
    for (int item = 0; item < ITEMS_PER_THREAD; item++)
    {
      keys[item] = oob_default;
    }

    Load(lane_id, group_mask, segment_size, input, keys, cache);
  }

  template <typename T>
  __device__ __forceinline__
  void Load(unsigned int lane_id,
            unsigned int group_mask,
            OffsetT segment_size,
            const T *input,
            T (&keys)[ITEMS_PER_THREAD],
            T *cache)
  {
    if (ITEMS_PER_THREAD > 10)
    {
      // COALESCED_LOAD

      #pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD; item++)
      {
        const int idx = THREADS_PER_SEGMENT * item + lane_id;

        if (idx < segment_size)
        {
          keys[item] = input[idx];
        }
      }

      // store in shared
      for (int item = 0; item < ITEMS_PER_THREAD; item++)
      {
        const int idx = THREADS_PER_SEGMENT * item + lane_id;
        cache[idx]    = keys[item];
      }
      __syncwarp(group_mask);

      // load blocked
      for (int item = 0; item < ITEMS_PER_THREAD; item++)
      {
        const int idx = ITEMS_PER_THREAD * lane_id + item;
        keys[item]    = cache[idx];
      }
      __syncwarp(group_mask);
    }
    else
    {
      // Naive load

      for (int item = 0; item < ITEMS_PER_THREAD; item++)
      {
        const unsigned int idx = lane_id * ITEMS_PER_THREAD + item;

        if (idx < segment_size)
        {
          keys[item] = input[idx];
        }
      }
    }
  }

  template <typename T>
  __device__ __forceinline__
  void Store(unsigned int lane_id,
             unsigned int group_mask,
             OffsetT segment_size,
             T *output,
             T (&keys)[ITEMS_PER_THREAD],
             T *cache)
  {
    if (ITEMS_PER_THREAD > 10)
    {
      // Coalesced store
      __syncwarp(group_mask);

      // load blocked
      for (int item = 0; item < ITEMS_PER_THREAD; item++)
      {
        const int idx = ITEMS_PER_THREAD * lane_id + item;
        cache[idx]    = keys[item];
      }
      __syncwarp(group_mask);

      // store in shared
      for (int item = 0; item < ITEMS_PER_THREAD; item++)
      {
        const int idx = THREADS_PER_SEGMENT * item + lane_id;
        keys[item]    = cache[idx];
      }

      for (int item = 0; item < ITEMS_PER_THREAD; item++)
      {
        const int idx = THREADS_PER_SEGMENT * item + lane_id;

        if (idx < segment_size)
        {
          output[idx] = keys[item];
        }
      }
    }
    else
    {
      // Naive store
      for (int item = 0; item < ITEMS_PER_THREAD; item++)
      {
        const int idx = lane_id * ITEMS_PER_THREAD + item;

        if (idx < segment_size)
        {
          output[idx] = keys[item];
        }
      }
    }
  }
};

CUB_NAMESPACE_END
