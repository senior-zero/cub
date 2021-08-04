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
  int         ITEMS_PER_THREAD,
  int         LOGICAL_WARP_THREADS    = CUB_PTX_WARP_THREADS,
  typename    ValueT                  = NullType,
  int         PTX_ARCH                = CUB_PTX_ARCH>
class WarpMergeSort
    : public BlockMergeSortStrategy<
        KeyT,
        ValueT,
        LOGICAL_WARP_THREADS,
        ITEMS_PER_THREAD,
        WarpMergeSort<KeyT, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, ValueT, PTX_ARCH>>
{
private:
  constexpr static bool IS_ARCH_WARP = LOGICAL_WARP_THREADS == CUB_WARP_THREADS(PTX_ARCH);
  constexpr static bool KEYS_ONLY = cub::Equals<ValueT, NullType>::VALUE;
  constexpr static int TILE_SIZE = ITEMS_PER_THREAD * LOGICAL_WARP_THREADS;

  using BlockMergeSortStrategyT = BlockMergeSortStrategy<KeyT,
                                                         ValueT,
                                                         LOGICAL_WARP_THREADS,
                                                         ITEMS_PER_THREAD,
                                                         WarpMergeSort>;

public:
  const unsigned int warp_id;
  const unsigned int member_mask;

  WarpMergeSort() = delete;

  __device__ __forceinline__
  WarpMergeSort(typename BlockMergeSortStrategyT::TempStorage &temp_storage)
      : BlockMergeSortStrategyT(temp_storage,
                                IS_ARCH_WARP
                                  ? LaneId()
                                  : (LaneId() % LOGICAL_WARP_THREADS))
      , warp_id(IS_ARCH_WARP ? 0 : (LaneId() / LOGICAL_WARP_THREADS))
      , member_mask(WarpMask<LOGICAL_WARP_THREADS, PTX_ARCH>(warp_id))
  {
  }

private:
  __device__ __forceinline__ void SyncImplementation() const
  {
    WARP_SYNC(member_mask);
  }

  friend BlockMergeSortStrategyT;
};

/** @} */       // end group WarpModule

CUB_NAMESPACE_END
