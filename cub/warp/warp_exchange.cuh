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


CUB_NAMESPACE_BEGIN


/**
 * \addtogroup WarpModule
 * @{
 */


template <typename InputT,
          int ITEMS_PER_THREAD,
          int LOGICAL_WARP_THREADS  = CUB_PTX_WARP_THREADS,
          int PTX_ARCH              = CUB_PTX_ARCH>
class WarpExchange
{
  static_assert(PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE,
                "LOGICAL_WARP_THREADS must be a power of two");

  constexpr static int ITEMS_PER_TILE =
    ITEMS_PER_THREAD * LOGICAL_WARP_THREADS + 1;

  constexpr static bool IS_ARCH_WARP = LOGICAL_WARP_THREADS ==
                                       CUB_WARP_THREADS(PTX_ARCH);

  constexpr static int LOG_SMEM_BANKS = CUB_LOG_SMEM_BANKS(PTX_ARCH);
  constexpr static int SMEM_BANKS     = 1 << LOG_SMEM_BANKS;

  // Insert padding if the number of items per thread is a power of two
  // and > 4 (otherwise we can typically use 128b loads)
  constexpr static int INSERT_PADDING = (ITEMS_PER_THREAD > 4) &&
                                        (PowerOfTwo<ITEMS_PER_THREAD>::VALUE);

  constexpr static int PADDING_ITEMS = INSERT_PADDING
                                     ? (ITEMS_PER_TILE >> LOG_SMEM_BANKS)
                                     : 0;

  union _TempStorage
  {
    InputT items_shared[ITEMS_PER_TILE + PADDING_ITEMS];
  }; // union TempStorage

  /// Shared storage reference
  _TempStorage &temp_storage;

public:

  /// \smemstorage{WarpExchange}
  struct TempStorage : Uninitialized<_TempStorage> {};

  const unsigned int lane_id;
  const unsigned int warp_id;
  const unsigned int member_mask;

  WarpExchange() = delete;

  explicit __device__ __forceinline__
  WarpExchange(TempStorage &temp_storage)
      : temp_storage(temp_storage.Alias())
      , lane_id(IS_ARCH_WARP ? LaneId() : (LaneId() % LOGICAL_WARP_THREADS))
      , warp_id(IS_ARCH_WARP ? 0 : (LaneId() / LOGICAL_WARP_THREADS))
      , member_mask(WarpMask<LOGICAL_WARP_THREADS>(warp_id))
  {

  }


  template <typename OutputT>
  __device__ __forceinline__ void BlockedToStriped(
    InputT          (&input_items)[ITEMS_PER_THREAD],      ///< [in] Items to exchange, converting between <em>blocked</em> and <em>striped</em> arrangements.
    OutputT         (&output_items)[ITEMS_PER_THREAD])     ///< [out] Items to exchange, converting between <em>blocked</em> and <em>striped</em> arrangements.
  {
    for (int item = 0; item < ITEMS_PER_THREAD; item++)
    {
      const int idx = ITEMS_PER_THREAD * lane_id + item;
      temp_storage.items_shared[idx] = input_items[item];
    }
    WARP_SYNC(member_mask);

    for (int item = 0; item < ITEMS_PER_THREAD; item++)
    {
      const int idx = LOGICAL_WARP_THREADS * item + lane_id;
      output_items[item] = temp_storage.items_shared[idx];
    }
  }

  template <typename OutputT>
  __device__ __forceinline__ void StripedToBlocked(
    InputT          input_items[ITEMS_PER_THREAD],      ///< [in] Items to exchange, converting between <em>blocked</em> and <em>striped</em> arrangements.
    OutputT         output_items[ITEMS_PER_THREAD])     ///< [out] Items to exchange, converting between <em>blocked</em> and <em>striped</em> arrangements.
  {
    for (int item = 0; item < ITEMS_PER_THREAD; item++)
    {
      const int idx = LOGICAL_WARP_THREADS * item + lane_id;
      temp_storage.items_shared[idx] = input_items[item];
    }
    WARP_SYNC(member_mask);

    for (int item = 0; item < ITEMS_PER_THREAD; item++)
    {
      const int idx = ITEMS_PER_THREAD * lane_id + item;
      output_items[item] = temp_storage.items_shared[idx];
    }
  }

  /**
   * \brief Exchanges valid data items annotated by rank
   *        into <em>striped</em> arrangement.
   *
   * \par
   * - \smemreuse
   *
   * \tparam OffsetT <b>[inferred]</b> Signed integer type for local offsets
   */
  template <typename OffsetT>
  __device__ __forceinline__ void ScatterToStriped(
    InputT          items[ITEMS_PER_THREAD],        ///< [in-out] Items to exchange
    OffsetT         ranks[ITEMS_PER_THREAD])        ///< [in] Corresponding scatter ranks
  {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
      if (INSERT_PADDING)
      {
        ranks[ITEM] = SHR_ADD(ranks[ITEM], LOG_SMEM_BANKS, ranks[ITEM]);
      }

      temp_storage.items_shared[ranks[ITEM]] = items[ITEM];
    }

    WARP_SYNC(member_mask);

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
      int item_offset = (ITEM * LOGICAL_WARP_THREADS) + lane_id;
      if (INSERT_PADDING)
      {
        item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
      }
      items[ITEM] = temp_storage.items_shared[item_offset];
    }
  }
};


/** @} */       // end group WarpModule

CUB_NAMESPACE_END
