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

#include <iterator>

#include "single_pass_scan_operators.cuh"
#include "../block/block_load.cuh"
#include "../block/block_store.cuh"
#include "../block/block_scan.cuh"
#include "../block/block_exchange.cuh"
#include "../block/block_discontinuity.cuh"
#include "../config.cuh"
#include "../grid/grid_queue.cuh"
#include "../iterator/cache_modified_input_iterator.cuh"

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

template <
    int                         _BLOCK_THREADS,                 ///< Threads per thread block
    int                         _ITEMS_PER_THREAD,              ///< Items per thread (per tile of input)
    BlockLoadAlgorithm          _LOAD_ALGORITHM,                ///< The BlockLoad algorithm to use
    CacheLoadModifier           _LOAD_MODIFIER,                 ///< Cache load modifier for reading input elements
    BlockScanAlgorithm          _SCAN_ALGORITHM>                ///< The BlockScan algorithm to use
struct AgentThreeWayPartitionPolicy
{
    enum
    {
        BLOCK_THREADS           = _BLOCK_THREADS,               ///< Threads per thread block
        ITEMS_PER_THREAD        = _ITEMS_PER_THREAD,            ///< Items per thread (per tile of input)
    };

    static const BlockLoadAlgorithm     LOAD_ALGORITHM          = _LOAD_ALGORITHM;      ///< The BlockLoad algorithm to use
    static const CacheLoadModifier      LOAD_MODIFIER           = _LOAD_MODIFIER;       ///< Cache load modifier for reading input elements
    static const BlockScanAlgorithm     SCAN_ALGORITHM          = _SCAN_ALGORITHM;      ///< The BlockScan algorithm to use
};




template <
  typename    AgentThreeWayPartitionPolicyT,
  typename    InputIteratorT,
  typename    FirstOutputIteratorT,
  typename    SecondOutputIteratorT,
  typename    UnselectedOutputIteratorT,
  typename    SelectFirstPartOp,
  typename    SelectSecondPartOp,
  typename    OffsetT>
struct AgentThreeWayPartition
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  // The input value type
  typedef typename std::iterator_traits<InputIteratorT>::value_type InputT;

  // Tile status descriptor interface type
  typedef cub::ScanTileState<cub::CubVector<OffsetT, 2>> ScanTileStateT;

  // Constants
  enum
  {
    USE_SELECT_OP,
    USE_SELECT_FLAGS,
    USE_DISCONTINUITY,

    BLOCK_THREADS           = AgentThreeWayPartitionPolicyT::BLOCK_THREADS,
    ITEMS_PER_THREAD        = AgentThreeWayPartitionPolicyT::ITEMS_PER_THREAD,
    TILE_ITEMS              = BLOCK_THREADS * ITEMS_PER_THREAD,
    TWO_PHASE_SCATTER       = (ITEMS_PER_THREAD > 1),

    SELECT_METHOD           = USE_SELECT_OP
  };

  // Cache-modified Input iterator wrapper type (for applying cache modifier) for items
  typedef typename cub::If<cub::IsPointer<InputIteratorT>::VALUE,
    cub::CacheModifiedInputIterator<AgentThreeWayPartitionPolicyT::LOAD_MODIFIER, InputT, OffsetT>,        // Wrap the native input pointer with CacheModifiedValuesInputIterator
    InputIteratorT>::Type                                                               // Directly use the supplied input iterator type
  WrappedInputIteratorT;

  // Parameterized BlockLoad type for input data
  typedef cub::BlockLoad<InputT,
    BLOCK_THREADS,
    ITEMS_PER_THREAD,
    AgentThreeWayPartitionPolicyT::LOAD_ALGORITHM>
    BlockLoadT;

  // Parameterized BlockScan type
  using BlockScanT =
    cub::BlockScan<cub::CubVector<OffsetT, 2>,
                   BLOCK_THREADS,
                   AgentThreeWayPartitionPolicyT::SCAN_ALGORITHM>;

  // Callback type for obtaining tile prefix during block scan
  using TilePrefixCallbackOpT = cub::
    TilePrefixCallbackOp<cub::CubVector<OffsetT, 2>, cub::Sum, ScanTileStateT>;

  // Item exchange type
  typedef InputT ItemExchangeT[TILE_ITEMS];

  // Shared memory type for this thread block
  union _TempStorage
  {
    struct ScanStorage
    {
      typename BlockScanT::TempStorage                scan;           // Smem needed for tile scanning
      typename TilePrefixCallbackOpT::TempStorage     prefix;         // Smem needed for cooperative prefix callback
    } scan_storage;

    // Smem needed for loading items
    typename BlockLoadT::TempStorage load_items;

    // Smem needed for compacting items (allows non POD items in this union)
    cub::Uninitialized<ItemExchangeT> raw_exchange;
  };

  // Alias wrapper allowing storage to be unioned
  struct TempStorage : cub::Uninitialized<_TempStorage> {};


  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  _TempStorage&                        temp_storage;       ///< Reference to temp_storage
  WrappedInputIteratorT                d_in;               ///< Input items
  FirstOutputIteratorT                 d_first_part_out;
  SecondOutputIteratorT                d_second_part_out;
  UnselectedOutputIteratorT            d_unselected_out;
  SelectFirstPartOp                    select_first_part_op;
  SelectSecondPartOp                   select_second_part_op;
  OffsetT                              num_items;          ///< Total number of input items


  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  // Constructor
  __device__ __forceinline__
  AgentThreeWayPartition(TempStorage &temp_storage,
                         InputIteratorT d_in,
                         FirstOutputIteratorT d_first_part_out,
                         SecondOutputIteratorT d_second_part_out,
                         UnselectedOutputIteratorT d_unselected_out,
                         SelectFirstPartOp select_first_part_op,
                         SelectSecondPartOp select_second_part_op,
                         OffsetT num_items)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_first_part_out(d_first_part_out)
      , d_second_part_out(d_second_part_out)
      , d_unselected_out(d_unselected_out)
      , select_first_part_op(select_first_part_op)
      , select_second_part_op(select_second_part_op)
      , num_items(num_items)
  {}

  //---------------------------------------------------------------------
  // Utility methods for initializing the selections
  //---------------------------------------------------------------------

  /**
   * Initialize selections (specialized for selection operator)
   */
  template <bool IS_LAST_TILE>
  __device__ __forceinline__ void Initialize(
    OffsetT                       num_tile_items,
    InputT                        (&items)[ITEMS_PER_THREAD],
    cub::CubVector<OffsetT, 2>    (&items_selection_flags)[ITEMS_PER_THREAD])
  {
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      // Out-of-bounds items are selection_flags
      items_selection_flags[ITEM].x = 1;
      items_selection_flags[ITEM].y = 1;

      if (!IS_LAST_TILE || (OffsetT(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items))
      {
        items_selection_flags[ITEM].x = select_first_part_op(items[ITEM]);
        items_selection_flags[ITEM].y = items_selection_flags[ITEM].x
                                      ? 0
                                      : select_second_part_op(items[ITEM]);
      }
    }
  }

  /**
   * Scatter flagged items to output offsets (specialized for two-phase scattering)
   */
  template <bool IS_LAST_TILE>
  __device__ __forceinline__ void Scatter(
    InputT                    (&items)[ITEMS_PER_THREAD],
    cub::CubVector<OffsetT,2> (&items_selection_flags)[ITEMS_PER_THREAD],
    cub::CubVector<OffsetT,2> (&items_selection_indices)[ITEMS_PER_THREAD],
    int                       num_tile_items,                               ///< Number of valid items in this tile
    int                       num_large_tile_selections,                    ///< Number of selections in this tile
    int                       num_small_tile_selections,                    ///< Number of selections in this tile
    OffsetT                   num_large_selections_prefix,                  ///< Total number of selections prior to this tile
    OffsetT                   num_small_selections_prefix,                  ///< Total number of selections prior to this tile
    OffsetT                   num_rejected_prefix)
  {
    __syncthreads();

    int large_item_end = num_large_tile_selections;
    int small_item_end = large_item_end + num_small_tile_selections;

    // Scatter items to shared memory (rejections first)
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      int item_idx = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;

      if (!IS_LAST_TILE || (item_idx < num_tile_items))
      {
        int local_scatter_offset = 0;

        if (items_selection_flags[ITEM].x)
        {
          local_scatter_offset = items_selection_indices[ITEM].x - num_large_selections_prefix;
        }
        else if (items_selection_flags[ITEM].y)
        {
          local_scatter_offset = large_item_end +
                                 items_selection_indices[ITEM].y -
                                 num_small_selections_prefix;
        }
        else
        {
          // Medium item
          int local_selection_idx = (items_selection_indices[ITEM].x - num_large_selections_prefix)
                                  + (items_selection_indices[ITEM].y - num_small_selections_prefix);
          local_scatter_offset = small_item_end + item_idx - local_selection_idx;
        }

        temp_storage.raw_exchange.Alias()[local_scatter_offset] = items[ITEM];
      }
    }

    __syncthreads();

    // Gather items from shared memory and scatter to global
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      int item_idx = (ITEM * BLOCK_THREADS) + threadIdx.x;

      if (!IS_LAST_TILE || (item_idx < num_tile_items))
      {
        InputT item = temp_storage.raw_exchange.Alias()[item_idx];

        if (item_idx < large_item_end)
        {
          d_first_part_out[num_large_selections_prefix + item_idx] = item;
        }
        else if (item_idx < small_item_end)
        {
          d_second_part_out[num_small_selections_prefix + item_idx - large_item_end] = item;
        }
        else
        {
          int rejection_idx = item_idx - small_item_end;
          // d_first_part_out[num_items - num_rejected_prefix - rejection_idx - 1] = item;
          d_unselected_out[num_rejected_prefix + rejection_idx] = item;
        }
      }
    }
  }


  //---------------------------------------------------------------------
  // Cooperatively scan a device-wide sequence of tiles with other CTAs
  //---------------------------------------------------------------------


  /**
   * Process first tile of input (dynamic chained scan).  Returns the running count of selections (including this tile)
   */
  template <bool IS_LAST_TILE>
  __device__ __forceinline__ void ConsumeFirstTile(
    int                 num_tile_items,     ///< Number of input items comprising this tile
    OffsetT             tile_offset,        ///< Tile offset
    ScanTileStateT&     tile_state,         ///< Global tile state descriptor
    OffsetT&            large_items_inclusive,
    OffsetT&            small_items_inclusive)
  {
    InputT      items[ITEMS_PER_THREAD];

    cub::CubVector<OffsetT,2> items_selection_flags[ITEMS_PER_THREAD];
    cub::CubVector<OffsetT,2> items_selection_indices[ITEMS_PER_THREAD];

    // Load items
    if (IS_LAST_TILE)
    {
      BlockLoadT(temp_storage.load_items).Load(d_in + tile_offset, items, num_tile_items);
    }
    else
    {
      BlockLoadT(temp_storage.load_items).Load(d_in + tile_offset, items);
    }

    // Initialize selection_flags
    Initialize<IS_LAST_TILE>(
      num_tile_items,
      items,
      items_selection_flags);

    __syncthreads();

    cub::CubVector<OffsetT, 2> block_aggregate;

    // Exclusive scan of selection_flags
    BlockScanT(temp_storage.scan_storage.scan)
      .ExclusiveSum(items_selection_flags,
                    items_selection_indices,
                    block_aggregate);

    if (threadIdx.x == 0)
    {
      // Update tile status if this is not the last tile
      if (!IS_LAST_TILE)
      {
        tile_state.SetInclusive(0, block_aggregate);
      }
    }

    __syncthreads();

    large_items_inclusive = block_aggregate.x;
    small_items_inclusive = block_aggregate.y;

    // Discount any out-of-bounds selections
    if (IS_LAST_TILE)
    {
      large_items_inclusive -= (TILE_ITEMS - num_tile_items);
      small_items_inclusive -= (TILE_ITEMS - num_tile_items);
    }

    // Scatter flagged items
    Scatter<IS_LAST_TILE>(
      items,
      items_selection_flags,
      items_selection_indices,
      num_tile_items,
      large_items_inclusive,
      small_items_inclusive,
      // all the prefixes equal to 0 because it's the first tile
      0, 0, 0);
  }


  /**
   * Process subsequent tile of input (dynamic chained scan).  Returns the running count of selections (including this tile)
   */
  template <bool IS_LAST_TILE>
  __device__ __forceinline__ void ConsumeSubsequentTile(
    int                 num_tile_items,     ///< Number of input items comprising this tile
    int                 tile_idx,           ///< Tile index
    OffsetT             tile_offset,        ///< Tile offset
    ScanTileStateT&     tile_state,         ///< Global tile state descriptor
    OffsetT&            num_large_items_selections,
    OffsetT&            num_small_items_selections)
  {
    InputT      items[ITEMS_PER_THREAD];

    cub::CubVector<OffsetT, 2> items_selection_flags[ITEMS_PER_THREAD];
    cub::CubVector<OffsetT, 2> items_selection_indices[ITEMS_PER_THREAD];

    // Load items
    if (IS_LAST_TILE)
    {
      BlockLoadT(temp_storage.load_items).Load(d_in + tile_offset, items, num_tile_items);
    }
    else
    {
      BlockLoadT(temp_storage.load_items).Load(d_in + tile_offset, items);
    }

    // Initialize selection_flags
    Initialize<IS_LAST_TILE>(
      num_tile_items,
      items,
      items_selection_flags);

    __syncthreads();

    // Exclusive scan of values and selection_flags
    TilePrefixCallbackOpT prefix_op(tile_state, temp_storage.scan_storage.prefix, cub::Sum(), tile_idx);
    BlockScanT(temp_storage.scan_storage.scan).ExclusiveSum(items_selection_flags, items_selection_indices, prefix_op);

    num_large_items_selections                  = prefix_op.GetInclusivePrefix().x;
    OffsetT num_large_items_in_tile_selections  = prefix_op.GetBlockAggregate().x;
    OffsetT num_large_items_selections_prefix   = prefix_op.GetExclusivePrefix().x;

    num_small_items_selections                  = prefix_op.GetInclusivePrefix().y;
    OffsetT num_small_items_in_tile_selections  = prefix_op.GetBlockAggregate().y;
    OffsetT num_small_items_selections_prefix   = prefix_op.GetExclusivePrefix().y;

    OffsetT num_rejected_prefix = (tile_idx * TILE_ITEMS)
                                  - num_large_items_selections_prefix
                                  - num_small_items_selections_prefix;

    // Discount any out-of-bounds selections. There are exactly
    // TILE_ITEMS - num_tile_items elements like that because we
    // marked them as selected in Initialize method.
    if (IS_LAST_TILE)
    {
      int num_discount                    = TILE_ITEMS - num_tile_items;
      num_large_items_selections         -= num_discount;
      num_large_items_in_tile_selections -= num_discount;
      num_small_items_selections         -= num_discount;
      num_small_items_in_tile_selections -= num_discount;
    }

    // Scatter flagged items
    Scatter<IS_LAST_TILE>(
      items,
      items_selection_flags,
      items_selection_indices,
      num_tile_items,
      num_large_items_in_tile_selections,
      num_small_items_in_tile_selections,
      num_large_items_selections_prefix,
      num_small_items_selections_prefix,
      num_rejected_prefix);
  }


  /**
   * Process a tile of input
   */
  template <bool IS_LAST_TILE>
  __device__ __forceinline__ void ConsumeTile(
    int                 num_tile_items,     ///< Number of input items comprising this tile
    int                 tile_idx,           ///< Tile index
    OffsetT             tile_offset,        ///< Tile offset
    ScanTileStateT&     tile_state,          ///< Global tile state descriptor
    OffsetT &large_items,
    OffsetT &small_items)
  {
    if (tile_idx == 0)
    {
      ConsumeFirstTile<IS_LAST_TILE>(num_tile_items,
                                     tile_offset,
                                     tile_state,
                                     large_items,
                                     small_items);
    }
    else
    {
      ConsumeSubsequentTile<IS_LAST_TILE>(num_tile_items,
                                          tile_idx,
                                          tile_offset,
                                          tile_state,
                                          large_items,
                                          small_items);
    }
  }


  /**
   * Scan tiles of items as part of a dynamic chained scan
   */
  template <typename NumSelectedIteratorT>        ///< Output iterator type for recording number of items selection_flags
  __device__ __forceinline__ void ConsumeRange(
    int                     num_tiles,          ///< Total number of input tiles
    ScanTileStateT&         tile_state,         ///< Global tile state descriptor
    NumSelectedIteratorT    d_num_selected_out) ///< Output total number selection_flags
  {
    // Blocks are launched in increasing order, so just assign one tile per block
    int     tile_idx    = static_cast<int>((blockIdx.x * gridDim.y) + blockIdx.y);  // Current tile index
    OffsetT tile_offset = tile_idx * TILE_ITEMS;                                    // Global offset for the current tile

    OffsetT num_large_selections;
    OffsetT num_small_selections;

    if (tile_idx < num_tiles - 1)
    {
      // Not the last tile (full)
      ConsumeTile<false>(TILE_ITEMS,
                         tile_idx,
                         tile_offset,
                         tile_state,
                         num_large_selections,
                         num_small_selections);
    }
    else
    {
      // The last tile (possibly partially-full)
      OffsetT num_remaining  = num_items - tile_offset;

      ConsumeTile<true>(num_remaining,
                        tile_idx,
                        tile_offset,
                        tile_state,
                        num_large_selections,
                        num_small_selections);

      // TODO Rename large/small to first/second
      if (threadIdx.x == 0)
      {
        // Output the total number of items selection_flags
        d_num_selected_out[0] = num_large_selections;
        d_num_selected_out[1] = num_small_selections;
      }
    }
  }

};


CUB_NAMESPACE_END
