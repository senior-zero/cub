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

/**
 * Parameterizable tuning policy type for AgentSelectIf
 */
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




/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/


/**
 * \brief AgentSelectIf implements a stateful abstraction of CUDA thread blocks for participating in device-wide selection
 *
 * Performs functor-based selection if SelectOpT functor type != NullType
 * Otherwise performs flag-based selection if FlagsInputIterator's value type != NullType
 * Otherwise performs discontinuity selection (keep unique)
 */
template <
  typename    AgentThreeWayPartitionPolicyT,  ///< Parameterized AgentSelectIfPolicy tuning policy type
  typename    InputIteratorT,                 ///< Random-access input iterator type for selection items
  typename    FlagsInputIteratorT,            ///< Random-access input iterator type for selections (NullType* if a selection functor or discontinuity flagging is to be used for selection)
  typename    SelectedOutputIteratorT,        ///< Random-access input iterator type for selection_flags items
  typename    SelectOp1T,                     ///< Selection operator type (NullType if selections or discontinuity flagging is to be used for selection)
  typename    SelectOp2T,                     ///< Selection operator type (NullType if selections or discontinuity flagging is to be used for selection)
  typename    OffsetT>                        ///< Signed integer type for global offsets
struct AgentThreeWayPartition
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  // The input value type
  typedef typename std::iterator_traits<InputIteratorT>::value_type InputT;

  // The output value type
  using OutputT = typename std::iterator_traits<SelectedOutputIteratorT>::value_type;

  // The flag value type
  typedef typename std::iterator_traits<FlagsInputIteratorT>::value_type FlagT;

  // Tile status descriptor interface type
  typedef cub::ScanTileState<OffsetT> ScanTileStateT;

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

  // Cache-modified Input iterator wrapper type (for applying cache modifier) for values
  typedef typename cub::If<cub::IsPointer<FlagsInputIteratorT>::VALUE,
    cub::CacheModifiedInputIterator<AgentThreeWayPartitionPolicyT::LOAD_MODIFIER, FlagT, OffsetT>,    // Wrap the native input pointer with CacheModifiedValuesInputIterator
    FlagsInputIteratorT>::Type                                                          // Directly use the supplied input iterator type
  WrappedFlagsInputIteratorT;

  // Parameterized BlockLoad type for input data
  typedef cub::BlockLoad<OutputT,
    BLOCK_THREADS,
    ITEMS_PER_THREAD,
    AgentThreeWayPartitionPolicyT::LOAD_ALGORITHM>
    BlockLoadT;

  // Parameterized BlockLoad type for flags
  typedef cub::BlockLoad<FlagT,
    BLOCK_THREADS,
    ITEMS_PER_THREAD,
    AgentThreeWayPartitionPolicyT::LOAD_ALGORITHM>
    BlockLoadFlags;

  // Parameterized BlockDiscontinuity type for items
  typedef cub::BlockDiscontinuity<OutputT, BLOCK_THREADS> BlockDiscontinuityT;

  // Parameterized BlockScan type
  typedef cub::BlockScan<OffsetT, BLOCK_THREADS, AgentThreeWayPartitionPolicyT::SCAN_ALGORITHM>
    BlockScanT;

  // Callback type for obtaining tile prefix during block scan
  typedef cub::TilePrefixCallbackOp<OffsetT, cub::Sum, ScanTileStateT>
    TilePrefixCallbackOpT;

  // Item exchange type
  typedef OutputT ItemExchangeT[TILE_ITEMS];

  // Shared memory type for this thread block
  union _TempStorage
  {
    struct ScanStorage
    {
      typename BlockScanT::TempStorage                scan;           // Smem needed for tile scanning
      typename TilePrefixCallbackOpT::TempStorage     prefix;         // Smem needed for cooperative prefix callback
      typename BlockDiscontinuityT::TempStorage       discontinuity;  // Smem needed for discontinuity detection
    } scan_storage;

    // Smem needed for loading items
    typename BlockLoadT::TempStorage load_items;

    // Smem needed for loading values
    typename BlockLoadFlags::TempStorage load_flags;

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
  SelectedOutputIteratorT              d_selected_out_1;   ///< Unique output items
  SelectedOutputIteratorT              d_selected_out_2;   ///< Unique output items
  WrappedFlagsInputIteratorT           d_flags_in;         ///< Input selection flags (if applicable)
  SelectOp1T                           select_op_1;        ///< Selection operator
  SelectOp2T                           select_op_2;        ///< Selection operator
  OffsetT                              num_items;          ///< Total number of input items


  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  // Constructor
  __device__ __forceinline__
  AgentThreeWayPartition(
    TempStorage                 &temp_storage,      ///< Reference to temp_storage
    InputIteratorT              d_in,               ///< Input data
    FlagsInputIteratorT         d_flags_in,         ///< Input selection flags (if applicable)
    SelectedOutputIteratorT     d_selected_out_1,     ///< Output data
    SelectedOutputIteratorT     d_selected_out_2,     ///< Output data
    SelectOp1T                  select_op_1,          ///< Selection operator
    SelectOp2T                  select_op_2,          ///< Selection operator
    OffsetT                     num_items)          ///< Total number of input items
    :
    temp_storage(temp_storage.Alias()),
    d_in(d_in),
    d_flags_in(d_flags_in),
    d_selected_out_1(d_selected_out_1),
    d_selected_out_2(d_selected_out_2),
    select_op_1(select_op_1),
    select_op_2(select_op_2),
    num_items(num_items)
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
    OutputT                       (&items)[ITEMS_PER_THREAD],
    OffsetT                       (&large_items_selection_flags)[ITEMS_PER_THREAD],
    OffsetT                       (&small_items_selection_flags)[ITEMS_PER_THREAD])
  {
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      // Out-of-bounds items are selection_flags
      large_items_selection_flags[ITEM] = 1;
      small_items_selection_flags[ITEM] = 1;

      if (!IS_LAST_TILE || (OffsetT(threadIdx.x * ITEMS_PER_THREAD) + ITEM < num_tile_items))
      {
        large_items_selection_flags[ITEM] = select_op_1(items[ITEM]);
        small_items_selection_flags[ITEM] = large_items_selection_flags[ITEM] ? 0 : select_op_2(items[ITEM]);
      }
    }
  }

  /**
   * Scatter flagged items to output offsets (specialized for two-phase scattering)
   */
  template <bool IS_LAST_TILE>
  __device__ __forceinline__ void Scatter(
    OutputT         (&items)[ITEMS_PER_THREAD],
    OffsetT         (&large_items_selection_flags)[ITEMS_PER_THREAD],
    OffsetT         (&large_items_selection_indices)[ITEMS_PER_THREAD],
    OffsetT         (&small_items_selection_flags)[ITEMS_PER_THREAD],
    OffsetT         (&small_items_selection_indices)[ITEMS_PER_THREAD],
    int             num_tile_items,                               ///< Number of valid items in this tile
    int             num_large_tile_selections,                    ///< Number of selections in this tile
    int             num_small_tile_selections,                    ///< Number of selections in this tile
    OffsetT         num_large_selections_prefix,                  ///< Total number of selections prior to this tile
    OffsetT         num_small_selections_prefix,                  ///< Total number of selections prior to this tile
    OffsetT         num_rejected_prefix)
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

        if (large_items_selection_flags[ITEM])
        {
          local_scatter_offset = large_items_selection_indices[ITEM] - num_large_selections_prefix;
        }
        else if (small_items_selection_flags[ITEM])
        {
          local_scatter_offset = large_item_end +
                                 small_items_selection_indices[ITEM] -
                                 num_small_selections_prefix;
        }
        else
        {
          // Medium item
          int local_selection_idx = (large_items_selection_indices[ITEM] - num_large_selections_prefix)
                                    + (small_items_selection_indices[ITEM] - num_small_selections_prefix);
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
        OutputT item = temp_storage.raw_exchange.Alias()[item_idx];

        if (item_idx < large_item_end)
        {
          d_selected_out_1[num_large_selections_prefix + item_idx] = item;
        }
        else if (item_idx < small_item_end)
        {
          d_selected_out_2[num_small_selections_prefix + item_idx - large_item_end] = item;
        }
        else
        {
          int rejection_idx = item_idx - small_item_end;
          d_selected_out_1[num_items - num_rejected_prefix - rejection_idx - 1] = item;
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
    ScanTileStateT&     large_tile_state,   ///< Global tile state descriptor
    ScanTileStateT&     small_tile_state,   ///< Global tile state descriptor
    OffsetT &large_items,
    OffsetT &small_items)
  {
    OutputT     items[ITEMS_PER_THREAD];

    OffsetT     large_items_selection_flags[ITEMS_PER_THREAD];
    OffsetT     large_items_selection_indices[ITEMS_PER_THREAD];

    OffsetT     small_items_selection_flags[ITEMS_PER_THREAD];
    OffsetT     small_items_selection_indices[ITEMS_PER_THREAD];

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
      large_items_selection_flags,
      small_items_selection_flags);

    __syncthreads();

    // Exclusive scan of selection_flags
    BlockScanT(temp_storage.scan_storage.scan)
      .ExclusiveSum(large_items_selection_flags,
                    large_items_selection_indices,
                    large_items);

    if (threadIdx.x == 0)
    {
      // Update tile status if this is not the last tile
      if (!IS_LAST_TILE)
      {
        large_tile_state.SetInclusive(0, large_items);
      }
    }

    __syncthreads();

    // Exclusive scan of selection_flags
    BlockScanT(temp_storage.scan_storage.scan)
      .ExclusiveSum(small_items_selection_flags,
                    small_items_selection_indices,
                    small_items);

    if (threadIdx.x == 0)
    {
      // Update tile status if this is not the last tile
      if (!IS_LAST_TILE)
      {
        small_tile_state.SetInclusive(0, small_items);
      }
    }

    // Discount any out-of-bounds selections
    if (IS_LAST_TILE)
    {
      large_items -= (TILE_ITEMS - num_tile_items);
      small_items -= (TILE_ITEMS - num_tile_items);
    }

    // Scatter flagged items
    Scatter<IS_LAST_TILE>(
      items,
      large_items_selection_flags,
      large_items_selection_indices,
      small_items_selection_flags,
      small_items_selection_indices,
      num_tile_items,
      large_items,
      small_items,
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
    ScanTileStateT&     large_tile_state,   ///< Global tile state descriptor
    ScanTileStateT&     small_tile_state,

    OffsetT &num_large_items_selections,
    OffsetT &num_small_items_selections)
  {
    OutputT     items[ITEMS_PER_THREAD];

    OffsetT     large_items_selection_flags[ITEMS_PER_THREAD];
    OffsetT     large_items_selection_indices[ITEMS_PER_THREAD];

    OffsetT     small_items_selection_flags[ITEMS_PER_THREAD];
    OffsetT     small_items_selection_indices[ITEMS_PER_THREAD];

    // Load items
    if (IS_LAST_TILE)
      BlockLoadT(temp_storage.load_items).Load(d_in + tile_offset, items, num_tile_items);
    else
      BlockLoadT(temp_storage.load_items).Load(d_in + tile_offset, items);

    // Initialize selection_flags
    Initialize<IS_LAST_TILE>(
      num_tile_items,
      items,
      large_items_selection_flags,
      small_items_selection_flags);

    __syncthreads();

    // Exclusive scan of values and selection_flags
    TilePrefixCallbackOpT large_prefix_op(large_tile_state, temp_storage.scan_storage.prefix, cub::Sum(), tile_idx);
    BlockScanT(temp_storage.scan_storage.scan).ExclusiveSum(large_items_selection_flags, large_items_selection_indices, large_prefix_op);

    num_large_items_selections                  = large_prefix_op.GetInclusivePrefix();
    OffsetT num_large_items_in_tile_selections  = large_prefix_op.GetBlockAggregate();
    OffsetT num_large_items_selections_prefix   = large_prefix_op.GetExclusivePrefix();

    __syncthreads();

    TilePrefixCallbackOpT small_prefix_op(small_tile_state, temp_storage.scan_storage.prefix, cub::Sum(), tile_idx);
    BlockScanT(temp_storage.scan_storage.scan).ExclusiveSum(small_items_selection_flags, small_items_selection_indices, small_prefix_op);

    num_small_items_selections                  = small_prefix_op.GetInclusivePrefix();
    OffsetT num_small_items_in_tile_selections  = small_prefix_op.GetBlockAggregate();
    OffsetT num_small_items_selections_prefix   = small_prefix_op.GetExclusivePrefix();

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
      large_items_selection_flags,
      large_items_selection_indices,
      small_items_selection_flags,
      small_items_selection_indices,
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
    ScanTileStateT&     large_tile_state,   ///< Global tile state descriptor
    ScanTileStateT&     small_tile_state,

    OffsetT &large_items,
    OffsetT &small_items)
  {
    if (tile_idx == 0)
    {
      ConsumeFirstTile<IS_LAST_TILE>(num_tile_items,
                                     tile_offset,
                                     large_tile_state,
                                     small_tile_state,
                                     large_items,
                                     small_items);
    }
    else
    {
      ConsumeSubsequentTile<IS_LAST_TILE>(num_tile_items,
                                          tile_idx,
                                          tile_offset,
                                          large_tile_state,
                                          small_tile_state,
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
    ScanTileStateT&         large_tile_state,   ///< Global tile state descriptor
    ScanTileStateT&         small_tile_state,   ///< Global tile state descriptor
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
                         large_tile_state,
                         small_tile_state,
                         num_large_selections,
                         num_small_selections);
    }
    else
    {
      // The last tile (possibly partially-full)
      OffsetT num_remaining   = num_items - tile_offset;

      ConsumeTile<true>(num_remaining,
                        tile_idx,
                        tile_offset,
                        large_tile_state,
                        small_tile_state,
                        num_large_selections,
                        num_small_selections);

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
