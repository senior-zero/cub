/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION. All rights reserved.
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

/**
 * \file
 * cub::AgentBatchMemcpy implements device-wide copying of a batch of device-accessible source-buffers to
 * device-accessible destination-buffers.
 */

#pragma once

#include "../block/block_exchange.cuh"
#include "../block/block_load.cuh"
#include "../block/block_run_length_decode.cuh"
#include "../block/block_scan.cuh"
#include "../block/block_store.cuh"
#include "../util_ptx.cuh"
#include "../util_type.cuh"
#include "single_pass_scan_operators.cuh"
#include <cstdint>

CUB_NAMESPACE_BEGIN

/**
 * @brief A helper class that allows threads to maintain multiple counters, where the counter that shall be incremented
 * can be addressed dynamically without incurring register spillage.
 *
 * @tparam NUM_ITEMS The number of counters to allocate
 * @tparam MAX_ITEM_VALUE The maximum count that must be supported.
 * @tparam PREFER_POW2_BITS Whether the number of bits to dedicate to each counter should be a power-of-two. If enabled,
 * this allows replacing integer multiplication with a bit-shift in exchange for higher register pressure.
 * @tparam BackingUnitT The data type that is used to provide the bits of all the counters that shall be allocated.
 */
template <uint32_t NUM_ITEMS, uint32_t MAX_ITEM_VALUE, bool PREFER_POW2_BITS, typename BackingUnitT = uint32_t>
class BitPackedCounter
{
private:
  /**
   * @brief Returns a bit-mask that has the \p num_bits least-significant bits being set.
   *
   * @param num_bits The number of least-significant bits to set to '1'
   */
  static __forceinline__ __host__ __device__ uint32_t GET_BIT_MASK(uint32_t num_bits)
  {
    return (num_bits >= (8U * sizeof(uint32_t))) ? Traits<uint32_t>::MAX_KEY : (0x01U << num_bits) - 1;
  }

  /**
   * @brief Returns a 64-bit unsigned integer that has the \p num_bits least-significant bits being set.
   *
   * @param num_bits The number of least-significant bits to set to '1'
   */
  static __forceinline__ __host__ __device__ uint64_t GET_BIT_MASKLL(uint32_t num_bits)
  {
    return (num_bits >= (8U * sizeof(uint64_t))) ? Traits<uint64_t>::MAX_KEY : (0x01ULL << num_bits) - 1ULL;
  }

  enum : uint32_t
  {
    /// The minimum number of bits required to represent all values from [0, MAX_ITEM_VALUE]
    MIN_BITS_PER_ITEM = MAX_ITEM_VALUE == 0U ? 1U : cub::Log2<(MAX_ITEM_VALUE + 1)>::VALUE,

    /// The number of btis allocated for each item. For pre-Volta, we prefer a power-of-2 here to have the compiler
    /// replace costly integer multiplication with bit-shifting.
    BITS_PER_ITEM = PREFER_POW2_BITS ? (0x01ULL << (cub::Log2<MIN_BITS_PER_ITEM>::VALUE)) : MIN_BITS_PER_ITEM,

    /// The number of bits that each backing data type can store
    NUM_BITS_PER_UNIT = sizeof(BackingUnitT) * 8,

    /// The number of items that each backing data type can store
    ITEMS_PER_UNIT = NUM_BITS_PER_UNIT / NUM_ITEMS,

    /// The number of bits the backing data type is actually making use of
    USED_BITS_PER_UNIT = ITEMS_PER_UNIT * BITS_PER_ITEM,

    /// The number of backing data types required to store the given number of items
    NUM_TOTAL_UNITS = CUB_QUOTIENT_CEILING(NUM_ITEMS, ITEMS_PER_UNIT),

    /// This is the net number of bit-storage provided by each unit (remainder bits are unused)
    UNIT_MASK = (USED_BITS_PER_UNIT >= (8U * sizeof(uint32_t))) ? 0xFFFFFFFF : (0x01U << USED_BITS_PER_UNIT) - 1,
    /// This is the bit-mask for each item
    ITEM_MASK = (BITS_PER_ITEM >= (8U * sizeof(uint32_t))) ? 0xFFFFFFFF : (0x01U << BITS_PER_ITEM) - 1
  };

  //------------------------------------------------------------------------------
  // ACCESSORS
  //------------------------------------------------------------------------------
public:
  __host__ __device__ __forceinline__ uint32_t Get(uint32_t index) const
  {
    int32_t target_offset = index * BITS_PER_ITEM;
    uint32_t val          = 0;

#pragma unroll
    for (int32_t i = 0; i < NUM_TOTAL_UNITS; ++i)
    {
      // In case the bit-offset of the counter at <index> is larger than the bit range of the current unit,
      // the bit_shift amount will be larger than the bits provided by this unit.
      // If the bit-offset of the counter at <index> is smaller than the bit range of the current unit,
      // the computed bit-offset will be negative, which, once casted to an unsigned type will be larger than the bits
      // provided by this unit.
      uint32_t bit_shift = static_cast<uint32_t>(target_offset - i * USED_BITS_PER_UNIT);
      val |= (data[i] >> bit_shift) & ITEM_MASK;
    }
    return val;
  }

  __host__ __device__ __forceinline__ void Add(uint32_t index, uint32_t value)
  {
    int32_t target_offset = index * BITS_PER_ITEM;

#pragma unroll
    for (int32_t i = 0; i < NUM_TOTAL_UNITS; ++i)
    {
      // In case the bit-offset of the counter at <index> is larger than the bit range of the current unit,
      // the bit_shift amount will be larger than the bits provided by this unit.
      // If the bit-offset of the counter at <index> is smaller than the bit range of the current unit,
      // the computed bit-offset will be negative, which, once casted to an unsigned type will be larger than the bits
      // provided by this unit.
      uint32_t bit_shift = static_cast<uint32_t>(target_offset - i * USED_BITS_PER_UNIT);
      data[i] += (value << bit_shift) & UNIT_MASK;
    }
  }

  __host__ __device__ __forceinline__ BitPackedCounter operator+(const BitPackedCounter &rhs) const
  {
    BitPackedCounter result;
    for (int32_t i = 0; i < NUM_TOTAL_UNITS; ++i)
    {
      result.data[i] = data[i] + rhs.data[i];
    }
    return result;
  }

  //------------------------------------------------------------------------------
  // CONSTRUCTORS
  //------------------------------------------------------------------------------
  __host__ __device__ __forceinline__ BitPackedCounter()
  {
#pragma unroll
    for (int32_t i = 0; i < NUM_TOTAL_UNITS; ++i)
    {
      data[i] = 0;
    }
  }

  //------------------------------------------------------------------------------
  // MEMBER VARIABLES
  //------------------------------------------------------------------------------
private:
  BackingUnitT data[NUM_TOTAL_UNITS];
};

/**
 * Parameterizable tuning policy type for AgentBatchMemcpy
 */
template <uint32_t _BLOCK_THREADS,
          uint32_t _BUFFERS_PER_THREAD,
          uint32_t _TLEV_BYTES_PER_THREAD,
          bool _PREFER_POW2_BITS,
          uint32_t _BLOCK_LEVEL_TILE_SIZE>
struct AgentBatchMemcpyPolicy
{
  enum
  {
    /// Threads per thread block
    BLOCK_THREADS = _BLOCK_THREADS,
    /// Items per thread (per tile of input)
    BUFFERS_PER_THREAD = _BUFFERS_PER_THREAD,
    /// The number of bytes that each thread will work on with each iteration of reading in bytes from one or more
    // source-buffers and writing them out to the respective destination-buffers.
    TLEV_BYTES_PER_THREAD = _TLEV_BYTES_PER_THREAD,
    /// Whether the BitPackedCounter should prefer allocating a power-of-2 number of btis per counter
    PREFER_POW2_BITS = _PREFER_POW2_BITS,
    /// BLEV tile size granularity
    BLOCK_LEVEL_TILE_SIZE = _BLOCK_LEVEL_TILE_SIZE
  };
};

template <typename AgentMemcpySmallBuffersPolicyT,
          typename InputBufferIt,
          typename OutputBufferIt,
          typename BufferSizeIteratorT,
          typename BufferOffsetT,
          typename BlevBufferSrcsOutItT,
          typename BlevBufferDstsOutItT,
          typename BlevBufferSizesOutItT,
          typename BlevBufferTileOffsetsOutItT,
          typename BlockOffsetT,
          typename BLevBufferOffsetTileState,
          typename BLevBlockOffsetTileState>
class AgentBatchMemcpy
{
private:
  //---------------------------------------------------------------------
  // CONFIGS / CONSTANTS
  //---------------------------------------------------------------------
  // Tuning policy-based configurations
  static constexpr uint32_t BLOCK_THREADS         = AgentMemcpySmallBuffersPolicyT::BLOCK_THREADS;
  static constexpr uint32_t BUFFERS_PER_THREAD    = AgentMemcpySmallBuffersPolicyT::BUFFERS_PER_THREAD;
  static constexpr uint32_t TLEV_BYTES_PER_THREAD = AgentMemcpySmallBuffersPolicyT::TLEV_BYTES_PER_THREAD;
  static constexpr bool PREFER_POW2_BITS          = AgentMemcpySmallBuffersPolicyT::PREFER_POW2_BITS;
  static constexpr uint32_t BLOCK_LEVEL_TILE_SIZE = AgentMemcpySmallBuffersPolicyT::BLOCK_LEVEL_TILE_SIZE;

  // Derived configs
  static constexpr uint32_t BUFFERS_PER_BLOCK       = BUFFERS_PER_THREAD * BLOCK_THREADS;
  static constexpr uint32_t TLEV_BUFFERS_PER_THREAD = BUFFERS_PER_THREAD;
  static constexpr uint32_t BLEV_BUFFERS_PER_THREAD = BUFFERS_PER_THREAD;

  static constexpr uint32_t WARP_LEVEL_THRESHOLD  = 64;
  static constexpr uint32_t BLOCK_LEVEL_THRESHOLD = 1024;

  // Constants
  enum : uint32_t
  {
    TLEV_SIZE_CLASS = 0,
    WLEV_SIZE_CLASS,
    BLEV_SIZE_CLASS,
    NUM_SIZE_CLASSES,
  };

  //---------------------------------------------------------------------
  // TYPE DECLARATIONS
  //---------------------------------------------------------------------
  /// Type that has to be sufficiently large to hold any of the buffers' sizes.
  /// The BufferSizeIteratorT's value type must be convertible to this type.
  using BufferSizeT = typename std::iterator_traits<BufferSizeIteratorT>::value_type;

  /// Type used to index into the tile of buffers that this thread block is assigned to.
  using BlockBufferOffsetT = uint16_t;

  /// Internal type used to index into the bytes of and represent size of a TLEV buffer
  using TLevBufferSizeT = uint16_t;

  /**
   * @brief Helper struct to simplify BlockExchange within a single four-byte word
   */
  struct ZippedTLevByteAssignment
  {
    // The buffer id within this tile
    BlockBufferOffsetT tile_buffer_id;

    // Byte-offset within that buffer
    TLevBufferSizeT buffer_byte_offset;
  };

  /**
   * POD to keep track of <buffer_id, buffer_size> pairs after having partitioned this tile's buffers by their size.
   */
  struct BufferTuple
  {
    // Size is only valid (and relevant) for buffers that are use thread-level collaboration
    TLevBufferSizeT size;

    // The buffer id relativ to this tile (i.e., the buffer id within this tile)
    BlockBufferOffsetT buffer_id;
  };

  // A vectorized counter that will count the number of buffers that fall into each of the size-classes. Where the size
  // class representes the collaboration level that is required to process a buffer. The collaboration level being
  // either:
  //-> (1) TLEV (thread-level collaboration), requiring one or multiple threads but not a FULL warp to collaborate
  //-> (2) WLEV (warp-level collaboration), requiring a full warp to collaborate on a buffer
  //-> (3) BLEV (block-level collaboration), requiring one or multiple thread blocks to collaborate on a buffer */
  using VectorizedSizeClassCounterT = BitPackedCounter<NUM_SIZE_CLASSES, BUFFERS_PER_BLOCK, PREFER_POW2_BITS>;

  // Block-level scan used to compute the write offsets
  using BlockSizeClassScanT = cub::BlockScan<VectorizedSizeClassCounterT, BLOCK_THREADS>;

  //
  using BlockBLevTileCountScanT = cub::BlockScan<BlockOffsetT, BLOCK_THREADS>;

  // Block-level run-length decode algorithm to evenly distribute work of all buffers requiring thread-level
  // collaboration
  using BlockRunLengthDecodeT = cub::BlockRunLengthDecode<BlockBufferOffsetT,
                                                          BLOCK_THREADS,
                                                          TLEV_BUFFERS_PER_THREAD,
                                                          TLEV_BYTES_PER_THREAD,
                                                          BlockRunLengthDecodeAlgorithm::OFFSETS>;

  using BlockExchangeTLevT = cub::BlockExchange<ZippedTLevByteAssignment, BLOCK_THREADS, TLEV_BYTES_PER_THREAD>;

  using BLevBuffScanPrefixCallbackOpT  = TilePrefixCallbackOp<BufferOffsetT, Sum, BLevBufferOffsetTileState>;
  using BLevBlockScanPrefixCallbackOpT = TilePrefixCallbackOp<BlockOffsetT, Sum, BLevBlockOffsetTileState>;

  //-----------------------------------------------------------------------------
  // SHARED MEMORY DECLARATIONS
  //-----------------------------------------------------------------------------
  struct _TempStorage
  {
    union
    {
      // Stage 1: histogram over the size classes in preparation for partitioning buffers by size
      typename BlockSizeClassScanT::TempStorage size_scan_storage;

      // Stage 2: Communicate the number ofer buffers requiring block-level collaboration
      struct
      {
        typename BLevBuffScanPrefixCallbackOpT::TempStorage buffer_scan_callback;
      };

      // Stage 3; batch memcpy buffers that require only thread-level collaboration
      struct
      {
        BufferTuple buffers_by_size_class[BUFFERS_PER_BLOCK];

        // Stage 3.1: Write buffers requiring block-level collaboration to queue
        union
        {
          struct
          {
            typename BLevBlockScanPrefixCallbackOpT::TempStorage block_scan_callback;
            typename BlockBLevTileCountScanT::TempStorage block_scan_storage;
          };

          // Stage 3.3:
          struct
          {
            typename BlockRunLengthDecodeT::TempStorage run_length_decode;
            typename BlockExchangeTLevT::TempStorage block_exchange_storage;
          };
        };
      };
    };
    BufferOffsetT blev_buffer_offset;
  };

  //-----------------------------------------------------------------------------
  // PUBLIC TYPE MEMBERS
  //-----------------------------------------------------------------------------
public:
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //-----------------------------------------------------------------------------
  // PRIVATE MEMBER FUNCTIONS
  //-----------------------------------------------------------------------------
private:
  /// Shared storage reference
  _TempStorage &temp_storage;

  /**
   * @brief Loads this tile's buffers' sizes, without any guards (i.e., out-of-bounds checks)
   */
  __device__ __forceinline__ void LoadBufferSizesFullTile(BufferSizeIteratorT tile_buffer_sizes_it,
                                                          BufferSizeT (&buffer_sizes)[BUFFERS_PER_THREAD])
  {
    LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, tile_buffer_sizes_it, buffer_sizes);
  }

  /**
   * @brief Loads this tile's buffers' sizes, making sure to read at most \p num_valid items.
   */
  template <typename NumValidT>
  __device__ __forceinline__ void LoadBufferSizesPartialTile(BufferSizeIteratorT tile_buffer_sizes_it,
                                                             BufferSizeT (&buffer_sizes)[BUFFERS_PER_THREAD],
                                                             NumValidT num_valid)
  {
    // Out-of-bounds buffer items are initialized to '0', so those buffers will simply be ignored later on
    constexpr BufferSizeT OOB_DEFAULT_BUFFER_SIZE = 0U;

    LoadDirectStriped<BLOCK_THREADS>(threadIdx.x,
                                     tile_buffer_sizes_it,
                                     buffer_sizes,
                                     num_valid,
                                     OOB_DEFAULT_BUFFER_SIZE);
  }

  /**
   * @brief Computes the histogram over the number of buffers belonging to each of the three size-classes (TLEV, WLEV,
   * BLEV).
   */
  template <typename VectorizedSizeClassCounterT>
  __device__ __forceinline__ void GetBufferSizeClassHistogram(const BufferSizeT (&buffer_sizes)[BUFFERS_PER_THREAD],
                                                              VectorizedSizeClassCounterT &vectorized_counters)
  {
#pragma unroll
    for (uint32_t i = 0; i < BUFFERS_PER_THREAD; i++)
    {
      // Whether to increment ANY of the buffer size classes at all
      uint32_t increment = buffer_sizes[i] > 0 ? 1U : 0U;
      // Identify the buffer's size class
      uint32_t buffer_size_class = 0;
      buffer_size_class += buffer_sizes[i] > WARP_LEVEL_THRESHOLD ? 1U : 0U;
      buffer_size_class += buffer_sizes[i] > BLOCK_LEVEL_THRESHOLD ? 1U : 0U;

      // Increment the count of the respective size class
      vectorized_counters.Add(buffer_size_class, increment);
    }
  }

  /**
   * @brief Scatters the buffers into the respective buffer's size-class partition.
   */
  template <typename VectorizedSizeClassCounterT>
  __device__ __forceinline__ void PartitionBuffersBySize(const BufferSizeT (&buffer_sizes)[BUFFERS_PER_THREAD],
                                                         VectorizedSizeClassCounterT &vectorized_offsets,
                                                         BufferTuple (&buffers_by_size_class)[BUFFERS_PER_BLOCK])
  {
    BlockBufferOffsetT buffer_id = threadIdx.x;
#pragma unroll
    for (uint32_t i = 0; i < BUFFERS_PER_THREAD; i++)
    {
      if (buffer_sizes[i] > 0)
      {
        uint32_t buffer_size_class = 0;
        buffer_size_class += buffer_sizes[i] > WARP_LEVEL_THRESHOLD ? 1U : 0U;
        buffer_size_class += buffer_sizes[i] > BLOCK_LEVEL_THRESHOLD ? 1U : 0U;
        uint32_t write_offset               = vectorized_offsets.Get(buffer_size_class);
        buffers_by_size_class[write_offset] = {static_cast<TLevBufferSizeT>(buffer_sizes[i]), buffer_id};
        vectorized_offsets.Add(buffer_size_class, 1U);
      }
      buffer_id += BLOCK_THREADS;
    }
  }

  __forceinline__ __device__ uint4 load_uint4(const char *ptr)
  {
    auto const offset       = reinterpret_cast<std::uintptr_t>(ptr) % 4;
    auto const *aligned_ptr = reinterpret_cast<unsigned int const *>(ptr - offset);
    auto const shift        = offset * 8;

    uint4 regs = {aligned_ptr[0], aligned_ptr[1], aligned_ptr[2], aligned_ptr[3]};
    uint tail  = 0;
    if (shift)
      tail = aligned_ptr[4];

    regs.x = __funnelshift_r(regs.x, regs.y, shift);
    regs.y = __funnelshift_r(regs.y, regs.z, shift);
    regs.z = __funnelshift_r(regs.z, regs.w, shift);
    regs.w = __funnelshift_r(regs.w, tail, shift);

    return regs;
  }

  /**
   * @brief Read in all the buffers that require block-level collaboration and put them to a queue that will get picked
   * up in a separate, subsequent kernel.
   */
  template <typename TileBufferOffsetT, typename TileOffsetT>
  __device__ __forceinline__ void EnqueueBLEVBuffers(BufferTuple *buffers_by_size_class,
                                                     InputBufferIt tile_buffer_srcs,
                                                     OutputBufferIt tile_buffer_dsts,
                                                     BufferSizeIteratorT tile_buffer_sizes,
                                                     BlockBufferOffsetT num_blev_buffers,
                                                     TileBufferOffsetT tile_buffer_offset,
                                                     TileOffsetT tile_id)
  {
    BlockOffsetT block_offset[BLEV_BUFFERS_PER_THREAD];
    // Read in the BLEV buffer partition (i.e., the buffers that require block-level collaboration)
    uint32_t blev_buffer_offset = threadIdx.x * BLEV_BUFFERS_PER_THREAD;
#pragma unroll
    for (uint32_t i = 0; i < BLEV_BUFFERS_PER_THREAD; i++)
    {
      if (blev_buffer_offset < num_blev_buffers)
      {
        BlockBufferOffsetT tile_buffer_id = buffers_by_size_class[blev_buffer_offset].buffer_id;
        block_offset[i] = CUB_QUOTIENT_CEILING(tile_buffer_sizes[tile_buffer_id], BLOCK_LEVEL_TILE_SIZE);
      }
      else
      {
        // Out-of-bounds buffers are assigned a tile count of '0'
        block_offset[i] = 0U;
      }
      blev_buffer_offset++;
    }

    if (tile_id == 0)
    {
      BlockOffsetT block_aggregate;
      BlockBLevTileCountScanT(temp_storage.block_scan_storage).ExclusiveSum(block_offset, block_offset, block_aggregate);
      if (threadIdx.x == 0)
      {
        blev_block_scan_state.SetInclusive(0, block_aggregate);
      }
    }
    else
    {
      BLevBlockScanPrefixCallbackOpT blev_tile_prefix_op(blev_block_scan_state,
                                                         temp_storage.block_scan_callback,
                                                         Sum(),
                                                         tile_id);
      BlockBLevTileCountScanT(temp_storage.block_scan_storage)
        .ExclusiveSum(block_offset, block_offset, blev_tile_prefix_op);
    }
    CTA_SYNC();

    // Read in the BLEV buffer partition (i.e., the buffers that require block-level collaboration)
    blev_buffer_offset = threadIdx.x * BLEV_BUFFERS_PER_THREAD;
#pragma unroll
    for (uint32_t i = 0; i < BLEV_BUFFERS_PER_THREAD; i++)
    {
      if (blev_buffer_offset < num_blev_buffers)
      {
        BlockBufferOffsetT tile_buffer_id                         = buffers_by_size_class[blev_buffer_offset].buffer_id;
        blev_buffer_srcs[tile_buffer_offset + blev_buffer_offset] = tile_buffer_srcs[tile_buffer_id];
        blev_buffer_dsts[tile_buffer_offset + blev_buffer_offset] = tile_buffer_dsts[tile_buffer_id];
        blev_buffer_sizes[tile_buffer_offset + blev_buffer_offset]        = tile_buffer_sizes[tile_buffer_id];
        blev_buffer_tile_offsets[tile_buffer_offset + blev_buffer_offset] = block_offset[i];
        blev_buffer_offset++;
      }
    }
  }

  /**
   * @brief Read in all the buffers of this tile that require warp-level collaboration and copy their bytes to the
   * corresponding destination buffer
   */
  __device__ __forceinline__ void BatchMemcpyWLEVBuffers(BufferTuple *buffers_by_size_class,
                                                         InputBufferIt tile_buffer_srcs,
                                                         OutputBufferIt tile_buffer_dsts,
                                                         BufferSizeIteratorT tile_buffer_sizes,
                                                         BlockBufferOffsetT num_wlev_buffers)
  {
    constexpr size_t out_datatype_size = sizeof(uint4);
    constexpr size_t in_datatype_size  = sizeof(uint);

    int warp_id                        = threadIdx.x / CUB_PTX_WARP_THREADS;
    int warp_lane                      = threadIdx.x % CUB_PTX_WARP_THREADS;
    constexpr uint32_t WARPS_PER_BLOCK = BLOCK_THREADS / CUB_PTX_WARP_THREADS;

    for (BlockBufferOffsetT istring = warp_id; istring < num_wlev_buffers; istring += WARPS_PER_BLOCK)
    {
      uint8_t *out_chars = reinterpret_cast<uint8_t *>(tile_buffer_dsts[buffers_by_size_class[istring].buffer_id]);
      auto const alignment_offset = reinterpret_cast<std::uintptr_t>(out_chars) % out_datatype_size;
      uint4 *out_chars_aligned    = reinterpret_cast<uint4 *>(out_chars - alignment_offset);

      auto const out_start = 0;
      auto const out_end   = out_start + tile_buffer_sizes[buffers_by_size_class[istring].buffer_id];

      const char *in_start = reinterpret_cast<const char *>(tile_buffer_srcs[buffers_by_size_class[istring].buffer_id]);

      // Both `out_start_aligned` and `out_end_aligned` are indices into `out_chars`.
      // `out_start_aligned` is the first 16B aligned memory location after `out_start + 4`.
      // `out_end_aligned` is the last 16B aligned memory location before `out_end - 4`. Characters
      // between `[out_start_aligned, out_end_aligned)` will be copied using uint4.
      // `out_start + 4` and `out_end - 4` are used instead of `out_start` and `out_end` to avoid
      // `load_uint4` reading beyond string boundaries.
      int32_t out_start_aligned = (out_start + in_datatype_size + alignment_offset + out_datatype_size - 1) /
                                    out_datatype_size * out_datatype_size -
                                  alignment_offset;
      int32_t out_end_aligned =
        (out_end - in_datatype_size + alignment_offset) / out_datatype_size * out_datatype_size - alignment_offset;

      for (BufferSizeT ichar = out_start_aligned + warp_lane * out_datatype_size; ichar < out_end_aligned;
           ichar += CUB_PTX_WARP_THREADS * out_datatype_size)
      {
        *(out_chars_aligned + (ichar + alignment_offset) / out_datatype_size) =
          load_uint4(in_start + ichar - out_start);
      }

      // Tail logic: copy characters of the current string outside `[out_start_aligned,
      // out_end_aligned)`.
      if (out_end_aligned <= out_start_aligned)
      {
        // In this case, `[out_start_aligned, out_end_aligned)` is an empty set, and we copy the
        // entire string.
        for (int32_t ichar = out_start + warp_lane; ichar < out_end; ichar += CUB_PTX_WARP_THREADS)
        {
          out_chars[ichar] = in_start[ichar - out_start];
        }
      }
      else
      {
        // Copy characters in range `[out_start, out_start_aligned)`.
        if (out_start + warp_lane < out_start_aligned)
        {
          out_chars[out_start + warp_lane] = in_start[warp_lane];
        }
        // Copy characters in range `[out_end_aligned, out_end)`.
        int32_t ichar = out_end_aligned + warp_lane;
        if (ichar < out_end)
        {
          out_chars[ichar] = in_start[ichar - out_start];
        }
      }
    }
  }

  /**
   * @brief Read in all the buffers of this tile that require thread-level collaboration and copy their bytes to the
   * corresponding destination buffer
   */
  __device__ __forceinline__ void BatchMemcpyTLEVBuffers(BufferTuple *buffers_by_size_class,
                                                         InputBufferIt tile_buffer_srcs,
                                                         OutputBufferIt tile_buffer_dsts,
                                                         BlockBufferOffsetT num_tlev_buffers)
  {
    // Read in the buffers' ids that require thread-level collaboration (where buffer id is the buffer within this tile)
    BlockBufferOffsetT tlev_buffer_ids[TLEV_BUFFERS_PER_THREAD];
    TLevBufferSizeT tlev_buffer_sizes[TLEV_BUFFERS_PER_THREAD];
    // Currently we do not go over the TLEV buffers in multiple iterations, so we need to make sure we are able to be
    // covered for the case that all our buffers are TLEV buffers
    static_assert(TLEV_BUFFERS_PER_THREAD >= BUFFERS_PER_THREAD);

    // Read in the TLEV buffer partition (i.e., the buffers that require thread-level collaboration)
    uint32_t tlev_buffer_offset = threadIdx.x * TLEV_BUFFERS_PER_THREAD;
#pragma unroll
    for (uint32_t i = 0; i < TLEV_BUFFERS_PER_THREAD; i++)
    {
      if (tlev_buffer_offset < num_tlev_buffers)
      {
        tlev_buffer_ids[i]   = buffers_by_size_class[tlev_buffer_offset].buffer_id;
        tlev_buffer_sizes[i] = buffers_by_size_class[tlev_buffer_offset].size;
      }
      else
      {
        // Out-of-bounds buffers are assigned a size of '0'
        tlev_buffer_sizes[i] = 0;
      }
      tlev_buffer_offset++;
    }

    // Evenly distribute all the bytes that have to be copied from all the buffers that require thread-level
    // collaboration using BlockRunLengthDecode
    uint32_t num_total_tlev_bytes = 0U;
    BlockRunLengthDecodeT block_run_length_decode(temp_storage.run_length_decode);
    block_run_length_decode.Init(tlev_buffer_ids, tlev_buffer_sizes, num_total_tlev_bytes);

    // Run-length decode the buffers' sizes into a window buffer of limited size. This is repeated until we were able to
    // cover all the bytes of TLEV buffers
    uint32_t decoded_window_offset = 0U;
    while (decoded_window_offset < num_total_tlev_bytes)
    {
      BlockBufferOffsetT buffer_id[TLEV_BYTES_PER_THREAD];
      TLevBufferSizeT buffer_byte_offset[TLEV_BYTES_PER_THREAD];

      // Now we have a balanced assignment: buffer_id[i] will hold the tile's buffer id and buffer_byte_offset[i] that
      // buffer's byte that we're supposed to copy
      block_run_length_decode.RunLengthDecode(buffer_id, buffer_byte_offset, decoded_window_offset);

      // Zip from SoA to AoS
      ZippedTLevByteAssignment zipped_byte_assignment[TLEV_BYTES_PER_THREAD];
#pragma unroll
      for (int32_t i = 0; i < TLEV_BYTES_PER_THREAD; i++)
      {
        zipped_byte_assignment[i] = {buffer_id[i], buffer_byte_offset[i]};
      }

      // Exchange from blocked to striped arrangement for coalesced memory reads and writes
      BlockExchangeTLevT(temp_storage.block_exchange_storage)
        .BlockedToStriped(zipped_byte_assignment, zipped_byte_assignment);

      // Read in the bytes that this thread is assigned to
      constexpr uint32_t WINDOW_SIZE = (TLEV_BYTES_PER_THREAD * BLOCK_THREADS);
      bool is_full_window            = decoded_window_offset + WINDOW_SIZE < num_total_tlev_bytes;
      if (is_full_window)
      {
        uint32_t absolute_tlev_byte_offset = decoded_window_offset + threadIdx.x;
#pragma unroll
        for (int32_t i = 0; i < TLEV_BYTES_PER_THREAD; i++)
        {
          uint8_t src_byte = reinterpret_cast<const uint8_t *>(
            tile_buffer_srcs[zipped_byte_assignment[i].tile_buffer_id])[zipped_byte_assignment[i].buffer_byte_offset];
          reinterpret_cast<uint8_t *>(
            tile_buffer_dsts[zipped_byte_assignment[i].tile_buffer_id])[zipped_byte_assignment[i].buffer_byte_offset] =
            src_byte;
          absolute_tlev_byte_offset += BLOCK_THREADS;
        }
      }
      else
      {
        uint32_t absolute_tlev_byte_offset = decoded_window_offset + threadIdx.x;
#pragma unroll
        for (int32_t i = 0; i < TLEV_BYTES_PER_THREAD; i++)
        {
          if (absolute_tlev_byte_offset < num_total_tlev_bytes)
          {
            uint8_t src_byte = reinterpret_cast<const uint8_t *>(
              tile_buffer_srcs[zipped_byte_assignment[i].tile_buffer_id])[zipped_byte_assignment[i].buffer_byte_offset];
            reinterpret_cast<uint8_t *>(
              tile_buffer_dsts[zipped_byte_assignment[i].tile_buffer_id])[zipped_byte_assignment[i].buffer_byte_offset] =
              src_byte;
          }
          absolute_tlev_byte_offset += BLOCK_THREADS;
        }
      }

      decoded_window_offset += WINDOW_SIZE;
    }
  }

  //-----------------------------------------------------------------------------
  // PUBLIC MEMBER FUNCTIONS
  //-----------------------------------------------------------------------------
public:
  __device__ __forceinline__ void ConsumeTile(BufferOffsetT tile_id)
  {
    // Offset into this tile's buffers
    BufferOffsetT buffer_offset = tile_id * BUFFERS_PER_BLOCK;

    // Indicates whether all of this tiles items are within bounds
    bool is_full_tile = buffer_offset + BUFFERS_PER_BLOCK < num_buffers;

    // Load the buffer sizes of this tile's buffers
    BufferSizeIteratorT tile_buffer_sizes_it = buffer_sizes_it + buffer_offset;
    BufferSizeT buffer_sizes[BUFFERS_PER_THREAD];
    if (is_full_tile)
    {
      LoadBufferSizesFullTile(tile_buffer_sizes_it, buffer_sizes);
    }
    else
    {
      LoadBufferSizesPartialTile(tile_buffer_sizes_it, buffer_sizes, num_buffers - buffer_offset);
    }

    // Count how many buffers fall into each size-class
    VectorizedSizeClassCounterT size_class_histogram = {};
    GetBufferSizeClassHistogram(buffer_sizes, size_class_histogram);

    // Compute the prefix sum over the histogram
    VectorizedSizeClassCounterT size_class_agg = {};
    BlockSizeClassScanT(temp_storage.size_scan_storage)
      .ExclusiveSum(size_class_histogram, size_class_histogram, size_class_agg);

    // Ensure we can repurpose the scan's temporary storage for scattering the buffer ids
    CTA_SYNC();

    // Factor in the per-size-class counts / offsets
    // That is, WLEV buffer offset has to be offset by the TLEV buffer count and BLEV buffer offset has to be offset by
    // the TLEV+WLEV buffer count
    uint32_t buffer_count = 0U;
    for (uint32_t i = 0; i < NUM_SIZE_CLASSES; i++)
    {
      size_class_histogram.Add(i, buffer_count);
      buffer_count += size_class_agg.Get(i);
    }

    // Signal the number of BLEV buffers we're planning to write out
    BufferOffsetT buffer_exclusive_prefix = 0;
    if (tile_id == 0)
    {
      if (threadIdx.x == 0)
      {
        blev_buffer_scan_state.SetInclusive(tile_id, size_class_agg.Get(BLEV_SIZE_CLASS));
      }
      buffer_exclusive_prefix = 0;
    }
    else
    {
      BLevBuffScanPrefixCallbackOpT blev_buffer_prefix_op(blev_buffer_scan_state,
                                                          temp_storage.buffer_scan_callback,
                                                          Sum(),
                                                          tile_id);

      // Signal our partial prefix and wait for the inclusive prefix of previous tiles
      if (threadIdx.x < CUB_PTX_WARP_THREADS)
      {
        buffer_exclusive_prefix = blev_buffer_prefix_op(size_class_agg.Get(BLEV_SIZE_CLASS));
      }
    }
    if (threadIdx.x == 0)
    {
      temp_storage.blev_buffer_offset = buffer_exclusive_prefix;
    }

    // Ensure the prefix callback has finished using its temporary storage and that it can be reused in the next stage
    CTA_SYNC();

    // Scatter the buffers into one of the three partitions (TLEV, WLEV, BLEV) depending on their size
    PartitionBuffersBySize(buffer_sizes, size_class_histogram, temp_storage.buffers_by_size_class);

    // Ensure all buffers have been partitioned by their size class AND
    // ensure that blev_buffer_offset has been written to shared memory
    CTA_SYNC();

    // TODO: think about prefetching tile_buffer_{srcs,dsts} into shmem
    InputBufferIt tile_buffer_srcs        = input_buffer_it + buffer_offset;
    InputBufferIt tile_buffer_dsts        = output_buffer_it + buffer_offset;
    BufferSizeIteratorT tile_buffer_sizes = buffer_sizes_it + buffer_offset;

    // Copy block-level buffers
    EnqueueBLEVBuffers(
      &temp_storage.buffers_by_size_class[size_class_agg.Get(TLEV_SIZE_CLASS) + size_class_agg.Get(WLEV_SIZE_CLASS)],
      tile_buffer_srcs,
      tile_buffer_dsts,
      tile_buffer_sizes,
      size_class_agg.Get(BLEV_SIZE_CLASS),
      temp_storage.blev_buffer_offset,
      tile_id);

    // Ensure we can repurpose the temporary storage required by EnqueueBLEVBuffers
    CTA_SYNC();

    // Copy warp-level buffers
    BatchMemcpyWLEVBuffers(&temp_storage.buffers_by_size_class[size_class_agg.Get(TLEV_SIZE_CLASS)],
                           tile_buffer_srcs,
                           tile_buffer_dsts,
                           tile_buffer_sizes,
                           size_class_agg.Get(WLEV_SIZE_CLASS));

    // Perform batch memcpy for all the buffers that require thread-level collaboration
    uint32_t num_tlev_buffers = size_class_agg.Get(TLEV_SIZE_CLASS);
    BatchMemcpyTLEVBuffers(temp_storage.buffers_by_size_class, tile_buffer_srcs, tile_buffer_dsts, num_tlev_buffers);
  }

  //-----------------------------------------------------------------------------
  // CONSTRUCTOR
  //-----------------------------------------------------------------------------
  __device__ __forceinline__ AgentBatchMemcpy(TempStorage &temp_storage,
                                              InputBufferIt input_buffer_it,
                                              OutputBufferIt output_buffer_it,
                                              BufferSizeIteratorT buffer_sizes_it,
                                              BufferOffsetT num_buffers,
                                              BlevBufferSrcsOutItT blev_buffer_srcs,
                                              BlevBufferDstsOutItT blev_buffer_dsts,
                                              BlevBufferSizesOutItT blev_buffer_sizes,
                                              BlevBufferTileOffsetsOutItT blev_buffer_tile_offsets,
                                              BLevBufferOffsetTileState blev_buffer_scan_state,
                                              BLevBlockOffsetTileState blev_block_scan_state)
      : temp_storage(temp_storage.Alias())
      , input_buffer_it(input_buffer_it)
      , output_buffer_it(output_buffer_it)
      , buffer_sizes_it(buffer_sizes_it)
      , num_buffers(num_buffers)
      , blev_buffer_srcs(blev_buffer_srcs)
      , blev_buffer_dsts(blev_buffer_dsts)
      , blev_buffer_sizes(blev_buffer_sizes)
      , blev_buffer_tile_offsets(blev_buffer_tile_offsets)
      , blev_buffer_scan_state(blev_buffer_scan_state)
      , blev_block_scan_state(blev_block_scan_state)
  {}

private:
  // Iterator providing the pointers to the source memory buffers
  InputBufferIt input_buffer_it;
  // Iterator providing the pointers to the destination memory buffers
  OutputBufferIt output_buffer_it;
  // Iterator providing the number of bytes to be copied for each pair of buffers
  BufferSizeIteratorT buffer_sizes_it;
  // The total number of buffer pairs
  BufferOffsetT num_buffers;
  // Output iterator to which the source pointers of the BLEV buffers are written
  BlevBufferSrcsOutItT blev_buffer_srcs;
  // Output iterator to which the destination pointers of the BLEV buffers are written
  BlevBufferDstsOutItT blev_buffer_dsts;
  // Output iterator to which the number of bytes to be copied of the BLEV buffers are written
  BlevBufferSizesOutItT blev_buffer_sizes;
  // Output iterator to which the mapping of tiles to BLEV buffers is written
  BlevBufferTileOffsetsOutItT blev_buffer_tile_offsets;
  // The single-pass prefix scan's tile state used for tracking the prefix sum over the number of BLEV buffers
  BLevBufferOffsetTileState blev_buffer_scan_state;
  // The single-pass prefix scan's tile state used for tracking the prefix sum over tiles of BLEV buffers
  BLevBlockOffsetTileState blev_block_scan_state;
};

CUB_NAMESPACE_END