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
 * cub::DispatchBatchMemcpy provides device-wide, parallel operations for copying data from a number of given source
 * buffers to their corresponding destination buffer.
 */

#pragma once

/// Optional outer namespace(s)
#include <cstdint>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include "../../agent/agent_batch_memcpy.cuh"
#include "../../agent/single_pass_scan_operators.cuh"
#include "../../config.cuh"
#include "../../thread/thread_search.cuh"

CUB_NAMESPACE_BEGIN

using uint = unsigned int;

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
 * Initialization kernel for tile status initialization (multi-block)
 */
template <typename ScanTileStateT, typename TileOffsetT>
__global__ void InitTileStateKernel(ScanTileStateT tile_state, TileOffsetT num_tiles)
{
  // Initialize tile status
  tile_state.InitializeStatus(num_tiles);
}

template <typename ByteOffsetT>
__device__ __forceinline__ void VectorizedCopy(uint8_t *out_chars, ByteOffsetT num_bytes, const char *in_start)
{
  constexpr size_t out_datatype_size = sizeof(uint4);
  constexpr size_t in_datatype_size  = sizeof(uint);

  int warp_lane = threadIdx.x;

  auto const alignment_offset = reinterpret_cast<std::uintptr_t>(out_chars) % out_datatype_size;
  uint4 *out_chars_aligned    = reinterpret_cast<uint4 *>(out_chars - alignment_offset);

  auto const out_start = 0;
  auto const out_end   = out_start + num_bytes;

  // Both `out_start_aligned` and `out_end_aligned` are indices into `out_chars`.
  // `out_start_aligned` is the first 16B aligned memory location after `out_start + 4`.
  // `out_end_aligned` is the last 16B aligned memory location before `out_end - 4`. Characters
  // between `[out_start_aligned, out_end_aligned)` will be copied using uint4.
  // `out_start + 4` and `out_end - 4` are used instead of `out_start` and `out_end` to avoid
  // `load_uint4` reading beyond string boundaries.
  int32_t out_start_aligned = (out_start + in_datatype_size + alignment_offset + out_datatype_size - 1) /
                                out_datatype_size * out_datatype_size -
                              alignment_offset;
  int32_t out_end_aligned = (out_end - in_datatype_size + alignment_offset) / out_datatype_size * out_datatype_size -
                            alignment_offset;

  for (ByteOffsetT ichar = out_start_aligned + warp_lane * out_datatype_size; ichar < out_end_aligned;
       ichar += blockDim.x * out_datatype_size)
  {
    *(out_chars_aligned + (ichar + alignment_offset) / out_datatype_size) = load_uint4(in_start + ichar - out_start);
  }

  // Tail logic: copy characters of the current string outside `[out_start_aligned,
  // out_end_aligned)`.
  if (out_end_aligned <= out_start_aligned)
  {
    // In this case, `[out_start_aligned, out_end_aligned)` is an empty set, and we copy the
    // entire string.
    for (int32_t ichar = out_start + warp_lane; ichar < out_end; ichar += blockDim.x)
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

/**
 * Parameterizable tuning policy type for AgentBatchMemcpy
 */
template <uint32_t _BLOCK_THREADS, uint32_t _BYTES_PER_THREAD>
struct AgentBatchMemcpyLargeBuffersPolicy
{
  enum
  {
    /// Threads per thread block
    BLOCK_THREADS = _BLOCK_THREADS,
    /// The number of bytes each thread copies
    BYTES_PER_THREAD = _BYTES_PER_THREAD
  };
};

/**
 * Kernel that copies buffers that need to be copied by at least one (and potentially many) thread blocks.
 */
template <typename ChainedPolicyT,
          typename BufferOffsetT,
          typename InputBufferIt,
          typename OutputBufferIt,
          typename BufferSizeIteratorT,
          typename BufferTileOffsetItT,
          typename TileT,
          typename TileOffsetT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::AgentLargeBufferPolicyT::BLOCK_THREADS)) __global__
  void MultiBlockBatchMemcpyKernel(InputBufferIt input_buffer_it,
                                   OutputBufferIt output_buffer_it,
                                   BufferSizeIteratorT buffer_sizes,
                                   BufferTileOffsetItT buffer_tile_offsets,
                                   TileT buffer_offset_tile,
                                   TileOffsetT last_tile_offset)
{
  using StatusWord    = typename TileT::StatusWord;
  using ActivePolicyT = typename ChainedPolicyT::ActivePolicy::AgentLargeBufferPolicyT;

  constexpr uint32_t BLOCK_THREADS    = ActivePolicyT::BLOCK_THREADS;
  constexpr uint32_t ITEMS_PER_THREAD = ActivePolicyT::BYTES_PER_THREAD;
  constexpr uint32_t ITEMS_PER_BLOCK  = BLOCK_THREADS * ITEMS_PER_THREAD;

  __shared__ BufferOffsetT num_blev_buffers;

  uint32_t tile_id = blockIdx.x;
  if (threadIdx.x < CUB_PTX_WARP_THREADS)
  {
    StatusWord status_word;
    BufferOffsetT tmp_blev_buffers;
    buffer_offset_tile.WaitForValid(last_tile_offset, status_word, tmp_blev_buffers);
    if (threadIdx.x == 0)
      num_blev_buffers = tmp_blev_buffers;
  }

  // Make sure we've fetched read the total number of block-level buffers into num_blev_buffers
  __syncthreads();

  // No block-level buffers => we're done here
  if (num_blev_buffers == 0)
    return;

  // While there's still tiles of bytes from block-level buffers to copied
  do
  {
    __shared__ uint32_t buffer_id;

    // Binary search the buffer that this tile belongs to
    if (threadIdx.x == 0)
      buffer_id = UpperBound(buffer_tile_offsets, num_blev_buffers, tile_id) - 1;

    __syncthreads();

    // The relative offset of this tile within the buffer it's assigned to
    uint32_t tile_offset_within_buffer = (tile_id - buffer_tile_offsets[buffer_id]) * ITEMS_PER_BLOCK;

    // If the tile has already reached beyond the work of the end of the last buffer
    if (buffer_id >= num_blev_buffers - 1 && tile_offset_within_buffer > buffer_sizes[buffer_id])
    {
      return;
    }

    // Tiny remainders are copied without vectorizing laods
    if (buffer_sizes[buffer_id] - tile_offset_within_buffer <= 32)
    {
      uint32_t thread_offset = tile_offset_within_buffer + threadIdx.x;
      for (int i = 0; i < ITEMS_PER_THREAD; i++)
      {
        if (thread_offset < buffer_sizes[buffer_id])
        {
          reinterpret_cast<uint8_t *>(output_buffer_it[buffer_id])[thread_offset] =
            reinterpret_cast<const uint8_t *>(input_buffer_it[buffer_id])[thread_offset];
        }
        thread_offset += BLOCK_THREADS;
      }
    }
    else
    {
      VectorizedCopy(&reinterpret_cast<uint8_t *>(output_buffer_it[buffer_id])[tile_offset_within_buffer],
                     min(buffer_sizes[buffer_id] - tile_offset_within_buffer, ITEMS_PER_BLOCK),
                     &reinterpret_cast<const char *>(input_buffer_it[buffer_id])[tile_offset_within_buffer]);
    }

    tile_id += gridDim.x;
  } while (true);
}

/**
 * @brief Kernel that copies data from a batch of given source buffers to their corresponding destination buffer. If a
 * buffer's size is to large to be copied by a single thread block, that buffer is put into a queue of buffers that
 * will get picked up later on, where multiple blocks collaborate on each of these buffers. All other buffers get copied
 * straight away.
 *
 * @param input_buffer_it [in] Iterator providing the pointers to the source memory buffers
 * @param output_buffer_it [in] Iterator providing the pointers to the destination memory buffers
 * @param buffer_sizes [in] Iterator providing the number of bytes to be copied for each pair of buffers
 * @param num_buffers [in] The total number of buffer pairs
 * @param blev_buffer_srcs [out] The source pointers of buffers that require block-level collaboration
 * @param blev_buffer_dsts [out] The destination pointers of buffers that require block-level collaboration
 * @param blev_buffer_sizes [out] The sizes of buffers that require block-level collaboration
 * @param blev_buffer_scan_state [in,out] Tile states for the prefix sum over the count of buffers requiring
 * block-level collaboration (to "stream compact" (aka "select") BLEV-buffers)
 * @param blev_block_scan_state [in,out] Tile states for the prefix sum over the number of thread blocks getting
 * assigned to each buffer that requires block-level collaboration
 */
template <typename ChainedPolicyT,
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
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::AgentSmallBufferPolicyT::BLOCK_THREADS)) __global__
  void BatchMemcpyKernel(InputBufferIt input_buffer_it,
                         OutputBufferIt output_buffer_it,
                         BufferSizeIteratorT buffer_sizes,
                         BufferOffsetT num_buffers,
                         BlevBufferSrcsOutItT blev_buffer_srcs,
                         BlevBufferDstsOutItT blev_buffer_dsts,
                         BlevBufferSizesOutItT blev_buffer_sizes,
                         BlevBufferTileOffsetsOutItT blev_buffer_tile_offsets,
                         BLevBufferOffsetTileState blev_buffer_scan_state,
                         BLevBlockOffsetTileState blev_block_scan_state)
{
  static_assert(std::is_pointer<typename std::iterator_traits<InputBufferIt>::value_type>::value);
  static_assert(std::is_pointer<typename std::iterator_traits<OutputBufferIt>::value_type>::value);

  // Internal type used for storing a buffer's size
  using BufferSizeT = typename std::iterator_traits<BufferSizeIteratorT>::value_type;

  // Alias the correct tuning policy for the current compilation pass' architecture
  using AgentBatchMemcpyPolicyT = typename ChainedPolicyT::ActivePolicy::AgentSmallBufferPolicyT;

  // Block-level specialization
  using AgentBatchMemcpyT = AgentBatchMemcpy<AgentBatchMemcpyPolicyT,
                                             InputBufferIt,
                                             OutputBufferIt,
                                             BufferSizeIteratorT,
                                             BufferOffsetT,
                                             BlevBufferSrcsOutItT,
                                             BlevBufferDstsOutItT,
                                             BlevBufferSizesOutItT,
                                             BlevBufferTileOffsetsOutItT,
                                             BlockOffsetT,
                                             BLevBufferOffsetTileState,
                                             BLevBlockOffsetTileState>;

  // Shared memory for AgentBatchMemcpy
  __shared__ typename AgentBatchMemcpyT::TempStorage temp_storage;

  // Process this block's tile of input&output buffer pairs
  AgentBatchMemcpyT(temp_storage,
                    input_buffer_it,
                    output_buffer_it,
                    buffer_sizes,
                    num_buffers,
                    blev_buffer_srcs,
                    blev_buffer_dsts,
                    blev_buffer_sizes,
                    blev_buffer_tile_offsets,
                    blev_buffer_scan_state,
                    blev_block_scan_state)
    .ConsumeTile(blockIdx.x);
}

template <typename BufferOffsetT, typename BlockOffsetT>
struct DeviceBatchMemcpyPolicy
{
  static constexpr uint32_t BLOCK_THREADS         = 128U;
  static constexpr uint32_t BUFFERS_PER_THREAD    = 2U;
  static constexpr uint32_t TLEV_BYTES_PER_THREAD = 8U;

  static constexpr uint32_t LARGE_BUFFER_BLOCK_THREADS    = 256U;
  static constexpr uint32_t LARGE_BUFFER_BYTES_PER_THREAD = 32U;
  // TODO make true default for pre-Volta, false for Volta and after
  static constexpr bool PREFER_POW2_BITS = false;

  /// SM35
  struct Policy350 : ChainedPolicy<350, Policy350, Policy350>
  {
    using AgentSmallBufferPolicyT = AgentBatchMemcpyPolicy<BLOCK_THREADS,
                                                           BUFFERS_PER_THREAD,
                                                           TLEV_BYTES_PER_THREAD,
                                                           PREFER_POW2_BITS,
                                                           LARGE_BUFFER_BLOCK_THREADS * LARGE_BUFFER_BYTES_PER_THREAD>;

    using AgentLargeBufferPolicyT =
      AgentBatchMemcpyLargeBuffersPolicy<LARGE_BUFFER_BLOCK_THREADS, LARGE_BUFFER_BYTES_PER_THREAD>;
  };

  using MaxPolicy = Policy350;
};

/**
 * @tparam InputBufferIt <b>[inferred]</b> Random-access input iterator type providing the pointers to the source
 * memory buffers
 * @tparam OutputBufferIt <b>[inferred]</b> Random-access input iterator type providing the pointers to the
 * destination memory buffers
 * @tparam BufferSizeIteratorT <b>[inferred]</b> Random-access input iterator type providing the number of bytes to be
 * copied for each pair of buffers
 * @tparam BufferOffsetT Integer type large enough to hold any offset in [0, num_buffers)
 * @tparam BlockOffsetT Integer type large enough to hold any offset in [0, num_thread_blocks_launched)
 */
template <typename InputBufferIt,
          typename OutputBufferIt,
          typename BufferSizeIteratorT,
          typename BufferOffsetT,
          typename BlockOffsetT>
struct DispatchBatchMemcpy : DeviceBatchMemcpyPolicy<BufferOffsetT, BlockOffsetT>
{
  //------------------------------------------------------------------------------
  // TYPE ALIASES
  //------------------------------------------------------------------------------
  // Tile state for the single-pass prefix scan to "stream compact" (aka "select") the buffers requiring block-level
  // collaboration
  using BufferPartitionScanTileStateT = typename cub::ScanTileState<BufferOffsetT>;

  // Tile state for the single-pass prefix scan to keep track of how many blocks are assigned to each of the buffers
  // requiring block-level collaboration
  using BufferTileOffsetScanStateT = typename cub::ScanTileState<BlockOffsetT>;

  // Internal type used to keep track of a buffer's size
  using BufferSizeT = typename std::iterator_traits<BufferSizeIteratorT>::value_type;

  //------------------------------------------------------------------------------
  // Member Veriables
  //------------------------------------------------------------------------------
  void *d_temp_storage;
  size_t &temp_storage_bytes;
  InputBufferIt input_buffer_it;
  OutputBufferIt output_buffer_it;
  BufferSizeIteratorT buffer_sizes;
  BufferOffsetT num_buffers;
  cudaStream_t stream;
  bool debug_synchronous;
  int ptx_version;

  //------------------------------------------------------------------------------
  // Constructor
  //------------------------------------------------------------------------------
  CUB_RUNTIME_FUNCTION __forceinline__ DispatchBatchMemcpy(void *d_temp_storage,
                                                           size_t &temp_storage_bytes,
                                                           InputBufferIt input_buffer_it,
                                                           OutputBufferIt output_buffer_it,
                                                           BufferSizeIteratorT buffer_sizes,
                                                           BufferOffsetT num_buffers,
                                                           cudaStream_t stream,
                                                           bool debug_synchronous,
                                                           int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , input_buffer_it(input_buffer_it)
      , output_buffer_it(output_buffer_it)
      , buffer_sizes(buffer_sizes)
      , num_buffers(num_buffers)
      , stream(stream)
      , debug_synchronous(debug_synchronous)
      , ptx_version(ptx_version)
  {}

  //------------------------------------------------------------------------------
  // Chained policy invocation
  //------------------------------------------------------------------------------
  /**
   * @brief Tuning policy invocation. This member function template is getting instantiated for all tuning policies in
   * the tuning policy chain. It is, however, *invoked* for the correct tuning policy only.
   */
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t Invoke()
  {
    using MaxPolicyT = typename DispatchBatchMemcpy::MaxPolicy;

    // Single-pass prefix scan tile states for the prefix-sum over the number of block-level buffers
    using BLevBufferOffsetTileState = cub::ScanTileState<BufferOffsetT>;

    // Single-pass prefix scan tile states for the prefix sum over the number of thread blocks assigned to each of the
    // block-level buffers
    using BLevBlockOffsetTileState = cub::ScanTileState<BlockOffsetT>;

    cudaError error = cudaSuccess;

    enum : uint32_t
    {
      // Memory for the source pointers of the buffers that require block-level collaboration
      MEM_BLEV_BUFFER_SRCS = 0,
      // Memory for the destination pointers of the buffers that require block-level collaboration
      MEM_BLEV_BUFFER_DSTS,
      // Memory for the block-level buffers' sizes
      MEM_BLEV_BUFFER_SIZES,
      // Memory to keep track of the assignment of thread blocks to block-level buffers
      MEM_BLEV_BUFFER_TBLOCK,
      // Memory for the tile states of the prefix sum over the number of buffers that require block-level collaboration
      MEM_BLEV_BUFFER_SCAN_STATE,
      // Memory for the scan tile states of the prefix sum over the number of thread block's assigned up to and
      // including a certain block-level buffer
      MEM_BLEV_BLOCK_SCAN_STATE,
      // Total number of distinct memory allocations in the temporary storage memory BLOB
      MEM_NUM_ALLOCATIONS
    };

    // Number of threads per block for initializing the grid states
    constexpr uint32_t INIT_KERNEL_THREADS = 128U;

    // The number of buffers that get processed per thread block
    constexpr uint32_t TILE_SIZE = ActivePolicyT::AgentSmallBufferPolicyT::BLOCK_THREADS *
                                   ActivePolicyT::AgentSmallBufferPolicyT::BUFFERS_PER_THREAD;

    // The number of thread blocks (or tiles) required to process all of the given buffers
    BlockOffsetT num_tiles = cub::DivideAndRoundUp(num_buffers, TILE_SIZE);

    // Temporary storage allocation requirements
    void *allocations[MEM_NUM_ALLOCATIONS]       = {};
    size_t allocation_sizes[MEM_NUM_ALLOCATIONS] = {};

    using BlevBufferSrcsOutItT        = void **;
    using BlevBufferDstsOutItT        = void **;
    using BlevBufferSizesOutItT       = BufferSizeT *;
    using BlevBufferTileOffsetsOutItT = BlockOffsetT *;

    allocation_sizes[MEM_BLEV_BUFFER_SRCS]   = num_buffers * sizeof(void *);
    allocation_sizes[MEM_BLEV_BUFFER_DSTS]   = num_buffers * sizeof(void *);
    allocation_sizes[MEM_BLEV_BUFFER_SIZES]  = num_buffers * sizeof(BufferSizeT);
    allocation_sizes[MEM_BLEV_BUFFER_TBLOCK] = num_buffers * sizeof(BlockOffsetT);
    CubDebug(
      error = BLevBufferOffsetTileState::AllocationSize(num_tiles, allocation_sizes[MEM_BLEV_BUFFER_SCAN_STATE]));
    if (error)
    {
      return error;
    }
    CubDebug(error = BLevBlockOffsetTileState::AllocationSize(num_tiles, allocation_sizes[MEM_BLEV_BLOCK_SCAN_STATE]));
    if (error)
    {
      return error;
    }

    // Compute allocation pointers into the single storage BLOB (or compute the necessary size of the blob)
    AliasTemporaries<MEM_NUM_ALLOCATIONS>(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes);

    // Just return if no temporary storage is provided
    if (d_temp_storage == nullptr)
    {
      return error;
    }

    // Return if empty problem
    if (num_buffers == 0)
    {
      return error;
    }

    // Alias into temporary storage allocation
    BlevBufferSrcsOutItT d_blev_src_buffers = reinterpret_cast<BlevBufferSrcsOutItT>(allocations[MEM_BLEV_BUFFER_SRCS]);
    BlevBufferDstsOutItT d_blev_dst_buffers = reinterpret_cast<BlevBufferDstsOutItT>(allocations[MEM_BLEV_BUFFER_DSTS]);
    BlevBufferSizesOutItT d_blev_buffer_sizes =
      reinterpret_cast<BlevBufferSizesOutItT>(allocations[MEM_BLEV_BUFFER_SIZES]);
    BlevBufferTileOffsetsOutItT d_blev_block_offsets =
      reinterpret_cast<BlevBufferTileOffsetsOutItT>(allocations[MEM_BLEV_BUFFER_TBLOCK]);

    // Kernels' grid sizes
    uint32_t init_grid_size         = cub::DivideAndRoundUp(num_tiles, INIT_KERNEL_THREADS);
    uint32_t scan_grid_size         = cub::DivideAndRoundUp(num_tiles, INIT_KERNEL_THREADS);
    uint32_t batch_memcpy_grid_size = num_tiles;

    // Kernels
    auto init_buffer_scan_states_kernel = InitTileStateKernel<BLevBufferOffsetTileState, BlockOffsetT>;
    auto init_block_scan_states_kernel  = InitTileStateKernel<BLevBlockOffsetTileState, BlockOffsetT>;
    auto bach_memcpy_non_blev_kernel    = BatchMemcpyKernel<MaxPolicyT,
                                                         InputBufferIt,
                                                         OutputBufferIt,
                                                         BufferSizeIteratorT,
                                                         BufferOffsetT,
                                                         BlevBufferSrcsOutItT,
                                                         BlevBufferDstsOutItT,
                                                         BlevBufferSizesOutItT,
                                                         BlevBufferTileOffsetsOutItT,
                                                         BlockOffsetT,
                                                         BLevBufferOffsetTileState,
                                                         BLevBlockOffsetTileState>;

    auto multi_block_memcpy_kernel = MultiBlockBatchMemcpyKernel<MaxPolicyT,
                                                                 BufferOffsetT,
                                                                 BlevBufferSrcsOutItT,
                                                                 BlevBufferDstsOutItT,
                                                                 BlevBufferSizesOutItT,
                                                                 BlevBufferTileOffsetsOutItT,
                                                                 BLevBufferOffsetTileState,
                                                                 BlockOffsetT>;

    constexpr uint32_t BLEV_BLOCK_THREADS = ActivePolicyT::AgentLargeBufferPolicyT::BLOCK_THREADS;

    // Get device ordinal
    int device_ordinal;
    if (CubDebug(error = cudaGetDevice(&device_ordinal)))
      return error;

    // Get SM count
    int sm_count;
    if (CubDebug(error = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal)))
      return error;

    // Get SM occupancy for the batch memcpy block-level buffers kernel
    int batch_memcpy_blev_occupancy;
    if (CubDebug(error = MaxSmOccupancy(batch_memcpy_blev_occupancy, multi_block_memcpy_kernel, BLEV_BLOCK_THREADS)))
      return error;

    uint32_t batch_memcpy_blev_grid_size = sm_count * batch_memcpy_blev_occupancy;

    // Construct the tile status for the buffer prefix sum
    BLevBufferOffsetTileState buffer_scan_tile_state;
    if (CubDebug(error = buffer_scan_tile_state.Init(num_tiles,
                                                     allocations[MEM_BLEV_BUFFER_SCAN_STATE],
                                                     allocation_sizes[MEM_BLEV_BUFFER_SCAN_STATE])))
    {
      return error;
    }

    // Construct the tile status for thread blocks-to-buffer-assignment prefix sum
    BLevBlockOffsetTileState block_scan_tile_state;
    if (CubDebug(error = block_scan_tile_state.Init(num_tiles,
                                                    allocations[MEM_BLEV_BLOCK_SCAN_STATE],
                                                    allocation_sizes[MEM_BLEV_BLOCK_SCAN_STATE])))
    {
      return error;
    }

    // Invoke init_kernel to initialize buffer prefix sum-tile descriptors
    thrust::cuda_cub::launcher::triple_chevron(init_grid_size, INIT_KERNEL_THREADS, 0, stream)
      .doit(init_buffer_scan_states_kernel, buffer_scan_tile_state, num_tiles);

    // Check for failure to launch
    if (CubDebug(error = cudaPeekAtLastError()))
      return error;

    // Sync the stream if specified to flush runtime errors
    if (debug_synchronous && (CubDebug(error = SyncStream(stream))))
      return error;

    // Invoke init_kernel to initialize thread blocks-to-buffer-assignment prefix sum-tile descriptors
    thrust::cuda_cub::launcher::triple_chevron(scan_grid_size, INIT_KERNEL_THREADS, 0, stream)
      .doit(init_block_scan_states_kernel, block_scan_tile_state, num_tiles);

    // Check for failure to launch
    if (CubDebug(error = cudaPeekAtLastError()))
      return error;

    // Sync the stream if specified to flush runtime errors
    if (debug_synchronous && (CubDebug(error = SyncStream(stream))))
      return error;

    // Invoke kernel to copy small buffers and put the larger ones into a queue that will get picked up by next kernel
    thrust::cuda_cub::launcher::triple_chevron(batch_memcpy_grid_size,
                                               ActivePolicyT::AgentSmallBufferPolicyT::BLOCK_THREADS,
                                               0,
                                               stream)
      .doit(bach_memcpy_non_blev_kernel,
            input_buffer_it,
            output_buffer_it,
            buffer_sizes,
            num_buffers,
            d_blev_src_buffers,
            d_blev_dst_buffers,
            d_blev_buffer_sizes,
            d_blev_block_offsets,
            buffer_scan_tile_state,
            block_scan_tile_state);

    thrust::cuda_cub::launcher::triple_chevron(batch_memcpy_blev_grid_size, BLEV_BLOCK_THREADS, 0, stream)
      .doit(multi_block_memcpy_kernel,
            d_blev_src_buffers,
            d_blev_dst_buffers,
            d_blev_buffer_sizes,
            d_blev_block_offsets,
            buffer_scan_tile_state,
            batch_memcpy_grid_size - 1);

    // Check for failure to launch
    if (CubDebug(error = cudaPeekAtLastError()))
      return error;

    // Sync the stream if specified to flush runtime errors
    if (debug_synchronous && (CubDebug(error = SyncStream(stream))))
      return error;

    return error;
  }

  //------------------------------------------------------------------------------
  // Dispatch entrypoints
  //------------------------------------------------------------------------------
  /**
   * Internal dispatch routine
   */
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t Dispatch(void *d_temp_storage,
                                                                   size_t &temp_storage_bytes,
                                                                   InputBufferIt input_buffer_it,
                                                                   OutputBufferIt output_buffer_it,
                                                                   BufferSizeIteratorT buffer_sizes,
                                                                   BufferOffsetT num_buffers,
                                                                   cudaStream_t stream,
                                                                   bool debug_synchronous)
  {
#if false
// #ifndef CUB_RUNTIME_ENABLED
    (void)d_temp_storage;
    (void)temp_storage_bytes;
    (void)input_buffer_it;
    (void)output_buffer_it;
    (void)buffer_sizes;
    (void)num_buffers;
    (void)stream;
    (void)debug_synchronous;

    // Kernel launch not supported from this device
    return CubDebug(cudaErrorNotSupported);
#else
    using MaxPolicyT = typename DispatchBatchMemcpy::MaxPolicy;

    cudaError_t error = cudaSuccess;

    // Get PTX version
    int ptx_version = 0;
    if (CubDebug(error = PtxVersion(ptx_version)))
    {
      return error;
    }

    // Create dispatch functor
    DispatchBatchMemcpy dispatch(d_temp_storage,
                                 temp_storage_bytes,
                                 input_buffer_it,
                                 output_buffer_it,
                                 buffer_sizes,
                                 num_buffers,
                                 stream,
                                 debug_synchronous,
                                 ptx_version);

    // Dispatch to chained policy
    if (CubDebug(error = MaxPolicyT::Invoke(ptx_version, dispatch)))
    {
      return error;
    }
    return error;
#endif
  }
};

CUB_NAMESPACE_END