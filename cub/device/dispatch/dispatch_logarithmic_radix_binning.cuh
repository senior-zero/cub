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

#include "../../config.cuh"
#include "../../util_device.cuh"
#include "../../block/block_scan.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

__device__ unsigned char clz(std::uint32_t val)
{
  return __clz(val);
}

__device__ unsigned char clz(int val)
{
  return __clz(val);
}

__device__ unsigned char clz(std::uint64_t val)
{
  return __clzll(val);
}


// TODO Extract agent
template <
  int BinsCount,
  typename SegmentHandlerT>
__global__ void BinCountKernel(const unsigned int num_segments,
                               int *bins,
                               SegmentHandlerT segment_handler)
{
  // thread per segment
  const unsigned int tid        = threadIdx.x;
  const unsigned int segment_id = tid + blockIdx.x * blockDim.x;

  __shared__ int bins_cache[BinsCount];

  if (tid < BinsCount)
  {
    bins_cache[tid] = 0;
  }
  __syncthreads();

  if (segment_id < num_segments)
  {
    const auto segment_size = segment_handler.get_segment_size(segment_id);
    const int bin_id        = clz(segment_size);

    atomicAdd(bins_cache + bin_id, 1);
  }
  __syncthreads();

  if (tid < BinsCount)
  {
    atomicAdd(bins + tid, bins_cache[tid]);
  }
}

// TODO Extract agent
template <int BinsCount,
  int BlockSize>
__global__ void BinsPrefixKernel(const int *bins, int *bins_prefix, int *bins_prefix_copy)
{
  using BlockScan = cub::BlockScan<int, BlockSize>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  const int tid = static_cast<int>(threadIdx.x);

  int thread_data = tid < BinsCount ? bins[tid] : 0u;

  BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);

  if (tid <= BinsCount)
  {
    bins_prefix[tid] = thread_data;
    bins_prefix_copy[tid] = thread_data;
  }
}

// TODO Extract agent
template <int BinsCount,
  typename SegmentHandlerT>
__global__ void BalanceSegments(
  const int num_segments,
  int *bins_prefix,
  SegmentHandlerT segment_handler)
{
  const int tid = static_cast<int>(threadIdx.x);
  const int segment_id = tid + static_cast<int>(blockDim.x * blockIdx.x);

  __shared__ int local_bins[BinsCount];
  __shared__ int local_pos[BinsCount];

  if (tid < BinsCount)
  {
    local_bins[tid] = 0;
    local_pos[tid] = 0;
  }
  __syncthreads();

  int32_t bin_id{};
  int32_t my_pos{};

  if (segment_id < num_segments)
  {
    const auto segment_size = segment_handler.get_segment_size(segment_id);

    bin_id = clz(segment_size);
    my_pos = atomicAdd(local_bins + bin_id, 1);
  }
  __syncthreads();

  if (tid < BinsCount)
  {
    local_pos[tid] = atomicAdd(bins_prefix + tid, local_bins[tid]);
  }
  __syncthreads();

  if (segment_id < num_segments)
  {
    const int32_t pos = local_pos[bin_id] + my_pos;
    segment_handler.set_balanced_position(segment_id, pos);
  }
}

struct LogarithmicRadixBinningResult
{
  LogarithmicRadixBinningResult()
    : bins_count(0)
    , d_bins(nullptr)
    , d_bins_prefix(nullptr)
  {}

  int bins_count;
  int *d_bins;
  int *d_bins_prefix;
};

template <typename SegmentHandlerT>
struct DispatchLogarithmicRadixBinning
{
  static constexpr int BITS_IN_COUNTER =
    sizeof(typename SegmentHandlerT::offset_size_type) * 8 + 1;
  static constexpr int BINS_SIZE   = BITS_IN_COUNTER;
  static constexpr int PREFIX_SIZE = BINS_SIZE + 1;
  static constexpr int WARP_SIZE   = 32;
  static constexpr int SCAN_BLOCK_SIZE =
    ((PREFIX_SIZE + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  static cudaError_t Dispatch(int num_segments,
                              SegmentHandlerT segment_handler,
                              void *d_storage,
                              size_t &storage_bytes,
                              LogarithmicRadixBinningResult &result)
  {
    cudaError_t status = cudaSuccess;

    do
    {
      void*  allocations[3]      = {nullptr, nullptr, nullptr};
      size_t allocation_sizes[3] = {BINS_SIZE * sizeof(int),
                                    PREFIX_SIZE * sizeof(int),
                                    PREFIX_SIZE * sizeof(int)};

      if (CubDebug(status = AliasTemporaries(d_storage,
                                             storage_bytes,
                                             allocations,
                                             allocation_sizes)))
      {
        break;
      }

      if (d_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage
        // allocation
        break;
      }

      result.bins_count    = BINS_SIZE;
      result.d_bins        = reinterpret_cast<int *>(allocations[0]);
      result.d_bins_prefix = reinterpret_cast<int *>(allocations[1]);

      int *d_counts = reinterpret_cast<int *>(allocations[2]);
      cudaMemsetAsync(result.d_bins, 0, sizeof(int) * BINS_SIZE);

      // TODO Extract into policy
      const unsigned int threads_in_block = 256;
      const unsigned int blocks_in_grid =
        (num_segments + threads_in_block - 1) / threads_in_block;

      // TODO Use thrust launch
      BinCountKernel<BINS_SIZE><<<blocks_in_grid, threads_in_block>>>(
        num_segments, result.d_bins, segment_handler);
      // TODO Check launch failure and sync

      // TODO Use thrust launch
      BinsPrefixKernel<BINS_SIZE, SCAN_BLOCK_SIZE><<<1, SCAN_BLOCK_SIZE>>>(
        result.d_bins, result.d_bins_prefix, d_counts);
      // TODO Check launch failure and sync

      // TODO Use thrust launch
      BalanceSegments<BINS_SIZE><<<blocks_in_grid, threads_in_block>>>(
        num_segments, d_counts, segment_handler);
      // TODO Check launch failure and sync
    } while (false);

    return status;
  }
};

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
