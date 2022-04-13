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

#include <cub/config.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_namespace.cuh>
#include <cub/agent/agent_for.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

CUB_NAMESPACE_BEGIN

namespace detail
{

// TODO Different settings for different PTX versions?

template <typename OffsetT,
          typename OpT,
          OffsetT BLOCK_THREADS,
          OffsetT ITEMS_PER_THREAD>
__global__ __launch_bounds__(BLOCK_THREADS)
void device_for_each_kernel(OffsetT num_items, OpT op)
{
  constexpr OffsetT ITEMS_PER_TILE = ITEMS_PER_THREAD * BLOCK_THREADS;

  const auto tile_base     = static_cast<OffsetT>(blockIdx.x) * ITEMS_PER_TILE;
  const auto num_remaining = num_items - tile_base;
  const auto items_in_tile = static_cast<OffsetT>(
    num_remaining < ITEMS_PER_TILE ? num_remaining : ITEMS_PER_TILE);

  using AgentT = AgentForBlockStriped<OffsetT, OpT, BLOCK_THREADS, ITEMS_PER_THREAD>;

  if (items_in_tile == ITEMS_PER_TILE)
  {
    // full tile
    AgentT(tile_base, op).ConsumeTile<true>(ITEMS_PER_TILE);
  }
  else
  {
    // partial tile
    AgentT(tile_base, op).ConsumeTile<false>(items_in_tile);
  }
}

template <unsigned BLOCK_THREADS, unsigned ITEMS>
struct for_each_configuration
{
  constexpr static unsigned block_threads = BLOCK_THREADS;
  constexpr static unsigned items_per_thread = ITEMS;
};

template <typename... Head>
struct for_each_configurations
{
  template <unsigned BLOCK_THREADS, unsigned ITEMS>
  for_each_configurations<Head..., for_each_configuration<BLOCK_THREADS, ITEMS>> Add()
  {
    return {};
  }
};

template <typename OffsetT, typename OpT>
int for_each_configuration_space_search(for_each_configurations<>)
{
  return 0;
}

template <typename OffsetT, typename OpT, typename Head, typename... Tail>
int for_each_configuration_space_search(for_each_configurations<Head, Tail...>)
{
  constexpr OffsetT block_threads = Head::block_threads;
  constexpr OffsetT items_per_thread = Head::items_per_thread;

  int num_blocks {};

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks,
    detail::device_for_each_kernel<OffsetT, OpT, block_threads, items_per_thread>,
    block_threads,
    0);

  const int num_warps = num_blocks * (block_threads / 32);

  printf("occupancy[%d] = %d\n", block_threads, num_warps);
  return std::max(num_warps,
                  for_each_configuration_space_search<OffsetT, OpT>(
                    for_each_configurations<Tail...>{}));
}

template <typename OffsetT, typename OpT>
cudaError_t for_each_configuration_launch(int, OffsetT, OpT, bool, cudaStream_t, for_each_configurations<>)
{
  return cudaErrorUnknown;
}

template <typename OffsetT, typename OpT, typename Head, typename... Tail>
cudaError_t for_each_configuration_launch(
  int target_occupancy, OffsetT num_items, OpT op, bool debug_synchronous, cudaStream_t stream, for_each_configurations<Head, Tail...>)
{
  constexpr OffsetT block_threads = Head::block_threads;
  constexpr OffsetT items_per_thread = Head::items_per_thread;

  int num_blocks {};

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks,
    detail::device_for_each_kernel<OffsetT, OpT, block_threads, items_per_thread>,
    block_threads,
    0);

  const int num_warps = num_blocks * (block_threads / 32);

  if (num_warps == target_occupancy)
  {

    constexpr OffsetT tile_size = block_threads * items_per_thread;
    const OffsetT num_tiles = cub::DivideAndRoundUp(num_items, tile_size);

    // Log single_tile_kernel configuration
    if (debug_synchronous)
    {
      _CubLog("Invoking detail::device_for_each_kernel<<<%d, %d, 0, %lld>>>(), "
              "%d items per thread\n",
              static_cast<int>(num_tiles),
              static_cast<int>(block_threads),
              reinterpret_cast<long long>(stream),
              items_per_thread);
    }

    cudaError_t error =
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(num_tiles,
                                                              block_threads,
                                                              0,
                                                              stream)
        .doit(detail::device_for_each_kernel<OffsetT, OpT, block_threads, items_per_thread>,
              num_items,
              op);

      if (debug_synchronous)
      {
        CubDebug(error = SyncStream(stream));
      }

      return error;
  }

  return for_each_configuration_launch<OffsetT, OpT>(
    target_occupancy,
    num_items,
    op,
    debug_synchronous,
    stream,
    for_each_configurations<Tail...>{});
}

}

using ForEachConfigurationSpace = detail::for_each_configurations<>;

enum class ForEachAlgorithm
{
  BLOCK_STRIPED,
  VECTORIZED
};

template <ForEachAlgorithm Algorithm,
          typename... Configurations>
struct ForEachTuning
{
  constexpr static ForEachAlgorithm algorithm = Algorithm;

  using configuration_space =
    detail::for_each_configurations<Configurations...>;
};

template <ForEachAlgorithm Algorithm, typename... Configurations>
ForEachTuning<Algorithm, Configurations...>
  TuneForEach(detail::for_each_configurations<Configurations...>)
{
  return {};
}

using ForEachDefaultTuning = decltype(TuneForEach<ForEachAlgorithm::BLOCK_STRIPED>(
  ForEachConfigurationSpace{}.Add<256, 2>()));

template <typename OffsetT,
          typename OpT,
          typename Tuning = ForEachDefaultTuning>
struct DispatchFor
{
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  Dispatch(OffsetT num_items,
           OpT op,
           cudaStream_t stream,
           bool debug_synchronous)
  {
    if (num_items == 0)
    {
      return cudaSuccess;
    }

    typename Tuning::configuration_space configuration_space{};

    const int target_occupancy =
      detail::for_each_configuration_space_search<OffsetT, OpT>(
        configuration_space);

    return for_each_configuration_launch(target_occupancy,
                                         num_items,
                                         op,
                                         debug_synchronous,
                                         stream,
                                         configuration_space);
  }
};

CUB_NAMESPACE_END
