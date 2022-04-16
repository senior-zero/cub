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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/agent/agent_for.cuh>
#include <cub/config.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_namespace.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

CUB_NAMESPACE_BEGIN

namespace detail
{

template <int PtxArch>
struct ptx_arch : std::integral_constant<int, PtxArch>
{};

} // namespace detail

using SM350 = detail::ptx_arch<350>;
using SM370 = detail::ptx_arch<370>;
using SM500 = detail::ptx_arch<500>;
using SM520 = detail::ptx_arch<520>;
using SM530 = detail::ptx_arch<530>;
using SM600 = detail::ptx_arch<600>;
using SM610 = detail::ptx_arch<610>;
using SM620 = detail::ptx_arch<620>;
using SM700 = detail::ptx_arch<700>;
using SM750 = detail::ptx_arch<750>;
using SM800 = detail::ptx_arch<800>;
using SM860 = detail::ptx_arch<860>;

// Starting points for architectures
using Kepler  = SM350;
using Maxwell = SM500;
using Pascal  = SM600;
using Volta   = SM700;
using Turing  = SM750;
using Ampere  = SM800;

namespace detail
{

namespace for_each
{

// TODO Different settings for different PTX versions?

template <typename V, typename T, typename = T>
struct has_unique_value_overload_func_impl : std::false_type
{};

template <typename V, typename T>
struct has_unique_value_overload_func_impl<
  V,
  T,
  std::enable_if_t<std::is_same<T, void(V)>::value, T>> : std::true_type
{};

template <typename V, typename T, typename = T>
struct has_unique_value_overload_class_impl : std::false_type
{};

template <typename V, typename T>
struct has_unique_value_overload_class_impl<
  V,
  T,
  std::enable_if_t<std::is_same<decltype(&T::operator()), void (T::*)(V)>::value,
                   T>> : std::true_type
{};

template <typename V, typename T>
struct has_unique_value_overload_class_impl<
  V,
  T,
  std::enable_if_t<
    std::is_same<decltype(&T::operator()), void (T::*)(V) const>::value,
    T>> : std::true_type
{};

template <typename V, typename T>
struct has_unique_value_overload_class_impl<
  V,
  T,
  std::enable_if_t<
    std::is_same<decltype(&T::operator()), void (T::*)(V) volatile>::value,
    T>> : std::true_type
{};

template <typename V, typename T>
using has_unique_value_overload =
  std::conditional_t<std::is_class<T>::value,
                     has_unique_value_overload_class_impl<V, T>,
                     has_unique_value_overload_func_impl<V, T>>;

template <typename OffsetT,
          typename OpT,
          OffsetT BLOCK_THREADS,
          OffsetT ITEMS_PER_THREAD>
__global__ __launch_bounds__(BLOCK_THREADS) void kernel(OffsetT num_items,
                                                        OpT op)
{
  constexpr OffsetT ITEMS_PER_TILE = ITEMS_PER_THREAD * BLOCK_THREADS;

  const auto tile_base     = static_cast<OffsetT>(blockIdx.x) * ITEMS_PER_TILE;
  const auto num_remaining = num_items - tile_base;
  const auto items_in_tile = static_cast<OffsetT>(
    num_remaining < ITEMS_PER_TILE ? num_remaining : ITEMS_PER_TILE);

  using agent_t =
    agent_block_striped<OffsetT, OpT, BLOCK_THREADS, ITEMS_PER_THREAD>;

  if (items_in_tile == ITEMS_PER_TILE)
  {
    // full tile
    agent_t(tile_base, op).consume_tile<true>(ITEMS_PER_TILE);
  }
  else
  {
    // partial tile
    agent_t(tile_base, op).consume_tile<false>(items_in_tile);
  }
}

template <unsigned BLOCK_THREADS, unsigned ITEMS>
struct configuration
{
  constexpr static unsigned block_threads    = BLOCK_THREADS;
  constexpr static unsigned items_per_thread = ITEMS;
};

template <typename... Head>
struct configurations
{
  template <unsigned BLOCK_THREADS, unsigned ITEMS>
  configurations<Head..., configuration<BLOCK_THREADS, ITEMS>> Add()
  {
    return {};
  }
};

template <typename OffsetT, typename OpT>
int configuration_space_search(configurations<>)
{
  return 0;
}

template <typename OffsetT, typename OpT, typename Head, typename... Tail>
int configuration_space_search(configurations<Head, Tail...>)
{
  constexpr OffsetT block_threads    = Head::block_threads;
  constexpr OffsetT items_per_thread = Head::items_per_thread;

  int num_blocks{};

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks,
    kernel<OffsetT, OpT, block_threads, items_per_thread>,
    block_threads,
    0);

  const int num_warps = num_blocks * (block_threads / 32);

  // printf("occupancy[%d] = %d\n", block_threads, num_warps);
  return std::max(num_warps,
                  configuration_space_search<OffsetT, OpT>(
                    configurations<Tail...>{}));
}

template <typename OffsetT, typename OpT>
cudaError_t
configuration_launch(int, OffsetT, OpT, bool, cudaStream_t, configurations<>)
{
  return cudaErrorUnknown;
}

template <typename OffsetT, typename OpT, typename Head, typename... Tail>
cudaError_t configuration_launch(int target_occupancy,
                                 OffsetT num_items,
                                 OpT op,
                                 bool debug_synchronous,
                                 cudaStream_t stream,
                                 configurations<Head, Tail...>)
{
  constexpr OffsetT block_threads    = Head::block_threads;
  constexpr OffsetT items_per_thread = Head::items_per_thread;

  int num_blocks{};

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks,
    kernel<OffsetT, OpT, block_threads, items_per_thread>,
    block_threads,
    0);

  const int num_warps = num_blocks * (block_threads / 32);

  if (num_warps == target_occupancy)
  {

    constexpr OffsetT tile_size = block_threads * items_per_thread;
    const OffsetT num_tiles     = cub::DivideAndRoundUp(num_items, tile_size);

    // Log single_tile_kernel configuration
    if (debug_synchronous)
    {
      _CubLog("Invoking detail::for_each::kernel<<<%d, %d, 0, %lld>>>(), "
              "%d items per thread\n",
              static_cast<int>(num_tiles),
              static_cast<int>(block_threads),
              reinterpret_cast<long long>(stream),
              static_cast<int>(items_per_thread));
    }

    cudaError_t error =
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        static_cast<unsigned int>(num_tiles),
        static_cast<unsigned int>(block_threads),
        0,
        stream)
        .doit(
          detail::for_each::kernel<OffsetT, OpT, block_threads, items_per_thread>,
          num_items,
          op);

    if (debug_synchronous)
    {
      CubDebug(error = SyncStream(stream));
    }

    return error;
  }

  return configuration_launch<OffsetT, OpT>(target_occupancy,
                                            num_items,
                                            op,
                                            debug_synchronous,
                                            stream,
                                            configurations<Tail...>{});
}

template <typename OffsetT, typename OpT, typename Tuning>
struct dispatch_t
{
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
  dispatch(OffsetT num_items,
           OpT op,
           cudaStream_t stream,
           bool debug_synchronous)
  {
    if (num_items == 0)
    {
      return cudaSuccess;
    }

    typename Tuning::CONFIGURATION_SPACE configuration_space{};

    const int target_occupancy =
      detail::for_each::configuration_space_search<OffsetT, OpT>(
        configuration_space);

    return configuration_launch(target_occupancy,
                                num_items,
                                op,
                                debug_synchronous,
                                stream,
                                configuration_space);
  }
};

} // namespace for_each

} // namespace detail

CUB_NAMESPACE_END
