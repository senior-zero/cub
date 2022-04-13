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
#include <cub/util_namespace.cuh>
#include <cub/device/dispatch/dispatch_for.cuh>

#include <thrust/detail/raw_reference_cast.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/distance.h>

#include <type_traits>

CUB_NAMESPACE_BEGIN

namespace detail
{

// TODO CacheLoadModifier

template <typename OffsetT, typename OpT, typename InputIteratorT>
struct ForEachWrapper
{
  OpT op;
  InputIteratorT input;

  __device__ void operator()(OffsetT i)
  {
    op(THRUST_NS_QUALIFIER::raw_reference_cast(input[i]));
  }
};

template <typename OffsetT, typename OpT, typename T>
struct ForEachWrapperVectorized
{
  OpT op;
  const T* in;
  OffsetT partially_filled_vector_id;
  int partially_filled_vector_size;

  constexpr static int vec_size = 4;
  using vector_t = typename CubVector<T, vec_size>::Type;

  __device__ void operator()(OffsetT i)
  {
    if (i != partially_filled_vector_id)
    {
      vector_t vec = *reinterpret_cast<const vector_t*>(in + vec_size * i);

      #pragma unroll
      for (int j = 0; j < vec_size; j++)
      {
        op(*(reinterpret_cast<T*>(&vec) + j));
      }
    }
    else
    {
      for (int j = 0; j < partially_filled_vector_size; j++)
      {
        op(in[i * vec_size + j]);
      }
    }
  }
};

using ForEachDefaultTuning = decltype(TuneForEach<ForEachAlgorithm::BLOCK_STRIPED>(
  ForEachConfigurationSpace{}.Add<256, 2>()));

template <typename V, typename OpT>
using ForEachDefaultTuningSelection =
  decltype(TuneForEach < detail::has_unique_value_overload<V, OpT>::value
             ? ForEachAlgorithm::VECTORIZED
             : ForEachAlgorithm::BLOCK_STRIPED >
                 (ForEachConfigurationSpace{}.Add<256, 2>()));

}

class DeviceFor
{
  template <
    typename InputIteratorT,
    typename OffsetT,
    typename OpT,
    typename Tuning>
  CUB_RUNTIME_FUNCTION static cudaError_t ForEachN(std::integral_constant<int, 0> /* do_not_vectorize */,
                                                   InputIteratorT begin,
                                                   OffsetT num_items,
                                                   OpT op,
                                                   cudaStream_t stream    = {},
                                                   bool debug_synchronous = {},
                                                   Tuning                 = {})
  {
    using wrapped_op_t = detail::ForEachWrapper<OffsetT, OpT, InputIteratorT>;

    return DispatchFor<OffsetT, wrapped_op_t, Tuning>::Dispatch(
      num_items,
      wrapped_op_t{op, begin},
      stream,
      debug_synchronous);
  }

  static unsigned int PowerOf2(std::intptr_t n)
  {
    unsigned count = 0;

    if (n && !(n & (n - 1)))
    {
      return static_cast<unsigned int>(n);
    }

    while(n != 0)
    {
      n >>= 1;
      count += 1;
    }

    return 1 << count;
  }

  template <
    typename InputIteratorT,
    typename OffsetT,
    typename OpT,
    typename Tuning>
  CUB_RUNTIME_FUNCTION static cudaError_t ForEachN(std::integral_constant<int, 1> /* vectorize */,
                                                   InputIteratorT begin,
                                                   OffsetT num_items,
                                                   OpT op,
                                                   cudaStream_t stream    = {},
                                                   bool debug_synchronous = {},
                                                   Tuning tuning          = {})
  {
    auto unwrapped_begin = THRUST_NS_QUALIFIER::raw_pointer_cast(&*begin);
    using wrapped_op_t = detail::ForEachWrapperVectorized<OffsetT, OpT, detail::value_t<InputIteratorT>>;

    if (PowerOf2(reinterpret_cast<std::intptr_t>(unwrapped_begin)) <=
        std::alignment_of<typename wrapped_op_t::vector_t>::value)
    {
      const OffsetT num_vec_items =
        cub::DivideAndRoundUp(num_items, wrapped_op_t::vec_size);

      return DispatchFor<OffsetT, wrapped_op_t, Tuning>::Dispatch(
        num_vec_items,
        wrapped_op_t{op,
                     unwrapped_begin,
                     num_vec_items - 1,
                     static_cast<int>(num_vec_items * 4 - num_items)},
        stream,
        debug_synchronous);
    }

    return ForEachN(std::integral_constant<int, 0>{},
                    begin,
                    num_items,
                    op,
                    stream,
                    debug_synchronous,
                    tuning);
  }

public:
  template <typename OffsetT,
            typename OpT,
            typename Tuning = detail::ForEachDefaultTuning>
  CUB_RUNTIME_FUNCTION static cudaError_t Bulk(OffsetT num_items,
                                               OpT op,
                                               cudaStream_t stream    = {},
                                               bool debug_synchronous = {},
                                               Tuning                 = {})
  {
    return DispatchFor<OffsetT, OpT, Tuning>::Dispatch(num_items,
                                                       op,
                                                       stream,
                                                       debug_synchronous);
  }

  template <
    typename InputIteratorT,
    typename OffsetT,
    typename OpT,
    typename Tuning =
      detail::ForEachDefaultTuningSelection<detail::value_t<InputIteratorT>, OpT>>
  CUB_RUNTIME_FUNCTION static cudaError_t ForEachN(InputIteratorT begin,
                                                   OffsetT num_items,
                                                   OpT op,
                                                   cudaStream_t stream    = {},
                                                   bool debug_synchronous = {},
                                                   Tuning tuning          = {})
  {
    // TODO Check alignment
    constexpr int use_vectorization = (Tuning::algorithm == ForEachAlgorithm::VECTORIZED)
                                   && (THRUST_NS_QUALIFIER::is_contiguous_iterator_v<InputIteratorT>);

    return ForEachN<InputIteratorT, OffsetT, OpT, Tuning>(
      std::integral_constant<int, use_vectorization>{},
      begin,
      num_items,
      op,
      stream,
      debug_synchronous,
      tuning);
  }

  template <
    typename InputIteratorT,
    typename OpT,
    typename Tuning =
      detail::ForEachDefaultTuningSelection<detail::value_t<InputIteratorT>, OpT>>
  CUB_RUNTIME_FUNCTION static cudaError_t ForEach(InputIteratorT begin,
                                                  InputIteratorT end,
                                                  OpT op,
                                                  cudaStream_t stream    = {},
                                                  bool debug_synchronous = {},
                                                  Tuning tuning          = {})
  {
    using OffsetT = typename THRUST_NS_QUALIFIER::iterator_traits<
      InputIteratorT>::difference_type;

    OffsetT num_items =
      static_cast<OffsetT>(THRUST_NS_QUALIFIER::distance(begin, end));

    return ForEachN(begin, num_items, op, stream, debug_synchronous, tuning);
  }
};

CUB_NAMESPACE_END
