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
  const T* __restrict in;

  __device__ void operator()(OffsetT i)
  {
    // TODO Deal with the last item

    constexpr int vec_size = 4;
    using vector_t = typename CubVector<T, vec_size>::Type;

    // TODO thrust::raw_pointer_case
    vector_t vec = *reinterpret_cast<const vector_t*>(in + vec_size * i);

    #pragma unroll
    for (int j = 0; j < vec_size; j++)
    {
      op(*(reinterpret_cast<T*>(&vec) + j));
    }
  }
};

}

struct DeviceFor
{
  template <typename OffsetT,
            typename OpT,
            typename Tuning = ForEachDefaultTuning>
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

  template <typename InputIteratorT,
            typename OffsetT,
            typename OpT,
            typename Tuning = ForEachDefaultTuning>
  CUB_RUNTIME_FUNCTION static cudaError_t ForEachN(InputIteratorT begin,
                                                   OffsetT num_items,
                                                   OpT op,
                                                   cudaStream_t stream    = {},
                                                   bool debug_synchronous = {},
                                                   Tuning                 = {})
  {
    // TODO Check alignment
    constexpr bool use_vectorization = Tuning::algorithm == ForEachAlgorithm::VECTORIZED;
    using wrapped_op_t = std::conditional_t<
      use_vectorization,
      detail::ForEachWrapperVectorized<OffsetT, OpT, detail::value_t<InputIteratorT>>,
      detail::ForEachWrapper<OffsetT, OpT, InputIteratorT>>;

    if (use_vectorization)
    {
      num_items = cub::DivideAndRoundUp(num_items, 4);
    }

    return DispatchFor<OffsetT, wrapped_op_t, Tuning>::Dispatch(
      num_items,
      wrapped_op_t{op, begin},
      stream,
      debug_synchronous);
  }
};

CUB_NAMESPACE_END
