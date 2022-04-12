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
#include <cub/util_type.cuh>


CUB_NAMESPACE_BEGIN

template <typename OffsetT,
          typename OpT,
          OffsetT BLOCK_THREADS,
          OffsetT ITEMS_PER_THREAD>
class AgentFor
{
  OffsetT tile_base;
  OpT op;

public:
  __device__ __forceinline__
  AgentFor(OffsetT tile_base, OpT op)
    : tile_base(tile_base)
    , op(op)
  {}

  template <bool IS_FULL_TILE>
  __device__ __forceinline__ void ConsumeTile(int items_in_tile)
  {
    #pragma unroll
    for (OffsetT item = 0; item < ITEMS_PER_THREAD; ++item)
    {
      OffsetT idx = BLOCK_THREADS * item + threadIdx.x;

      if (IS_FULL_TILE || idx < items_in_tile)
        op(tile_base + idx);
    }
  }
};


CUB_NAMESPACE_END
