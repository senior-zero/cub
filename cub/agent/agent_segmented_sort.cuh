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

#include "../config.cuh"
#include "../util_type.cuh"
#include "../util_namespace.cuh"


CUB_NAMESPACE_BEGIN


template <
  int                      BLOCK_THREADS_ARG,
  int                      ITEMS_PER_THREAD_ARG,
  int                      THREADS_PER_MEDIUM_SEGMENT_ARG,
  int                      THREADS_PER_SMALL_SEGMENT_ARG>
struct AgentSmallAndMediumSegmentedSortPolicy
{
  static constexpr int BLOCK_THREADS    = BLOCK_THREADS_ARG;
  constexpr static int ITEMS_PER_THREAD = ITEMS_PER_THREAD_ARG;

  constexpr static int THREADS_PER_MEDIUM_SEGMENT =
    THREADS_PER_MEDIUM_SEGMENT_ARG;
  constexpr static int SEGMENTS_PER_MEDIUM_BLOCK = BLOCK_THREADS /
                                                   THREADS_PER_MEDIUM_SEGMENT;
  constexpr static int MEDIUM_SEGMENT_MAX_SIZE = ITEMS_PER_THREAD *
                                                 THREADS_PER_MEDIUM_SEGMENT;

  constexpr static int THREADS_PER_SMALL_SEGMENT =
    THREADS_PER_SMALL_SEGMENT_ARG;
  constexpr static int SEGMENTS_PER_SMALL_BLOCK = BLOCK_THREADS /
                                                  THREADS_PER_SMALL_SEGMENT;
  constexpr static int SMALL_SEGMENT_MAX_SIZE = ITEMS_PER_THREAD *
                                                THREADS_PER_SMALL_SEGMENT;
};


CUB_NAMESPACE_END