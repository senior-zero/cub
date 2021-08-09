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

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iterator>

#include <cub/warp/warp_load.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/iterator/cache_modified_output_iterator.cuh>
#include <cub/iterator/discard_output_iterator.cuh>
#include <cub/util_allocator.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include "test_util.h"

using namespace cub;

template <int                 BlockThreads,
          int                 WarpThreads,
          int                 ItemsPerThread,
          WarpLoadAlgorithm   LoadAlgorithm,
          typename            InputIteratorT>
__global__ void kernel(InputIteratorT input,
                       int *err)
{
  using InputT = typename std::iterator_traits<InputIteratorT>::value_type;

  using WarpLoadT = WarpLoad<InputT,
                             ItemsPerThread,
                             LoadAlgorithm,
                             WarpThreads>;

  constexpr int warps_in_block = BlockThreads / WarpThreads;
  const int warp_id = static_cast<int>(threadIdx.x) / WarpThreads;

  __shared__
    typename WarpLoadT::TempStorage temp_storage[warps_in_block];

  InputT reg[ItemsPerThread];
  WarpLoadT(temp_storage[warp_id]).Load(input, reg);

  for (int item = 0; item < ItemsPerThread; item++)
  {
    const auto expected_value =
      static_cast<InputT>(threadIdx.x * ItemsPerThread + item);

    if (reg[item] != expected_value)
    {
      atomicAdd(err, 1);
    }
  }
}

template <typename            T,
          int                 BlockThreads,
          int                 WarpThreads,
          int                 ItemsPerThread,
          WarpLoadAlgorithm   LoadAlgorithm>
void TestPointer()
{
  const int tile_size = WarpThreads * ItemsPerThread;
  const int total_warps = BlockThreads / WarpThreads;
  const int elements = total_warps * tile_size;

  thrust::device_vector<int> err(1);
  thrust::device_vector<T> input(elements);
  thrust::sequence(input.begin(), input.end());

  kernel<BlockThreads, WarpThreads, ItemsPerThread, LoadAlgorithm>
    <<<1, BlockThreads>>>(thrust::raw_pointer_cast(input.data()),
                          thrust::raw_pointer_cast(err.data()));
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());

  AssertEquals(input[0], 0);
}

template <typename            T,
          int                 BlockThreads,
          int                 WarpThreads,
          int                 ItemsPerThread,
          WarpLoadAlgorithm   LoadAlgorithm,
          CacheLoadModifier   LoadModifier>
void TestIterator()
{
  const int tile_size = WarpThreads * ItemsPerThread;
  const int total_warps = BlockThreads / WarpThreads;
  const int elements = total_warps * tile_size;

  thrust::device_vector<int> err(1);
  thrust::device_vector<T> input(elements);
  thrust::sequence(input.begin(), input.end());

  kernel<BlockThreads, WarpThreads, ItemsPerThread, LoadAlgorithm>
    <<<1, BlockThreads>>>(CacheModifiedInputIterator<LoadModifier, T>(
                            thrust::raw_pointer_cast(input.data())),
                          thrust::raw_pointer_cast(err.data()));
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());

  AssertEquals(input[0], 0);
}

template <typename            T,
          int                 BlockThreads,
          int                 WarpThreads,
          int                 ItemsPerThread,
          WarpLoadAlgorithm   LoadAlgorithm>
void TestIterator()
{
  TestIterator<T, BlockThreads, WarpThreads, ItemsPerThread, LoadAlgorithm, CacheLoadModifier::LOAD_DEFAULT>();
  TestIterator<T, BlockThreads, WarpThreads, ItemsPerThread, LoadAlgorithm, CacheLoadModifier::LOAD_CA>();
  TestIterator<T, BlockThreads, WarpThreads, ItemsPerThread, LoadAlgorithm, CacheLoadModifier::LOAD_CG>();
  TestIterator<T, BlockThreads, WarpThreads, ItemsPerThread, LoadAlgorithm, CacheLoadModifier::LOAD_CS>();
  TestIterator<T, BlockThreads, WarpThreads, ItemsPerThread, LoadAlgorithm, CacheLoadModifier::LOAD_CV>();
  TestIterator<T, BlockThreads, WarpThreads, ItemsPerThread, LoadAlgorithm, CacheLoadModifier::LOAD_LDG>();
  TestIterator<T, BlockThreads, WarpThreads, ItemsPerThread, LoadAlgorithm, CacheLoadModifier::LOAD_VOLATILE>();
}


template <typename            T,
          int                 BlockThreads,
          int                 WarpThreads,
          int                 ItemsPerThread,
          WarpLoadAlgorithm   LoadAlgorithm>
void Test()
{
  TestPointer<T, BlockThreads, WarpThreads, ItemsPerThread, LoadAlgorithm>();
  TestIterator<T, BlockThreads, WarpThreads, ItemsPerThread, LoadAlgorithm>();
}

template <typename  T,
          int       BlockThreads,
          int       WarpThreads,
          int       ItemsPerThread>
void Test()
{
  Test<T, BlockThreads, WarpThreads, ItemsPerThread, WarpLoadAlgorithm::WARP_LOAD_DIRECT>();
  Test<T, BlockThreads, WarpThreads, ItemsPerThread, WarpLoadAlgorithm::WARP_LOAD_STRIPED>();
  Test<T, BlockThreads, WarpThreads, ItemsPerThread, WarpLoadAlgorithm::WARP_LOAD_TRANSPOSE>();
  Test<T, BlockThreads, WarpThreads, ItemsPerThread, WarpLoadAlgorithm::WARP_LOAD_VECTORIZE>();
}


template <typename T,
          int      BlockThreads,
          int      WarpThreads>
void Test()
{
  Test<T, BlockThreads, WarpThreads, 1>();
  Test<T, BlockThreads, WarpThreads, 4>();
  Test<T, BlockThreads, WarpThreads, 7>();
}

template <typename T,
          int BlockThreads>
void Test()
{
  Test<T, BlockThreads, 4>();
  Test<T, BlockThreads, 16>();
  Test<T, BlockThreads, 32>();
}

template <int BlockThreads>
void Test()
{
  Test<std::uint16_t, BlockThreads>();
  Test<std::uint32_t, BlockThreads>();
  Test<std::uint64_t, BlockThreads>();
}

int main(int argc, char** argv)
{
  // Initialize command line
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  Test<256>();
}



