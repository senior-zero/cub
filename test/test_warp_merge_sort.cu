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

/******************************************************************************
 * Test of BlockMergeSort utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <limits>
#include <typeinfo>
#include <memory>

#include <cub/util_allocator.cuh>
#include <cub/warp/warp_merge_sort.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

#include "test_util.h"

using namespace cub;

struct CustomType
{
  std::uint8_t key;
  std::uint64_t count;

  __device__ __host__ CustomType()
    : key(0)
    , count(0)
  {}

  __device__ __host__ CustomType(std::uint64_t value)
    : key(value) // overflow
    , count(value)
  {}

  __device__ __host__ void operator=(std::uint64_t value)
  {
    key = value; // overflow
    count = value;
  }
};


struct CustomLess
{
  template <typename DataType>
  __device__ bool operator()(DataType &lhs, DataType &rhs)
  {
    return lhs < rhs;
  }

  __device__ bool operator()(CustomType &lhs, CustomType &rhs)
  {
    return lhs.key < rhs.key;
  }
};

template <
  typename DataType,
  unsigned int ThreadsInBlock,
  unsigned int ThreadsInWarp,
  unsigned int ItemsPerThread,
  bool Stable = false>
__global__ void WarpMergeSortTestKernel(unsigned int valid_segments,
                                        DataType *data,
                                        const unsigned int *segment_sizes)
{
  using WarpMergeSortT =
    cub::WarpMergeSort<DataType, ItemsPerThread, ThreadsInWarp>;

  constexpr unsigned int WarpsInBlock = ThreadsInBlock / ThreadsInWarp;
  const unsigned int segment_id = threadIdx.x / ThreadsInWarp;

  if (segment_id >= valid_segments)
  {
    // Test case of partially finished CTA
    return;
  }

  __shared__ typename WarpMergeSortT::TempStorage temp_storage[WarpsInBlock];
  WarpMergeSortT warp_sort(temp_storage[segment_id]);

  DataType thread_data[ItemsPerThread];

  const unsigned int thread_offset = ThreadsInWarp * ItemsPerThread * segment_id
                                   + warp_sort.linear_tid * ItemsPerThread;
  const unsigned int valid_items = segment_sizes[segment_id];

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    const unsigned int idx = thread_offset + item;
    thread_data[item] = item < valid_items ? data[idx] : DataType();

  }
  WARP_SYNC(warp_sort.member_mask);

  // Tests below use sequence to fill the data.
  // Therefore the following value should be greater than any that
  // is present in the input data.
  const DataType oob_default =
    static_cast<std::uint64_t>(ThreadsInBlock * ItemsPerThread + 1);

  if (Stable)
  {
    if (valid_items == ThreadsInBlock * ItemsPerThread)
    {
      warp_sort.StableSort(
        thread_data,
        CustomLess());
    }
    else
    {
      warp_sort.StableSort(
        thread_data,
        CustomLess(),
        valid_items,
        oob_default);
    }
  }
  else
  {
    if (valid_items == ThreadsInBlock * ItemsPerThread)
    {
      warp_sort.Sort(
        thread_data,
        CustomLess());
    }
    else
    {
      warp_sort.Sort(
        thread_data,
        CustomLess(),
        valid_items,
        oob_default);
    }
  }

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    const unsigned int idx = thread_offset + item;

    if (item >= valid_items)
      break;

    data[idx] = thread_data[item];
  }
}

template<
  typename DataType,
  unsigned int ThreadsInBlock,
  unsigned int ThreadsInWarp,
  unsigned int ItemsPerThread,
  bool Stable>
void WarpMergeSortTest(
  unsigned int valid_segments,
  DataType *data,
  const unsigned int *segment_sizes)
{
  WarpMergeSortTestKernel<DataType,
                          ThreadsInBlock,
                          ThreadsInWarp,
                          ItemsPerThread,
                          Stable>
    <<<1, ThreadsInBlock>>>(valid_segments, data, segment_sizes);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
}

template <typename DataType,
          unsigned int ThreadsInWarp,
          unsigned int ItemsPerThread>
bool CheckResult(unsigned int valid_segments,
                 thrust::device_vector<DataType> &d_data,
                 thrust::host_vector<DataType> &h_data,
                 const thrust::host_vector<unsigned int> &segment_sizes)
{
  thrust::copy(d_data.begin(), d_data.end(), h_data.begin());

  constexpr unsigned int max_segment_size = ThreadsInWarp * ItemsPerThread;

  for (unsigned int segment_id = 0; segment_id < valid_segments; segment_id++)
  {
    unsigned int segment_size = segment_sizes[segment_id];

    for (unsigned int i = 0; i < segment_size; i++)
    {
      const auto actual_value = h_data[max_segment_size * segment_id + i];
      const auto expected_value = static_cast<DataType>(i);

      if (actual_value != expected_value)
      {
        return false;
      }
    }
  }

  return true;
}

template <
  typename DataType,
  unsigned int ThreadsInBlock,
  unsigned int ThreadsInWarp,
  unsigned int ItemsPerThread,
  bool Stable>
void Test(unsigned int valid_segments,
          thrust::default_random_engine &rng,
          thrust::device_vector<DataType> &d_data,
          thrust::host_vector<DataType> &h_data,
          const thrust::host_vector<unsigned int> &h_segment_sizes,
          const thrust::device_vector<unsigned int> &d_segment_sizes)
{
  thrust::fill(d_data.begin(), d_data.end(), DataType{});

  constexpr unsigned int max_segment_size = ThreadsInWarp * ItemsPerThread;

  for (unsigned int segment_id = 0; segment_id < valid_segments; segment_id++)
  {
    const unsigned int segment_offset = max_segment_size * segment_id;
    const unsigned int segment_size = h_segment_sizes[segment_id];
    auto segment_begin = d_data.begin() + segment_offset;
    auto segment_end = segment_begin + segment_size;

    thrust::sequence(segment_begin, segment_end);
    thrust::shuffle(segment_begin, segment_end, rng);
  }

  WarpMergeSortTest<DataType,
                    ThreadsInBlock,
                    ThreadsInWarp,
                    ItemsPerThread,
                    Stable>(valid_segments,
                            thrust::raw_pointer_cast(d_data.data()),
                            thrust::raw_pointer_cast(d_segment_sizes.data()));

  const bool check =
    CheckResult<DataType, ThreadsInWarp, ItemsPerThread>(valid_segments,
                                                         d_data,
                                                         h_data,
                                                         h_segment_sizes);

  AssertTrue(check);
}

template <
  typename KeyType,
  typename ValueType,
  unsigned int ThreadsInBlock,
  unsigned int ThreadsInWarp,
  unsigned int ItemsPerThread,
  bool Stable>
void Test(thrust::default_random_engine &rng)
{
  constexpr unsigned int max_segments = ThreadsInBlock / ThreadsInWarp;
  constexpr unsigned int max_segment_size = ThreadsInWarp * ItemsPerThread;

  thrust::device_vector<unsigned int> h_segment_sizes_set(max_segment_size);
  thrust::sequence(h_segment_sizes_set.begin(), h_segment_sizes_set.end());

  thrust::device_vector<unsigned int> h_segment_sizes;
  for (unsigned int segment_id = 0; segment_id < max_segments; segment_id++)
  {
    h_segment_sizes.insert(h_segment_sizes.end(),
                           h_segment_sizes_set.begin(),
                           h_segment_sizes_set.end());
  }

  thrust::device_vector<unsigned int> d_segment_sizes(h_segment_sizes);

  thrust::device_vector<KeyType> d_keys(max_segments * max_segment_size);
  thrust::device_vector<ValueType> d_values(max_segments * max_segment_size);
  thrust::host_vector<KeyType> h_keys(max_segments * max_segment_size);
  thrust::host_vector<ValueType> h_values(max_segments * max_segment_size);


  for (unsigned int valid_segments = 1; valid_segments < max_segments; valid_segments++)
  {
    thrust::shuffle(h_segment_sizes.begin(), h_segment_sizes.end(), rng);
    thrust::copy(h_segment_sizes.begin(), h_segment_sizes.end(), d_segment_sizes.begin());

    Test<KeyType, ThreadsInBlock, ThreadsInWarp, ItemsPerThread, Stable>(
      valid_segments,
      rng,
      d_keys,
      h_keys,
      h_segment_sizes,
      d_segment_sizes);
  }
}

template <unsigned int ThreadsInBlock,
          unsigned int ThreadsInWarp,
          unsigned int ItemsPerThread,
          bool Stable>
void Test(thrust::default_random_engine &rng)
{
  Test<std::int32_t, std::int32_t, ThreadsInBlock, ThreadsInWarp, ItemsPerThread, Stable>(rng);
  Test<std::int64_t, std::int64_t, ThreadsInBlock, ThreadsInWarp, ItemsPerThread, Stable>(rng);

  // Mixed types
  Test<std::int16_t, std::int64_t, ThreadsInBlock, ThreadsInWarp, ItemsPerThread, Stable>(rng);
  Test<std::int32_t, std::int64_t, ThreadsInBlock, ThreadsInWarp, ItemsPerThread, Stable>(rng);
}

template <
  unsigned int ThreadsInWarp,
  unsigned int ItemsPerThread,
  bool Stable>
void Test(thrust::default_random_engine &rng)
{
  Test<32,  ThreadsInWarp, ItemsPerThread, Stable>(rng);
  Test<64,  ThreadsInWarp, ItemsPerThread, Stable>(rng);
  Test<128, ThreadsInWarp, ItemsPerThread, Stable>(rng);
  Test<512, ThreadsInWarp, ItemsPerThread, Stable>(rng);
}

template <unsigned int ItemsPerThread,
          bool Stable>
void Test(thrust::default_random_engine &rng)
{
  Test<1,  ItemsPerThread, Stable>(rng);
  Test<2,  ItemsPerThread, Stable>(rng);
  Test<4,  ItemsPerThread, Stable>(rng);
  Test<8,  ItemsPerThread, Stable>(rng);
  Test<16, ItemsPerThread, Stable>(rng);
  Test<32, ItemsPerThread, Stable>(rng);
}

template <unsigned int ItemsPerThread>
void Test(thrust::default_random_engine &rng)
{
  Test<ItemsPerThread, false>(rng);
  Test<ItemsPerThread, true>(rng);
}

struct CountToType
{
  __device__ __host__ CustomType operator()(std::uint64_t val)
  {
    return { val };
  }
};

struct CountComparator
{
  __device__ __host__ bool operator()(const CustomType &lhs, const CustomType &rhs)
  {
    if (lhs.key == rhs.key)
      return lhs.count < rhs.count;

    return lhs.key < rhs.key;
  }
};

// TODO
/*
void TestStability()
{
  constexpr unsigned int items_per_thread = 10;
  constexpr unsigned int threads_per_block = 128;
  constexpr unsigned int elements = items_per_thread * threads_per_block;
  constexpr bool stable = true;

  thrust::device_vector<CustomType> d_keys(elements);
  thrust::device_vector<std::uint64_t> d_counts(elements);
  thrust::sequence(d_counts.begin(), d_counts.end());
  thrust::transform(d_counts.begin(), d_counts.end(), d_keys.begin(), CountToType{});

  // Sort keys
  BlockMergeSortTest<CustomType, items_per_thread, threads_per_block, stable>(
    thrust::raw_pointer_cast(d_keys.data()),
    elements);

  // Check counts
  AssertTrue(thrust::is_sorted(d_keys.begin(), d_keys.end(), CountComparator{}));
}
 */

int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  thrust::default_random_engine rng;

  Test<2>(rng);
  Test<7>(rng);

  // TestStability();

  return 0;
}
