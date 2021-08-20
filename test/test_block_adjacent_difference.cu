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
 * Test of BlockAdjacentDifference utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <limits>
#include <typeinfo>
#include <memory>

#include <cub/util_allocator.cuh>
#include <cub/block/block_adjacent_difference.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/mismatch.h>
#include <thrust/tabulate.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

#include "test_util.h"

using namespace cub;


/**
 * \brief Generates integer sequence \f$S_n=i(i-1)/2\f$.
 *
 * The adjacent difference of this sequence produce consecutive numbers:
 * \f[
 *   p = \frac{i(i - 1)}{2} \\
 *   n = \frac{(i + 1) i}{2} \\
 *   n - p = i \\
 *   \frac{(i + 1) i}{2} - \frac{i (i - 1)}{2} = i \\
 *   (i + 1) i - i (i - 1) = 2 i \\
 *   (i + 1) - (i - 1) = 2 \\
 *   2 = 2
 * \f]
 */
template <typename DestT>
struct TestSequenceGenerator
{
  template <typename SourceT>
  __device__ __host__ DestT operator()(SourceT index) const
  {
    return static_cast<DestT>(index * (index - 1) / SourceT(2));
  }
};



struct CustomType
{
  unsigned int key;
  unsigned int value;

  __device__ __host__ CustomType()
    : key(0)
    , value(0)
  {}

  __device__ __host__ CustomType(unsigned int key, unsigned int value)
    : key(key)
    , value(value)
  {}
};


__device__ __host__ bool operator==(const CustomType& lhs,
                                    const CustomType& rhs)
{
  return lhs.key == rhs.key && lhs.value == rhs.value;
}

__device__ __host__ bool operator!=(const CustomType& lhs,
                                    const CustomType& rhs)
{
  return !(lhs == rhs);
}

__device__ __host__ CustomType operator-(const CustomType& lhs,
                                         const CustomType& rhs)
{
  return CustomType{lhs.key - rhs.key, lhs.value - rhs.value};
}

struct CustomDifference
{
  template <typename DataType>
  __device__ DataType operator()(DataType &lhs, DataType &rhs)
  {
    return lhs - rhs;
  }
};


template <typename DataType,
          unsigned int ThreadsInBlock,
          unsigned int ItemsPerThread,
          bool ReadLeft = false>
__global__ void
BlockAdjacentDifferenceLastTileTestKernel(const DataType *input,
                                          DataType *output,
                                          unsigned int valid_items)
{
  using BlockAdjacentDifferenceT =
    cub::BlockAdjacentDifference<DataType, ThreadsInBlock>;

  __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;

  DataType thread_data[ItemsPerThread];
  DataType thread_result[ItemsPerThread];

  const unsigned int thread_offset = threadIdx.x * ItemsPerThread;

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    thread_data[item] = input[thread_offset + item];
  }
  __syncthreads();

  if (ReadLeft)
  {
    BlockAdjacentDifferenceT(temp_storage).SubtractLeftPartialTile(
      thread_result,
      thread_data,
      CustomDifference(),
      valid_items);
  }
  else
  {
    BlockAdjacentDifferenceT(temp_storage).SubtractRightPartialTile(
      thread_result,
      thread_data,
      CustomDifference(),
      valid_items);
  }

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    output[thread_offset + item] = thread_result[item];
  }
}


template <typename DataType,
          unsigned int ThreadsInBlock,
          unsigned int ItemsPerThread,
          bool ReadLeft = false>
__global__ void BlockAdjacentDifferenceTestKernel(DataType *data)
{
  using BlockAdjacentDifferenceT =
    cub::BlockAdjacentDifference<DataType, ThreadsInBlock>;

  __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;

  DataType thread_data[ItemsPerThread];
  DataType thread_result[ItemsPerThread];

  const unsigned int thread_offset = threadIdx.x * ItemsPerThread;

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    thread_data[item] = data[thread_offset + item];
  }
  __syncthreads();

  if (ReadLeft)
  {
    BlockAdjacentDifferenceT(temp_storage)
      .SubtractLeft(thread_result,
                    thread_data,
                    CustomDifference());
  }
  else
  {
    BlockAdjacentDifferenceT(temp_storage)
      .SubtractRight(thread_result,
                     thread_data,
                     CustomDifference());
  }

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    data[thread_offset + item] = thread_result[item];
  }
}


template <typename DataType,
          unsigned int ThreadsInBlock,
          unsigned int ItemsPerThread,
          bool ReadLeft = false>
__global__ void
BlockAdjacentDifferenceLastTileTestInplaceKernel(const DataType *input,
                                                 DataType *output,
                                                 unsigned int valid_items)
{
  using BlockAdjacentDifferenceT =
    cub::BlockAdjacentDifference<DataType, ThreadsInBlock>;

  __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;

  DataType thread_data[ItemsPerThread];

  const unsigned int thread_offset = threadIdx.x * ItemsPerThread;

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    thread_data[item] = input[thread_offset + item];
  }
  __syncthreads();

  if (ReadLeft)
  {
    BlockAdjacentDifferenceT(temp_storage)
      .SubtractLeftPartialTile(thread_data,
                               thread_data,
                               CustomDifference(),
                               valid_items);
  }
  else
  {
    BlockAdjacentDifferenceT(temp_storage)
      .SubtractRightPartialTile(thread_data,
                                thread_data,
                                CustomDifference(),
                                valid_items);
  }

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    output[thread_offset + item] = thread_data[item];
  }
}


template <typename DataType,
          unsigned int ThreadsInBlock,
          unsigned int ItemsPerThread,
          bool ReadLeft = false>
__global__ void
BlockAdjacentDifferenceTestInplaceKernel(DataType *data)
{
  using BlockAdjacentDifferenceT =
    cub::BlockAdjacentDifference<DataType, ThreadsInBlock>;

  __shared__ typename BlockAdjacentDifferenceT::TempStorage temp_storage;

  DataType thread_data[ItemsPerThread];

  const unsigned int thread_offset = threadIdx.x * ItemsPerThread;

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    thread_data[item] = data[thread_offset + item];
  }
  __syncthreads();

  if (ReadLeft)
  {
    BlockAdjacentDifferenceT(temp_storage)
      .SubtractLeft(thread_data,
                    thread_data,
                    CustomDifference());
  }
  else
  {
    BlockAdjacentDifferenceT(temp_storage)
      .SubtractRight(thread_data,
                     thread_data,
                     CustomDifference());
  }

  for (unsigned int item = 0; item < ItemsPerThread; item++)
  {
    data[thread_offset + item] = thread_data[item];
  }
}


template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock,
          bool ReadLeft = false>
void BlockAdjacentDifferenceLastTileTest(const DataType *input,
                                         DataType *output,
                                         unsigned int valid_items)
{
  BlockAdjacentDifferenceLastTileTestKernel<DataType,
                                            ThreadsInBlock,
                                            ItemsPerThread,
                                            ReadLeft>
    <<<1, ThreadsInBlock>>>(input, output, valid_items);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
}


template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock,
          bool ReadLeft = false>
void BlockAdjacentDifferenceTest(DataType *data)
{
  BlockAdjacentDifferenceTestKernel<DataType,
                                    ThreadsInBlock,
                                    ItemsPerThread,
                                    ReadLeft><<<1, ThreadsInBlock>>>(data);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
}


template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock,
          bool ReadLeft = false>
void BlockAdjacentDifferenceLastTileInplaceTest(const DataType *input,
                                                DataType *output,
                                                unsigned int valid_items)
{
  BlockAdjacentDifferenceLastTileTestInplaceKernel<DataType,
                                                   ThreadsInBlock,
                                                   ItemsPerThread,
                                                   ReadLeft>
    <<<1, ThreadsInBlock>>>(input, output, valid_items);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
}


template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock,
          bool ReadLeft = false>
void BlockAdjacentDifferenceInplaceTest(DataType *data)
{
  BlockAdjacentDifferenceTestInplaceKernel<DataType,
                                           ThreadsInBlock,
                                           ItemsPerThread,
                                           ReadLeft>
    <<<1, ThreadsInBlock>>>(data);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
}


template <typename FirstIteratorT,
          typename SecondOperatorT>
bool CheckResult(FirstIteratorT first_begin,
                 FirstIteratorT first_end,
                 SecondOperatorT second_begin)
{
  auto err = thrust::mismatch(first_begin, first_end, second_begin);

  if (err.first != first_end)
  {
    return false;
  }

  return true;
}

template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock>
void TestLastTile(bool inplace,
                  unsigned int num_items,
                  thrust::device_vector<DataType> &d_input)
{
  thrust::tabulate(d_input.begin(),
                   d_input.end(),
                   TestSequenceGenerator<DataType>{});
  thrust::device_vector<DataType> d_output(d_input.size());

  constexpr bool read_left = true;
  constexpr bool read_right = false;

  DataType *d_input_ptr = thrust::raw_pointer_cast(d_input.data());
  DataType *d_output_ptr = thrust::raw_pointer_cast(d_output.data());

  if (inplace)
  {
    BlockAdjacentDifferenceLastTileInplaceTest<DataType,
                                               ItemsPerThread,
                                               ThreadsInBlock,
                                               read_left>(d_input_ptr,
                                                          d_output_ptr,
                                                          num_items);
  }
  else
  {
    BlockAdjacentDifferenceLastTileTest<DataType,
                                        ItemsPerThread,
                                        ThreadsInBlock,
                                        read_left>(d_input_ptr,
                                                   d_output_ptr,
                                                   num_items);
  }

  {
    using CountingIteratorT =
      typename thrust::counting_iterator<DataType,
        thrust::use_default,
        std::size_t,
        std::size_t>;

    AssertEquals(d_output.front(), d_input.front());
    AssertTrue(CheckResult(d_output.begin() + 1,
                           d_output.begin() + num_items,
                           CountingIteratorT(DataType{0})));
    AssertTrue(CheckResult(d_output.begin() + num_items,
                           d_output.end(),
                           d_input.begin() + num_items));
  }


  thrust::tabulate(d_input.begin(),
                   d_input.end(),
                   TestSequenceGenerator<DataType>{});

  if (inplace)
  {
    BlockAdjacentDifferenceLastTileInplaceTest<DataType,
                                               ItemsPerThread,
                                               ThreadsInBlock,
                                               read_right>(d_input_ptr,
                                                           d_output_ptr,
                                                           num_items);
  }
  else
  {
    BlockAdjacentDifferenceLastTileTest<DataType,
                                        ItemsPerThread,
                                        ThreadsInBlock,
                                        read_right>(d_input_ptr,
                                                    d_output_ptr,
                                                    num_items);
  }

  {
    thrust::device_vector<DataType> reference(num_items);
    thrust::sequence(reference.begin(),
                     reference.end(),
                     static_cast<DataType>(0),
                     static_cast<DataType>(-1));

    AssertTrue(CheckResult(d_output.begin(),
                           d_output.begin() + num_items - 1,
                           reference.begin()));
    AssertTrue(CheckResult(d_output.begin() + num_items - 1,
                           d_output.end(),
                           d_input.begin() + num_items - 1));
  }
}

struct IntToCustomType
{
  unsigned int offset;

  IntToCustomType()
      : offset(0)
  {}

  explicit IntToCustomType(unsigned int offset)
      : offset(offset)
  {}

  __device__ __host__ CustomType operator()(unsigned int idx) const
  {
    return { idx + offset, idx + offset };
  }
};


template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock>
void TestFullTile(bool inplace,
                  thrust::device_vector<DataType> &d_data)
{
  thrust::tabulate(d_data.begin(),
                   d_data.end(),
                   TestSequenceGenerator<DataType>{});

  constexpr bool read_left  = true;
  constexpr bool read_right = false;

  DataType *d_data_ptr = thrust::raw_pointer_cast(d_data.data());

  if (inplace)
  {
    BlockAdjacentDifferenceInplaceTest<DataType,
                                       ItemsPerThread,
                                       ThreadsInBlock,
                                       read_left>(d_data_ptr);
  }
  else
  {
    BlockAdjacentDifferenceTest<DataType,
                                ItemsPerThread,
                                ThreadsInBlock,
                                read_left>(d_data_ptr);
  }

  {
    using CountingIteratorT =
    typename thrust::counting_iterator<DataType,
      thrust::use_default,
      std::size_t,
      std::size_t>;

    AssertEquals(d_data.front(), TestSequenceGenerator<DataType>{}(0));
    AssertTrue(CheckResult(d_data.begin() + 1,
                           d_data.end(),
                           CountingIteratorT(DataType{0})));
  }

  thrust::tabulate(d_data.begin(),
                   d_data.end(),
                   TestSequenceGenerator<DataType>{});

  if (inplace)
  {
    BlockAdjacentDifferenceInplaceTest<DataType,
                                       ItemsPerThread,
                                       ThreadsInBlock,
                                       read_right>(d_data_ptr);
  }
  else
  {
    BlockAdjacentDifferenceTest<DataType,
                                ItemsPerThread,
                                ThreadsInBlock,
                                read_right>(d_data_ptr);
  }

  {
    thrust::device_vector<DataType> reference(d_data.size());
    thrust::sequence(reference.begin(),
                     reference.end(),
                     static_cast<DataType>(0),
                     static_cast<DataType>(-1));

    AssertTrue(CheckResult(d_data.begin(),
                           d_data.end() - 1,
                           reference.begin()));
    AssertEquals(d_data.back(),
                 TestSequenceGenerator<DataType>{}(d_data.size() - 1));
  }
}


template <unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock>
void TestCustomType(bool inplace,
                    thrust::device_vector<CustomType> &d_data)
{
  thrust::tabulate(d_data.begin(), d_data.end(), IntToCustomType{1});
  CustomType *d_data_ptr = thrust::raw_pointer_cast(d_data.data());

  constexpr bool read_left  = true;
  constexpr bool read_right = false;

  if (inplace)
  {
    BlockAdjacentDifferenceInplaceTest<CustomType,
                                       ItemsPerThread,
                                       ThreadsInBlock,
                                       read_left>(d_data_ptr);
  }
  else
  {
    BlockAdjacentDifferenceTest<CustomType,
                                ItemsPerThread,
                                ThreadsInBlock,
                                read_left>(d_data_ptr);
  }

  {
    const std::size_t expected_count = d_data.size();
    const std::size_t actual_count =
      thrust::count(d_data.begin(), d_data.end(), CustomType{1, 1});

    AssertEquals(expected_count, actual_count);
  }

  thrust::tabulate(d_data.begin(), d_data.end(), IntToCustomType{});

  if (inplace)
  {
    BlockAdjacentDifferenceInplaceTest<CustomType,
                                       ItemsPerThread,
                                       ThreadsInBlock,
                                       read_right>(d_data_ptr);
  }
  else
  {
    BlockAdjacentDifferenceTest<CustomType,
      ItemsPerThread,
      ThreadsInBlock,
      read_right>(d_data_ptr);
  }

  {
    const auto unsigned_minus_one = static_cast<unsigned int>(-1);

    const std::size_t expected_count = d_data.size() - 1;
    const std::size_t actual_count =
      thrust::count(d_data.begin(),
                    d_data.end() - 1,
                    CustomType{unsigned_minus_one, unsigned_minus_one});

    AssertEquals(expected_count, actual_count);
  }
}


template <
  typename ValueType,
  unsigned int ItemsPerThread,
  unsigned int ThreadsInBlock>
void Test(bool inplace)
{
  constexpr int tile_size = ItemsPerThread * ThreadsInBlock;
  thrust::device_vector<ValueType> d_values(tile_size);

  for (unsigned int num_items = tile_size; num_items > 1; num_items /= 2)
  {
    TestLastTile<ValueType, ItemsPerThread, ThreadsInBlock>(inplace,
                                                            num_items,
                                                            d_values);
  }

  TestFullTile<ValueType, ItemsPerThread, ThreadsInBlock>(inplace, d_values);
}


template <unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock>
void TestCustomType(bool inplace)
{
  constexpr int tile_size = ItemsPerThread * ThreadsInBlock;
  thrust::device_vector<CustomType> d_values(tile_size);
  TestCustomType<ItemsPerThread, ThreadsInBlock>(inplace, d_values);
}


template <unsigned int ItemsPerThread, unsigned int ThreadsPerBlock>
void Test(bool inplace)
{
  Test<std::uint8_t,  ItemsPerThread, ThreadsPerBlock>(inplace);
  Test<std::uint16_t, ItemsPerThread, ThreadsPerBlock>(inplace);
  Test<std::uint32_t, ItemsPerThread, ThreadsPerBlock>(inplace);
  Test<std::uint64_t, ItemsPerThread, ThreadsPerBlock>(inplace);
}


template <unsigned int ItemsPerThread>
void Test(bool inplace)
{
  Test<ItemsPerThread, 32>(inplace);
  Test<ItemsPerThread, 256>(inplace);
}


template <unsigned int ItemsPerThread>
void Test()
{
  Test<ItemsPerThread>(false);
  Test<ItemsPerThread>(true);
}


int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  Test<1>();
  Test<2>();
  Test<10>();
  Test<15>();

  // More of a compilation check
  TestCustomType<5, 256>(true);

  return 0;
}
