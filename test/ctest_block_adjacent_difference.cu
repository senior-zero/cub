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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
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

#include <tuple>
#include <type_traits>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <limits>
#include <memory>
#include <typeinfo>

#include <cub/block/block_adjacent_difference.cuh>
#include <cub/util_allocator.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/mismatch.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>

using namespace cub;

template <unsigned int ItemsPerThreadArg>
struct ThreadConfiguration
{
  constexpr static unsigned int ItemsPerThread = ItemsPerThreadArg;
};

template <unsigned int ThreadsInBlockArg>
struct ThreadBlockConfiguration
{
  constexpr static unsigned int ThreadsInBlock = ThreadsInBlockArg;
};

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
  std::size_t offset;

  TestSequenceGenerator(std::size_t offset = 0)
      : offset(offset)
  {}

  template <typename SourceT>
  __device__ __host__ DestT operator()(SourceT index) const
  {
    index += static_cast<SourceT>(offset);
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

__device__ __host__ bool operator==(const CustomType &lhs,
                                    const CustomType &rhs)
{
  return lhs.key == rhs.key && lhs.value == rhs.value;
}

__device__ __host__ bool operator!=(const CustomType &lhs,
                                    const CustomType &rhs)
{
  return !(lhs == rhs);
}

__device__ __host__ CustomType operator-(const CustomType &lhs,
                                         const CustomType &rhs)
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
__global__ void LastTileTestKernel(const DataType *input,
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
    BlockAdjacentDifferenceT(temp_storage)
      .SubtractLeftPartialTile(thread_result,
                               thread_data,
                               CustomDifference(),
                               valid_items);
  }
  else
  {
    BlockAdjacentDifferenceT(temp_storage)
      .SubtractRightPartialTile(thread_result,
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
__global__ void MiddleTileTestKernel(const DataType *input,
                                     DataType *output,
                                     DataType neighbour_tile_value)
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
    BlockAdjacentDifferenceT(temp_storage)
      .SubtractLeft(thread_result,
                    thread_data,
                    CustomDifference(),
                    neighbour_tile_value);
  }
  else
  {
    BlockAdjacentDifferenceT(temp_storage)
      .SubtractRight(thread_result,
                     thread_data,
                     CustomDifference(),
                     neighbour_tile_value);
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
__global__ void MiddleTileInplaceTestKernel(const DataType *input,
                                            DataType *output,
                                            DataType neighbour_tile_value)
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
      .SubtractLeft(thread_data,
                    thread_data,
                    CustomDifference(),
                    neighbour_tile_value);
  }
  else
  {
    BlockAdjacentDifferenceT(temp_storage)
      .SubtractRight(thread_data,
                     thread_data,
                     CustomDifference(),
                     neighbour_tile_value);
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
__global__ void TestKernel(DataType *data)
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
      .SubtractLeft(thread_result, thread_data, CustomDifference());
  }
  else
  {
    BlockAdjacentDifferenceT(temp_storage)
      .SubtractRight(thread_result, thread_data, CustomDifference());
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
__global__ void LastTileTestInplaceKernel(const DataType *input,
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
__global__ void TestInplaceKernel(DataType *data)
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
      .SubtractLeft(thread_data, thread_data, CustomDifference());
  }
  else
  {
    BlockAdjacentDifferenceT(temp_storage)
      .SubtractRight(thread_data, thread_data, CustomDifference());
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
void LastTileTest(const DataType *input,
                  DataType *output,
                  unsigned int valid_items)
{
  LastTileTestKernel<DataType, ThreadsInBlock, ItemsPerThread, ReadLeft>
    <<<1, ThreadsInBlock>>>(input, output, valid_items);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
}

template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock,
          bool ReadLeft = false>
void Test(DataType *data)
{
  TestKernel<DataType, ThreadsInBlock, ItemsPerThread, ReadLeft>
    <<<1, ThreadsInBlock>>>(data);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
}

template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock,
          bool ReadLeft = false>
void MiddleTileTest(const DataType *input,
                    DataType *output,
                    DataType neighbour_tile_value)
{
  MiddleTileTestKernel<DataType, ThreadsInBlock, ItemsPerThread, ReadLeft>
    <<<1, ThreadsInBlock>>>(input, output, neighbour_tile_value);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
}

template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock,
          bool ReadLeft = false>
void LastTileInplaceTest(const DataType *input,
                         DataType *output,
                         unsigned int valid_items)
{
  LastTileTestInplaceKernel<DataType, ThreadsInBlock, ItemsPerThread, ReadLeft>
    <<<1, ThreadsInBlock>>>(input, output, valid_items);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
}

template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock,
          bool ReadLeft = false>
void InplaceTest(DataType *data)
{
  TestInplaceKernel<DataType, ThreadsInBlock, ItemsPerThread, ReadLeft>
    <<<1, ThreadsInBlock>>>(data);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
}

template <typename DataType,
          unsigned int ItemsPerThread,
          unsigned int ThreadsInBlock,
          bool ReadLeft = false>
void MiddleTileInplaceTest(const DataType *input,
                           DataType *output,
                           DataType neighbour_tile_value)
{
  MiddleTileInplaceTestKernel<DataType, ThreadsInBlock, ItemsPerThread, ReadLeft>
    <<<1, ThreadsInBlock>>>(input, output, neighbour_tile_value);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
}

template <typename FirstIteratorT, typename SecondOperatorT>
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

template <typename T>
struct Configuration256TB1IPT
{
  using Type = T;
  constexpr static unsigned int ThreadsInBlock = 256;
  constexpr static unsigned int ItemsPerThread = 1;
};

template <typename T>
struct Configuration256TB2IPT
{
  using Type = T;
  constexpr static unsigned int ThreadsInBlock = 256;
  constexpr static unsigned int ItemsPerThread = 2;
};

template <typename T>
struct Configuration256TB4IPT
{
  using Type = T;
  constexpr static unsigned int ThreadsInBlock = 256;
  constexpr static unsigned int ItemsPerThread = 4;
};

template <typename T>
struct Configuration128TB1IPT
{
  using Type = T;
  constexpr static unsigned int ThreadsInBlock = 128;
  constexpr static unsigned int ItemsPerThread = 1;
};

template <typename T>
struct Configuration128TB2IPT
{
  using Type = T;
  constexpr static unsigned int ThreadsInBlock = 128;
  constexpr static unsigned int ItemsPerThread = 2;
};

template <typename T>
struct Configuration128TB4IPT
{
  using Type = T;
  constexpr static unsigned int ThreadsInBlock = 128;
  constexpr static unsigned int ItemsPerThread = 4;
};


TEMPLATE_PRODUCT_TEST_CASE("BlockAdjacentDifference in last tile",
                           "[left][right]",
                           (Configuration128TB1IPT,
                            Configuration128TB2IPT,
                            Configuration128TB4IPT,
                            Configuration256TB1IPT,
                            Configuration256TB2IPT,
                            Configuration256TB4IPT),
                           (std::uint16_t, std::uint32_t, std::uint64_t))
{
  using DataType = typename TestType::Type;
  constexpr unsigned int ItemsPerThread = TestType::ItemsPerThread; // GENERATE doesn't return compile-time result
  constexpr unsigned int ThreadsInBlock = TestType::ThreadsInBlock;

  constexpr unsigned int tile_size = ItemsPerThread * ThreadsInBlock;
  thrust::device_vector<DataType> d_input(tile_size);

  for (bool inplace : {false, true})
  {
    for (unsigned int num_items = tile_size; num_items > 1; num_items /= 2)
    {
      thrust::tabulate(d_input.begin(),
                       d_input.end(),
                       TestSequenceGenerator<DataType>{});
      thrust::device_vector<DataType> d_output(d_input.size());

      constexpr bool read_left  = true;
      constexpr bool read_right = false;

      DataType *d_input_ptr  = thrust::raw_pointer_cast(d_input.data());
      DataType *d_output_ptr = thrust::raw_pointer_cast(d_output.data());

      SECTION( "calculating left adjacent difference" )
      {
        if (inplace)
        {
          LastTileInplaceTest<DataType,
                              ItemsPerThread,
                              ThreadsInBlock,
                              read_left>(d_input_ptr, d_output_ptr, num_items);
        }
        else
        {
          LastTileTest<DataType, ItemsPerThread, ThreadsInBlock, read_left>(
            d_input_ptr,
            d_output_ptr,
            num_items);
        }

        {
          using CountingIteratorT =
          typename thrust::counting_iterator<DataType,
            thrust::use_default,
            std::size_t,
            std::size_t>;

          REQUIRE( d_output.front() == d_input.front() );
          REQUIRE( CheckResult(d_output.begin() + 1,
                               d_output.begin() + num_items,
                               CountingIteratorT(DataType{0})));
          REQUIRE( CheckResult(d_output.begin() + num_items,
                               d_output.end(),
                               d_input.begin() + num_items));
        }
      }

      thrust::tabulate(d_input.begin(),
                       d_input.end(),
                       TestSequenceGenerator<DataType>{});

      SECTION( "calculating right adjacent difference" )
      {
        if (inplace)
        {
          LastTileInplaceTest<DataType,
                              ItemsPerThread,
                              ThreadsInBlock,
                              read_right>(d_input_ptr, d_output_ptr, num_items);
        }
        else
        {
          LastTileTest<DataType, ItemsPerThread, ThreadsInBlock, read_right>(
            d_input_ptr,
            d_output_ptr,
            num_items);
        }

        {
          thrust::device_vector<DataType> reference(num_items);
          thrust::sequence(reference.begin(),
                           reference.end(),
                           static_cast<DataType>(0),
                           static_cast<DataType>(-1));

          REQUIRE(CheckResult(d_output.begin(),
                              d_output.begin() + num_items - 1,
                              reference.begin()));
          REQUIRE(CheckResult(d_output.begin() + num_items - 1,
                              d_output.end(),
                              d_input.begin() + num_items - 1));
        }
      }
    }
  }
}

int main(int argc, char *argv[])
{
  // global setup...

  int result = Catch::Session().run(argc, argv);

  // global clean-up...

  return result;
}