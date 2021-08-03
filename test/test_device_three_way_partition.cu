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

#include <cub/device/device_partition.cuh>
#include <test_util.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/reduce.h>

using namespace cub;

template <typename T>
struct LessThan
{
  T compare;

  explicit __host__ LessThan(T compare)
      : compare(compare)
  {}

  __device__ bool operator()(const T &a) const
  {
    return a < compare;
  }
};

template <typename T>
struct GreaterOrEqual
{
  T compare;

  explicit __host__ GreaterOrEqual(T compare)
    : compare(compare)
  {}

  __device__ bool operator()(const T &a) const
  {
    return a >= compare;
  }
};

template <typename T>
void TestEmpty()
{
  int num_items = 0;

  T *in {};
  T *d_first_part_out {};
  T *d_second_part_out {};
  T *d_unselected_out {};
  T *d_num_selected_out {};

  LessThan<T> le(T{0});
  GreaterOrEqual<T> ge(T{1});

  std::size_t temp_storage_size {};
  CubDebugExit(cub::DevicePartition::If(nullptr,
                                        temp_storage_size,
                                        in,
                                        d_first_part_out,
                                        d_second_part_out,
                                        d_unselected_out,
                                        d_num_selected_out,
                                        num_items,
                                        le,
                                        ge,
                                        0,
                                        true));

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_size);
  std::uint8_t *d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  CubDebugExit(cub::DevicePartition::If(d_temp_storage,
                                        temp_storage_size,
                                        in,
                                        d_first_part_out,
                                        d_second_part_out,
                                        d_unselected_out,
                                        d_num_selected_out,
                                        num_items,
                                        le,
                                        ge,
                                        0,
                                        true));
}

template <typename T>
class ThreeWayPartitionResult
{
public:
  ThreeWayPartitionResult() = delete;
  ThreeWayPartitionResult(int num_items)
    : first_part(num_items)
    , second_part(num_items)
    , unselected(num_items)
  {}

  thrust::device_vector<T> first_part;
  thrust::device_vector<T> second_part;
  thrust::device_vector<T> unselected;

  int num_items_in_first_part {};
  int num_items_in_second_part {};
  int num_unselected_items {};

  bool operator!=(const ThreeWayPartitionResult<T> &other)
  {
    return std::tie(num_items_in_first_part,
                    num_items_in_second_part,
                    num_unselected_items,
                    first_part,
                    second_part,
                    unselected) != std::tie(other.num_items_in_first_part,
                                            other.num_items_in_second_part,
                                            other.num_unselected_items,
                                            other.first_part,
                                            other.second_part,
                                            other.unselected);
  }
};

template <
  typename FirstPartSelectionOp,
  typename SecondPartSelectionOp,
  typename T>
ThreeWayPartitionResult<T> CUBPartition(
  FirstPartSelectionOp first_selector,
  SecondPartSelectionOp second_selector,
  thrust::device_vector<T> &in)
{
  const int num_items = static_cast<int>(in.size());
  ThreeWayPartitionResult<T> result(num_items);

  T *d_in = thrust::raw_pointer_cast(in.data());
  T *d_first_part_out = thrust::raw_pointer_cast(result.first_part.data());
  T *d_second_part_out = thrust::raw_pointer_cast(result.second_part.data());
  T *d_unselected_out = thrust::raw_pointer_cast(result.unselected.data());

  thrust::device_vector<int> num_selected_out(2);
  int *d_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  std::size_t temp_storage_size {};
  CubDebugExit(cub::DevicePartition::If(nullptr,
                                        temp_storage_size,
                                        d_in,
                                        d_first_part_out,
                                        d_second_part_out,
                                        d_unselected_out,
                                        d_num_selected_out,
                                        num_items,
                                        first_selector,
                                        second_selector,
                                        0,
                                        true));

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_size);
  std::uint8_t *d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  CubDebugExit(cub::DevicePartition::If(d_temp_storage,
                                        temp_storage_size,
                                        d_in,
                                        d_first_part_out,
                                        d_second_part_out,
                                        d_unselected_out,
                                        d_num_selected_out,
                                        num_items,
                                        first_selector,
                                        second_selector,
                                        0,
                                        true));

  thrust::host_vector<int> h_num_selected_out(num_selected_out);

  result.num_items_in_first_part = h_num_selected_out[0];
  result.num_items_in_second_part = h_num_selected_out[1];

  result.num_unselected_items = num_items
                              - h_num_selected_out[0]
                              - h_num_selected_out[1];

  return result;
}

template <
  typename FirstPartSelectionOp,
  typename SecondPartSelectionOp,
  typename T>
ThreeWayPartitionResult<T> ThrustPartition(
  FirstPartSelectionOp first_selector,
  SecondPartSelectionOp second_selector,
  thrust::device_vector<T> &in)
{
  const int num_items = static_cast<int>(in.size());
  ThreeWayPartitionResult<T> result(num_items);

  thrust::device_vector<T> intermediate_result(num_items);

  auto intermediate_iterators =
    thrust::partition_copy(in.begin(),
                           in.end(),
                           result.first_part.begin(),
                           intermediate_result.begin(),
                           first_selector);

  result.num_items_in_first_part =
    thrust::distance(result.first_part.begin(), intermediate_iterators.first);

  auto final_iterators = thrust::partition_copy(
    intermediate_result.begin(),
    intermediate_result.begin() + (num_items - result.num_items_in_first_part),
    result.second_part.begin(),
    result.unselected.begin(),
    second_selector);

  result.num_items_in_second_part = thrust::distance(result.second_part.begin(),
                                                     final_iterators.first);

  result.num_unselected_items = thrust::distance(result.unselected.begin(),
                                                 final_iterators.second);

  return result;
}

template <typename T>
void TestStability(int num_items)
{
  thrust::device_vector<T> in(num_items);
  thrust::sequence(in.begin(), in.end());

  T first_unselected_val = static_cast<T>(num_items / 3);
  T first_val_of_second_part = static_cast<T>(2 * num_items / 3);

  LessThan<T> le(first_unselected_val);
  GreaterOrEqual<T> ge(first_val_of_second_part);

  auto cub_result = CUBPartition(le, ge, in);
  auto thrust_result = ThrustPartition(le, ge, in);

  AssertEquals(cub_result, thrust_result);
}

template <typename T>
void TestDependent(int num_items)
{
  TestStability<T>(num_items);
}

template <typename T>
void TestDependent()
{
  for (int num_items = 1; num_items < 1000000; num_items <<= 2)
  {
    TestDependent<T>(num_items);
    TestDependent<T>(num_items + 31);
  }
}

template <typename T>
void Test()
{
  TestEmpty<T>();
  TestDependent<T>();
}

// TODO Iterators
int main(int argc, char **argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  Test<std::uint8_t>();
  Test<std::uint16_t>();
  Test<std::uint32_t>();

  return 0;
}
