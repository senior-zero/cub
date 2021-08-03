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
struct GreaterThan
{
  T compare;

  explicit __host__ GreaterThan(T compare)
    : compare(compare)
  {}

  __device__ bool operator()(const T &a) const
  {
    return a > compare;
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
  GreaterThan<T> ge(T{1});

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
void Test()
{
  TestEmpty<T>();
}

int main(int argc, char **argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  Test<std::uint32_t>();

  return 0;
}
