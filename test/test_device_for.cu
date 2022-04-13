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

#include <cub/device/device_for.cuh>

#include <thrust/device_vector.h>
#include <thrust/count.h>

#include "test_util.h"


struct Marker
{
  int *d_marks{};

  __device__ void operator()(int i) const
  {
    d_marks[i] = 1;
  }
};

struct ConstRemover
{
  __device__ void operator()(const int &i) const
  {
    const_cast<int&>(i) = 1;
  }
};

struct Counter
{
  int *d_count{};

  __device__ void operator()(int i) const
  {
    if (i == 42)
    {
      atomicAdd(d_count, 1);
    }
  }
};

struct Checker
{
  int *d_marks{};

  __device__ void operator()(int val) const
  {
    if (val == 0)
    {
      printf("Wrong result!\n");
    }
  }
};


int main(int argc, char** argv)
{
  // Initialize command line
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  // Test
  {
    const int n = 32 * 1024 * 1024;

    thrust::device_vector<int> marks(n);
    int *d_marks = thrust::raw_pointer_cast(marks.data());
    Marker op{d_marks};

    auto tuning = cub::TuneForEach<cub::ForEachAlgorithm::BLOCK_STRIPED>(
      cub::ForEachConfigurationSpace{}.Add<1024, 4>()
        .Add<256, 4>());

    cub::DeviceFor::Bulk(n, op, {}, true, tuning);
    AssertEquals(n, thrust::count(marks.begin(), marks.end(), 1));
  }

  // Test
  {
    const int n = 32 * 1024 * 1024;

    thrust::device_vector<int> marks(n);

    cub::DeviceFor::ForEach(marks.begin(), marks.end(), ConstRemover{});
    AssertEquals(n, thrust::count(marks.begin(), marks.end(), 1));
  }

  // Test
  {
    const int n = 32 * 1024 * 1024;

    thrust::device_vector<int> marks(n);
    thrust::device_vector<int> counter(n);

    Counter op{thrust::raw_pointer_cast(counter.data())};

    cub::DeviceFor::ForEach(marks.begin(), marks.end(), op);
  }

  // Bench
  {
    constexpr int max_iterations = 1;
    float striped_ms = 0;
    float vectorized_ms = 0;

    cudaEvent_t begin, end;

    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    for (int n = 1024; n < 512 * 1024 * 1024; n *= 16)
    {
      thrust::device_vector<int> in_1(n, n);
      thrust::device_vector<int> in_2(n, n);

      thrust::device_vector<int> counter_1(n);
      thrust::device_vector<int> counter_2(n);

      int *d_in_1 = thrust::raw_pointer_cast(in_1.data());
      int *d_in_2 = thrust::raw_pointer_cast(in_2.data());

      int *d_counter_1 = thrust::raw_pointer_cast(counter_1.data());
      int *d_counter_2 = thrust::raw_pointer_cast(counter_2.data());

      Counter op_1{d_counter_1};
      Counter op_2{d_counter_2};

      auto striped_tuning = cub::TuneForEach<cub::ForEachAlgorithm::BLOCK_STRIPED>(
        cub::ForEachConfigurationSpace{}.Add<256, 8>());

      auto vectorized_tuning = cub::TuneForEach<cub::ForEachAlgorithm::VECTORIZED>(
        cub::ForEachConfigurationSpace{}.Add<256, 2>());

      cudaEventRecord(begin);
      for (int iteration = 0; iteration < max_iterations; iteration++)
      {
        cub::DeviceFor::ForEachN(d_in_1, n, op_1, {}, {}, striped_tuning);
      }
      cudaEventRecord(end);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&striped_ms, begin, end);

      cudaEventRecord(begin);
      for (int iteration = 0; iteration < max_iterations; iteration++)
      {
        cub::DeviceFor::ForEachN(d_in_2, n, op_2, {}, {}, vectorized_tuning);
      }
      cudaEventRecord(end);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&vectorized_ms, begin, end);

      vectorized_ms /= max_iterations;
      striped_ms /= max_iterations;

      std::cout << n << ", " <<  striped_ms << ", " << vectorized_ms << std::endl;
    }

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
  }
}
