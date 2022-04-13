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

  const int n = 1024 * 1024;
  thrust::device_vector<int> marks(n);
  int *d_marks = thrust::raw_pointer_cast(marks.data());
  Marker op{d_marks};

  auto striped_tuning = cub::TuneForEach<cub::ForEachAlgorithm::BLOCK_STRIPED>(
    cub::ForEachConfigurationSpace{}.Add<1024, 4>()
                                    .Add<256, 4>());

  auto vectorized_tuning = cub::TuneForEach<cub::ForEachAlgorithm::VECTORIZED>(
    cub::ForEachConfigurationSpace{}.Add<1024, 4>()
                                    .Add<256, 4>());

  cub::DeviceFor::Bulk(n, op, {}, true, striped_tuning);
  cub::DeviceFor::ForEachN(d_marks, n, op, {}, true, vectorized_tuning);

  AssertEquals(n, thrust::count(marks.begin(), marks.end(), 1));
}
