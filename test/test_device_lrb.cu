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

#include <stdio.h>
#include <limits>
#include <typeinfo>
#include <memory>

#include <cub/util_allocator.cuh>
#include <cub/device/device_logarithmic_radix_binning.cuh>

#include <thrust/iterator/reverse_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>

#include "test_util.h"

using namespace cub;

std::vector<int32_t> gen_huge_task_with_stride(int32_t huge_task_stride,
                                               int32_t huge_task_size,
                                               int32_t small_task_size,
                                               int32_t num_fragments)
{
  const int32_t num_tasks = huge_task_stride * num_fragments;
  std::vector<int32_t> tasks(num_tasks, small_task_size);

  for (int32_t i = huge_task_stride - 1; i < num_tasks; i += huge_task_stride)
  {
    tasks[i] = huge_task_size;
  }

  return tasks;
}

thrust::host_vector<int32_t> compute_offsets(const std::vector<int32_t> &tasks)
{
  thrust::host_vector<int32_t> offsets;
  offsets.reserve(tasks.size() + 1);

  int32_t last_offset = 0;
  for (int size : tasks)
  {
    offsets.push_back(last_offset);
    last_offset += size;
  }
  offsets.push_back(last_offset);

  return offsets;
}

struct SegmentToSize
{
  const int32_t *begin_offsets;
  const int32_t *end_offsets;

  explicit SegmentToSize(const int32_t *begin_offsets,
                         const int32_t *end_offsets)
      : begin_offsets(begin_offsets)
      , end_offsets(end_offsets)
  {}

  __device__ int32_t operator()(int32_t segment_id) const
  {
    return end_offsets[segment_id] - begin_offsets[segment_id];
  }
};

struct IsEqual
{
  const int32_t val;

  explicit IsEqual(int32_t val)
    : val(val)
  {}

  __device__ bool operator()(int32_t rhs) const
  {
    return val == rhs;
  }
};

void test(const std::vector<int32_t> &task,
          int32_t huge_segment_size,
          int32_t small_segment_size,
          int32_t num_fragments)
{
  const int num_segments = task.size();
  const thrust::host_vector<int32_t> h_offsets = compute_offsets(task);
  const thrust::device_vector<int32_t> d_offsets = h_offsets;
  thrust::device_vector<int32_t> d_begin_balanced_offsets(num_segments);
  thrust::device_vector<int32_t> d_end_balanced_offsets(num_segments);

  const int32_t *offsets = thrust::raw_pointer_cast(d_offsets.data());
  int32_t *begin_balanced_offsets = thrust::raw_pointer_cast(d_begin_balanced_offsets.data());
  int32_t *end_balanced_offsets = thrust::raw_pointer_cast(d_end_balanced_offsets.data());

  LogarithmicRadixBinningResult result;

  size_t tmp_bytes = 0;
  DeviceLogarithmicRadixBinning::BinOffsets(num_segments,
                                            offsets,
                                            offsets + 1,
                                            begin_balanced_offsets,
                                            end_balanced_offsets,
                                            nullptr,
                                            tmp_bytes,
                                            result);

  thrust::device_vector<uint8_t> tmp_storage(tmp_bytes);

  DeviceLogarithmicRadixBinning::BinOffsets(num_segments,
                                            offsets,
                                            offsets + 1,
                                            begin_balanced_offsets,
                                            end_balanced_offsets,
                                            thrust::raw_pointer_cast(
                                              tmp_storage.data()),
                                            tmp_bytes,
                                            result);

  thrust::device_vector<int32_t> d_balanced_segment_sizes(num_segments);
  thrust::tabulate(d_balanced_segment_sizes.begin(),
                   d_balanced_segment_sizes.end(),
                   SegmentToSize(begin_balanced_offsets, end_balanced_offsets));

  // There are only two classes of segments, so the result should be partitioned
  AssertEquals(1,
               thrust::is_partitioned(d_balanced_segment_sizes.begin(),
                                      d_balanced_segment_sizes.end(),
                                      IsEqual(huge_segment_size)));

  // There is only one huge segment per fragment
  AssertEquals(num_fragments,
               thrust::count(d_balanced_segment_sizes.begin(),
                             d_balanced_segment_sizes.end(),
                             huge_segment_size));

  AssertEquals(num_segments - num_fragments,
               thrust::count(d_balanced_segment_sizes.begin(),
                             d_balanced_segment_sizes.end(),
                             small_segment_size));
}

int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  int32_t huge_segment_size = 1024;
  int32_t small_segment_size = 2;
  int32_t num_fragments = 128;
  test(gen_huge_task_with_stride(42, huge_segment_size, small_segment_size, num_fragments),
       huge_segment_size,
       small_segment_size,
       num_fragments);

  return 0;
}