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

#include <cub/util_allocator.cuh>
#include <cub/device/device_partition.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_logarithmic_radix_binning.cuh>

#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

using namespace cub;

struct SegmentHandler
{
  using offset_size_type = std::size_t;

  std::size_t *work_count;
  int *balanced_segment_ids;

  SegmentHandler(std::size_t *work_count, int *balanced_segment_ids)
      : work_count(work_count)
      , balanced_segment_ids(balanced_segment_ids)
  {}

  __device__ offset_size_type get_segment_size(int segment_id) const
  {
    return work_count[segment_id];
  }

  __device__ void set_balanced_position(int segment_id, int balanced_pos) const
  {
    balanced_segment_ids[balanced_pos] = segment_id;
  }
};

float MeasureLRB(thrust::device_vector<std::size_t> &segment_work_count,
                 thrust::device_vector<int> &result)
{
  const int num_segments = static_cast<int>(segment_work_count.size());
  SegmentHandler segment_handler(
    thrust::raw_pointer_cast(segment_work_count.data()),
    thrust::raw_pointer_cast(result.data()));

  std::size_t temp_storage_size = 0;

  cub::LogarithmicRadixBinningResult bin_result;
  cub::DeviceLogarithmicRadixBinning::Bin(num_segments,
                                          segment_handler,
                                          nullptr,
                                          temp_storage_size,
                                          bin_result);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_size);

  cudaEvent_t begin, end;

  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin);
  cub::DeviceLogarithmicRadixBinning::Bin(num_segments,
                                          segment_handler,
                                          thrust::raw_pointer_cast(temp_storage.data()),
                                          temp_storage_size,
                                          bin_result);
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, begin, end);

  cudaEventDestroy(end);
  cudaEventDestroy(begin);

  return ms;
}

float MeasureRadixSort(thrust::device_vector<std::size_t> &segment_work_count,
                       thrust::device_vector<int> &result)
{
  const int num_segments = static_cast<int>(segment_work_count.size());
  thrust::device_vector<int> segment_ids(num_segments);

  thrust::device_vector<std::size_t> balanced_segment_work_count(segment_work_count);

  std::size_t temp_storage_size = 0;

  cub::DeviceRadixSort::SortPairs(nullptr,
                                  temp_storage_size,
                                  thrust::raw_pointer_cast(segment_work_count.data()),
                                  thrust::raw_pointer_cast(balanced_segment_work_count.data()),
                                  thrust::raw_pointer_cast(segment_ids.data()),
                                  thrust::raw_pointer_cast(result.data()),
                                  num_segments);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_size);

  cudaEvent_t begin, end;

  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin);

  thrust::sequence(segment_ids.begin(), segment_ids.end());
  cub::DeviceRadixSort::SortPairs(thrust::raw_pointer_cast(temp_storage.data()),
                                  temp_storage_size,
                                  thrust::raw_pointer_cast(segment_work_count.data()),
                                  thrust::raw_pointer_cast(balanced_segment_work_count.data()),
                                  thrust::raw_pointer_cast(segment_ids.data()),
                                  thrust::raw_pointer_cast(result.data()),
                                  num_segments);

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, begin, end);

  cudaEventDestroy(end);
  cudaEventDestroy(begin);

  return ms;
}

struct WorkCountToBinId
{
  __device__ std::uint8_t operator()(std::size_t work_count)
  {
    return __clzll(static_cast<long long int>(work_count));
  }
};

float MeasureRadixSort8B(thrust::device_vector<std::size_t> &segment_work_count,
                         thrust::device_vector<int> &result)
{
  const int num_segments = static_cast<int>(segment_work_count.size());
  thrust::device_vector<int> segment_ids(num_segments);

  thrust::device_vector<std::uint8_t> bin_ids(segment_work_count);
  thrust::device_vector<std::uint8_t> balanced_bin_ids(segment_work_count);

  std::size_t temp_storage_size = 0;

  cub::DeviceRadixSort::SortPairs(nullptr,
                                  temp_storage_size,
                                  thrust::raw_pointer_cast(bin_ids.data()),
                                  thrust::raw_pointer_cast(balanced_bin_ids.data()),
                                  thrust::raw_pointer_cast(segment_ids.data()),
                                  thrust::raw_pointer_cast(result.data()),
                                  num_segments);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_size);

  cudaEvent_t begin, end;

  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin);

  thrust::sequence(segment_ids.begin(), segment_ids.end());
  thrust::transform(segment_work_count.begin(), segment_work_count.end(), bin_ids.begin(), WorkCountToBinId());
  cub::DeviceRadixSort::SortPairs(thrust::raw_pointer_cast(temp_storage.data()),
                                  temp_storage_size,
                                  thrust::raw_pointer_cast(bin_ids.data()),
                                  thrust::raw_pointer_cast(balanced_bin_ids.data()),
                                  thrust::raw_pointer_cast(segment_ids.data()),
                                  thrust::raw_pointer_cast(result.data()),
                                  num_segments);

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, begin, end);

  cudaEventDestroy(end);
  cudaEventDestroy(begin);

  return ms;
}

struct SegmentDescription
{
  int segment_id;
  std::size_t segment_size;

  __host__ __device__ SegmentDescription() {}
  __host__ __device__ SegmentDescription(int segment_id, std::size_t segment_size)
      : segment_id(segment_id)
      , segment_size(segment_size)
  {}
};

struct SegmentComparator
{
  __host__ __device__ bool operator()(const SegmentDescription &lhs,
                                      const SegmentDescription &rhs) const
  {
    return lhs.segment_size < rhs.segment_size;
  }
};

struct WorkCountToSegmentDescription
{
  const std::size_t * segment_work_count;

           __device__ __host__ WorkCountToSegmentDescription() {}
  explicit __device__ __host__ WorkCountToSegmentDescription(const std::size_t *segment_work_count)
      : segment_work_count(segment_work_count)
  {}

  __device__ __host__ SegmentDescription operator()(int segment_id) const
  {
    return {segment_id, segment_work_count[segment_id]};
  }
};

float MeasureMergeSort(thrust::device_vector<std::size_t> &segment_work_count,
                       thrust::device_vector<int> &/* result */)
{
  const int num_segments = static_cast<int>(segment_work_count.size());
  thrust::device_vector<SegmentDescription> keys(num_segments);
  thrust::tabulate(keys.begin(),
                   keys.end(),
                   WorkCountToSegmentDescription(
                     thrust::raw_pointer_cast(segment_work_count.data())));

  std::size_t temp_storage_size = 0;

  cub::DeviceMergeSort::SortKeys(nullptr,
                                 temp_storage_size,
                                 thrust::raw_pointer_cast(keys.data()),
                                 num_segments,
                                 SegmentComparator());

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_size);

  cudaEvent_t begin, end;

  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin);

  cub::DeviceMergeSort::SortKeys(thrust::raw_pointer_cast(temp_storage.data()),
                                 temp_storage_size,
                                 thrust::raw_pointer_cast(keys.data()),
                                 num_segments,
                                 SegmentComparator());

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, begin, end);

  cudaEventDestroy(end);
  cudaEventDestroy(begin);

  return ms;
}

struct Partition
{
  std::size_t segment_size;
  const std::size_t *segment_sizes;

  __device__ __host__ Partition(std::size_t segment_size,
                                const std::size_t *segment_sizes)
      : segment_size(segment_size)
      , segment_sizes(segment_sizes)
  {}

  __device__ __host__ bool operator()(int segment_id) const
  {
    return segment_sizes[segment_id] > segment_size;
  }
};

float MeasurePartition2(thrust::device_vector<std::size_t> &segment_work_count,
                       thrust::device_vector<int> result)
{
  const int num_segments = static_cast<int>(segment_work_count.size());

  std::size_t temp_storage_size = 0;

  Partition partition(num_segments / 2, // random value for now
                      thrust::raw_pointer_cast(segment_work_count.data()));

  thrust::device_vector<std::size_t> huge_segments_count(1);

  cub::DevicePartition::If(nullptr,
                           temp_storage_size,
                           thrust::counting_iterator<int>(0),
                           thrust::raw_pointer_cast(result.data()),
                           thrust::raw_pointer_cast(huge_segments_count.data()),
                           num_segments,
                           partition);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_size);

  cudaEvent_t begin, end;

  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin);

  cub::DevicePartition::If(thrust::raw_pointer_cast(temp_storage.data()),
                           temp_storage_size,
                           thrust::counting_iterator<int>(0),
                           thrust::raw_pointer_cast(result.data()),
                           thrust::raw_pointer_cast(huge_segments_count.data()),
                           num_segments,
                           partition);

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, begin, end);

  cudaEventDestroy(end);
  cudaEventDestroy(begin);

  return ms;
}

float MeasurePartition3(thrust::device_vector<std::size_t> &segment_work_count,
                        thrust::device_vector<int> result)
{
  const int num_segments = static_cast<int>(segment_work_count.size());

  std::size_t temp_storage_size = 0;

  Partition partition_1(num_segments / 2 /*2 << 24*/, // huge segments
                        thrust::raw_pointer_cast(segment_work_count.data()));
  Partition partition_2(2 << 5, // small segments
                        thrust::raw_pointer_cast(segment_work_count.data()));

  thrust::device_vector<std::size_t> huge_segments_count(1);

  cub::DevicePartition::If(nullptr,
                           temp_storage_size,
                           thrust::counting_iterator<int>(0),
                           thrust::raw_pointer_cast(result.data()),
                           thrust::raw_pointer_cast(huge_segments_count.data()),
                           num_segments,
                           partition_1);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_size);

  cudaEvent_t begin, end;

  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin);

  cub::DevicePartition::If(thrust::raw_pointer_cast(temp_storage.data()),
                           temp_storage_size,
                           thrust::counting_iterator<int>(0),
                           thrust::raw_pointer_cast(result.data()),
                           thrust::raw_pointer_cast(huge_segments_count.data()),
                           num_segments,
                           partition_1);

  std::size_t h_huge_segments_count;
  cudaMemcpy(&h_huge_segments_count, thrust::raw_pointer_cast(huge_segments_count.data()), sizeof(std::size_t), cudaMemcpyDeviceToHost);

  cub::DevicePartition::If(thrust::raw_pointer_cast(temp_storage.data()),
                           temp_storage_size,
                           thrust::counting_iterator<int>(static_cast<int>(h_huge_segments_count)),
                           thrust::raw_pointer_cast(result.data() + h_huge_segments_count),
                           thrust::raw_pointer_cast(huge_segments_count.data()),
                           num_segments - h_huge_segments_count,
                           partition_2);

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, begin, end);

  cudaEventDestroy(end);
  cudaEventDestroy(begin);

  return ms;
}

void CompareLRB(unsigned int groups)
{
  const std::size_t max_power_of_two = 29;
  const std::size_t max_elements = 1 << max_power_of_two;

  thrust::device_vector<std::size_t> segment_work_count(max_elements);
  thrust::device_vector<int> result(max_elements);

  thrust::default_random_engine rng;

  std::cout << "Elements, LRB, RadixSort, RadixSort8B, MergeSort, Partition(2), Partition(3)" << std::endl;

  for (std::size_t power_of_two = 8;
       power_of_two <= max_power_of_two;
       power_of_two++)
  {
    const std::size_t elements = 1 << power_of_two;
    const std::size_t group_size = elements / groups;

    segment_work_count.resize(elements);
    result.resize(elements);

    for (unsigned int group_id = 0; group_id < groups; group_id++)
    {
      const std::size_t group_begin = group_size * group_id;
      const std::size_t group_end   = std::min(group_begin + group_size,
                                             elements);
      thrust::fill(segment_work_count.begin() + group_begin,
                   segment_work_count.begin() + group_end,
                   group_size * group_id);
    }

    thrust::shuffle(segment_work_count.begin(), segment_work_count.end(), rng);

    std::cout << elements << ", "
              << MeasureLRB(segment_work_count, result) << ", "
              << MeasureRadixSort(segment_work_count, result) << ", "
              << MeasureRadixSort8B(segment_work_count, result) << ", "
              << MeasureMergeSort(segment_work_count, result) << ", "
              << MeasurePartition2(segment_work_count, result) << ", "
              << MeasurePartition3(segment_work_count, result) << ", "
              << std::endl;
  }
}


int main()
{
  CompareLRB(128);

  return 0;
}
