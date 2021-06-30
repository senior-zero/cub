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

#include "../../test/test_util.h"

#include <cub/util_allocator.cuh>
#include <cub/device/device_partition.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_logarithmic_radix_binning.cuh>

#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <curand.h>

#include <fstream>

void CURandCall(curandStatus_t status)
{
  if (status != CURAND_STATUS_SUCCESS)
  {
    throw std::runtime_error("CURand error!");
  }
}

using namespace cub;

struct SegmentHandler
{
  using offset_size_type = std::uint32_t;

  const std::uint32_t *work_count;
  int *balanced_segment_ids;

  SegmentHandler(const std::uint32_t *work_count, int *balanced_segment_ids)
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

float MeasureLRB(const thrust::device_vector<std::uint32_t> &segment_work_count,
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

  // Bins are inverted, so it's 32 - 6
  const int first_small_bin = 26;
  int large_segments_count = 0;

  cudaMemcpy(&large_segments_count,
             bin_result.d_bins_prefix + first_small_bin,
             sizeof(int),
             cudaMemcpyDeviceToHost);

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  // std::cout << num_segments - large_segments_count << " small segments" << std::endl;

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, begin, end);

  cudaEventDestroy(end);
  cudaEventDestroy(begin);

  return ms;
}

struct WorkCountToBinId
{
  __device__ std::uint8_t operator()(std::uint32_t work_count)
  {
    return __clz(static_cast<int>(work_count));
  }
};

float MeasureRadixSort8B(const thrust::device_vector<std::uint32_t> &segment_work_count,
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

struct Partition
{
  std::uint32_t segment_size;
  const std::uint32_t *segment_sizes;

  __device__ __host__ Partition(std::uint32_t segment_size,
                                const std::uint32_t *segment_sizes)
      : segment_size(segment_size)
      , segment_sizes(segment_sizes)
  {}

  __device__ __host__ bool operator()(int segment_id) const
  {
    return segment_sizes[segment_id] > segment_size;
  }
};

float MeasurePartition(
  const thrust::device_vector<std::uint32_t> &segment_work_count,
  thrust::device_vector<int> result)
{
  const int num_segments = static_cast<int>(segment_work_count.size());

  std::size_t temp_storage_size = 0;

  Partition partition(32, // random value for now
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

struct ReversedIota
{
  const std::uint32_t *segment_offsets;
  std::uint32_t *data;

  ReversedIota(const std::uint32_t *segment_offsets,
               std::uint32_t *data)
      : segment_offsets(segment_offsets)
      , data(data)
  {}

  __device__ void operator()(std::uint32_t segment_id) const
  {
    const std::uint32_t segment_begin = segment_offsets[segment_id];
    const std::uint32_t segment_end = segment_offsets[segment_id + 1];
    const std::uint32_t segment_size = segment_end - segment_begin;

    std::size_t count = 0;
    for (std::uint32_t i = segment_begin; i < segment_end; i++)
    {
      data[i] = segment_size - 1 - count++;
    }
  }
};

thrust::device_vector<std::uint32_t>
GenKeys(const thrust::device_vector<std::uint32_t> &offsets)
{
  std::size_t total_items = offsets.back();
  thrust::device_vector<std::uint32_t> keys(total_items);

  thrust::for_each(thrust::counting_iterator<std::uint32_t>(0),
                   thrust::counting_iterator<std::uint32_t>(offsets.size() - 1),
                   ReversedIota(thrust::raw_pointer_cast(offsets.data()),
                                thrust::raw_pointer_cast(keys.data())));

  return keys;
}

thrust::device_vector<std::uint32_t>
sizes_to_offsets(const thrust::device_vector<std::uint32_t> &segment_sizes)
{
  thrust::device_vector<std::uint32_t> offsets(segment_sizes.size() + 1, 0u);
  thrust::copy(segment_sizes.begin(), segment_sizes.end(), offsets.begin());
  thrust::exclusive_scan(offsets.begin(),
                         offsets.end(),
                         offsets.begin());
  return offsets;
}

template <typename T>
struct SegmentChecker
{
  const std::uint32_t *segment_offsets;
  const T *sorted_keys;

  explicit SegmentChecker(const std::uint32_t *segment_offsets,
                          const T *sorted_keys)
      : segment_offsets(segment_offsets)
      , sorted_keys(sorted_keys)
  {}

  __device__ bool operator()(std::uint32_t segment_id)
  {
    const std::uint32_t segment_begin = segment_offsets[segment_id];
    const std::uint32_t segment_end = segment_offsets[segment_id + 1];

    std::uint32_t counter = 0;
    for (std::uint32_t i = segment_begin; i < segment_end; i++)
    {
      if (sorted_keys[i] != counter++)
      {
        return false;
      }
    }

    return true;
  }
};

template <typename T>
bool CheckResult(const thrust::device_vector<uint32_t> &offsets,
                  const thrust::device_vector<T> &sorted_keys)
{
  const unsigned int total_segments = offsets.size() - 1;
  thrust::device_vector<bool> is_segment_sorted(total_segments, true);

  thrust::transform(
    thrust::counting_iterator<std::uint32_t>(0),
    thrust::counting_iterator<std::uint32_t>(total_segments),
    is_segment_sorted.begin(),
    SegmentChecker<T>(thrust::raw_pointer_cast(offsets.data()),
                      thrust::raw_pointer_cast(sorted_keys.data())));

  return thrust::reduce(is_segment_sorted.begin(),
                        is_segment_sorted.end(),
                        true,
                        thrust::logical_and<bool>());
}

template <typename T>
float RunSegmentedSort(
  const thrust::device_vector<std::uint32_t> &offsets,
  const thrust::device_vector<T> &input_keys,
  const thrust::device_vector<T> &input_values,
  thrust::device_vector<T> &sorted_keys,
  thrust::device_vector<T> &sorted_values)
{
  const unsigned int total_items = input_keys.size();
  const unsigned int total_segments = offsets.size() - 1;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairs(
    nullptr,
    temp_storage_bytes,
    thrust::raw_pointer_cast(input_keys.data()),
    thrust::raw_pointer_cast(sorted_keys.data()),
    thrust::raw_pointer_cast(input_values.data()),
    thrust::raw_pointer_cast(sorted_values.data()),
    total_items,
    total_segments,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1);

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin);

  cub::DeviceSegmentedRadixSort::SortPairs(
    thrust::raw_pointer_cast(tmp_storage.data()),
    temp_storage_bytes,
    thrust::raw_pointer_cast(input_keys.data()),
    thrust::raw_pointer_cast(sorted_keys.data()),
    thrust::raw_pointer_cast(input_values.data()),
    thrust::raw_pointer_cast(sorted_values.data()),
    total_items,
    total_segments,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1);

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms {};
  cudaEventElapsedTime(&ms, begin, end);

  if (!CheckResult(offsets, sorted_keys))
    throw std::runtime_error("Wrong result");

  return ms;
}

struct ComparisonResult
{
  std::size_t segments_num;
  float base_sorting_time;
  float lrb_balancing_time;
  float radix_balancing_time;
  float partition_balancing_time;

  ComparisonResult(std::size_t segments_num,
                   float base_sorting_time,
                   float lrb_balancing_time,
                   float radix_balancing_time,
                   float partition_balancing_time)
    : segments_num(segments_num)
    , base_sorting_time(base_sorting_time)
    , lrb_balancing_time(lrb_balancing_time)
    , radix_balancing_time(radix_balancing_time)
    , partition_balancing_time(partition_balancing_time)
  {

  }
};

ComparisonResult
CompareLRB(const thrust::device_vector<std::uint32_t> &segment_sizes)
{
  thrust::device_vector<int> result(segment_sizes.size());
  thrust::device_vector<std::uint32_t> offsets =
    sizes_to_offsets(segment_sizes);

  thrust::device_vector<std::uint32_t> input_keys = GenKeys(offsets);
  thrust::device_vector<std::uint32_t> input_values(input_keys.size());

  thrust::device_vector<std::uint32_t> output_keys(input_keys.size());
  thrust::device_vector<std::uint32_t> output_values(input_keys.size());

  const float base_sorting_time = RunSegmentedSort(offsets,
                                                   input_keys,
                                                   input_values,
                                                   output_keys,
                                                   output_values);

  const float lrb_elapsed = MeasureLRB(segment_sizes, result);
  const float radix_elapsed = MeasureRadixSort8B(segment_sizes, result);
  const float partition_elapsed = MeasurePartition(segment_sizes, result);

  return ComparisonResult(segment_sizes.size(),
                          base_sorting_time,
                          lrb_elapsed,
                          radix_elapsed,
                          partition_elapsed);
}

enum class workload_distribution_pattern : int
{
  constant_small = 0,
  constant_big = 1,
  ratio = 2,
  uniform = 3,
  normal_small= 4,
  normal_mid = 5,
  normal_big = 6,
};

struct SegmentSizeInGroup
{
  std::uint32_t group_size;
  std::uint32_t big_segments_in_group;
  std::uint32_t small_segment_size;
  std::uint32_t big_segment_size;

  SegmentSizeInGroup(std::uint32_t group_size,
                     std::uint32_t big_segments_in_group,
                     std::uint32_t small_segment_size,
                     std::uint32_t big_segment_size)
      : group_size(group_size)
      , big_segments_in_group(big_segments_in_group)
      , small_segment_size(small_segment_size)
      , big_segment_size(big_segment_size)
  {}

  __device__ std::uint32_t operator()(int segment_id) const
  {
    const std::uint32_t segment_id_in_group = segment_id % group_size;
    return segment_id_in_group < big_segments_in_group
                               ? big_segment_size
                               : small_segment_size;
  }
};

struct UniformDistributionToSegmentSize
{
  std::uint32_t max_value;

  explicit UniformDistributionToSegmentSize(std::uint32_t max_value)
      : max_value(max_value)
  {}

  __device__ std::uint32_t operator()(float x) const
  {
    return static_cast<std::uint32_t>(ceilf(x * static_cast<float>(max_value)));
  }
};

struct NormalDistributionToSegmentSize
{
  std::uint32_t lower_limit;
  std::uint32_t upper_limit;

  NormalDistributionToSegmentSize(std::uint32_t lower_limit,
                                  std::uint32_t upper_limit)
      : lower_limit(lower_limit)
      , upper_limit(upper_limit)
  {}

  __device__ std::uint32_t operator()(float x) const
  {
    const auto rounded_value = static_cast<std::uint32_t>(ceilf(fabs(x)));

    if (rounded_value < lower_limit)
      return lower_limit;
    if (rounded_value > upper_limit)
      return upper_limit;

    return rounded_value;
  }
};

thrust::device_vector<std::uint32_t> ReadWorkload(const std::string &filename)
{
  std::ifstream is(filename, std::ios::binary);

  std::size_t segments_num = 0;
  is.read(reinterpret_cast<char*>(&segments_num), sizeof(std::size_t));

  thrust::host_vector<std::uint32_t> result(segments_num);
  std::uint32_t *result_data = thrust::raw_pointer_cast(result.data());

  for (unsigned int segment_id = 0; segment_id < segments_num; segment_id++)
  {
    is.read(reinterpret_cast<char *>(result_data + segment_id),
            sizeof(std::uint32_t));
  }

  return result;
}

thrust::device_vector<std::uint32_t> GenWorkload(CommandLineArgs &args,
                                                 curandGenerator_t &gen,
                                                 int total_segments)
{
  // Default values
  std::uint32_t pattern_id = 2;             // groups
  std::uint32_t group_size = 1024;          // 1024 segments form a group (ratio pattern)
  std::uint32_t big_segments_in_group = 1;  // one of the 100 segments in group is big
  std::uint32_t small_segment_size = 32;    // each small segment has 32 items in it
  std::uint32_t big_segment_size = 1 << 16; // each big segment has 65'536 items in it

  args.GetCmdLineArgument("pattern", pattern_id);
  args.GetCmdLineArgument("group-size", group_size);
  args.GetCmdLineArgument("beg-segments-in-group", big_segments_in_group);
  args.GetCmdLineArgument("small-segment-size", small_segment_size);
  args.GetCmdLineArgument("big-segment-size", big_segment_size);

  const auto pattern = static_cast<workload_distribution_pattern>(pattern_id);

  thrust::device_vector<std::uint32_t> segment_sizes(total_segments);
  thrust::device_vector<std::float_t> distribution(total_segments);

  switch(pattern)
  {
    case workload_distribution_pattern::constant_small:
    {
      thrust::fill(segment_sizes.begin(), segment_sizes.end(), small_segment_size);
      break;
    }
    case workload_distribution_pattern::constant_big:
    {
      thrust::fill(segment_sizes.begin(), segment_sizes.end(), big_segment_size);
      break;
    }
    case workload_distribution_pattern::ratio:
    {
      thrust::tabulate(segment_sizes.begin(),
                       segment_sizes.end(),
                       SegmentSizeInGroup(group_size,
                                          big_segments_in_group,
                                          small_segment_size,
                                          big_segment_size));
      break;
    }
    case workload_distribution_pattern::uniform:
    {
      CURandCall(
        curandGenerateUniform(gen,
                              thrust::raw_pointer_cast(distribution.data()),
                              total_segments));
      thrust::transform(distribution.begin(),
                        distribution.end(),
                        segment_sizes.begin(),
                        UniformDistributionToSegmentSize(big_segment_size));
      break;
    }
    case workload_distribution_pattern::normal_small:
    {
      const auto stddev = static_cast<float>((big_segment_size - small_segment_size)) / 2.0f;
      const auto mean = static_cast<float>(small_segment_size);

      CURandCall(
        curandGenerateNormal(gen,
                             thrust::raw_pointer_cast(distribution.data()),
                             total_segments,
                             mean,
                             stddev));
      thrust::transform(distribution.begin(),
                        distribution.end(),
                        segment_sizes.begin(),
                        NormalDistributionToSegmentSize(0, big_segment_size));
      break;
    }
    case workload_distribution_pattern::normal_mid:
    {
      const auto stddev = static_cast<float>((big_segment_size - small_segment_size)) / 2.0f;
      const auto mean = static_cast<float>(small_segment_size) + stddev;

      CURandCall(
        curandGenerateNormal(gen,
                             thrust::raw_pointer_cast(distribution.data()),
                             total_segments,
                             mean,
                             stddev));
      thrust::transform(distribution.begin(),
                        distribution.end(),
                        segment_sizes.begin(),
                        NormalDistributionToSegmentSize(0, big_segment_size));
      break;
    }
    case workload_distribution_pattern::normal_big:
    {
      const auto stddev = static_cast<float>((big_segment_size - small_segment_size)) / 2.0f;
      const auto mean = static_cast<float>(big_segment_size);

      CURandCall(
        curandGenerateNormal(gen,
                             thrust::raw_pointer_cast(distribution.data()),
                             total_segments,
                             mean,
                             stddev));
      thrust::transform(distribution.begin(),
                        distribution.end(),
                        segment_sizes.begin(),
                        NormalDistributionToSegmentSize(0, big_segment_size));
      break;
    }
  }

  return segment_sizes;
}

struct LessThan
{
  std::uint32_t threshold;

  explicit LessThan(std::uint32_t threshold)
      : threshold(threshold)
  {}

  __device__ bool operator()(std::uint32_t val) const
  {
    return val < threshold;
  }
};

void CalculateAndPrintHistogram(
  const thrust::device_vector<std::uint32_t> &segment_sizes)
{
  const std::uint32_t bins_count = 32;
  thrust::host_vector<long> bin_counters(bins_count + 1, 0l);

  for (std::uint32_t power_of_two = 0; power_of_two < bins_count; power_of_two++)
  {
    const std::uint32_t bin_boundary = 1u << power_of_two;

    bin_counters[power_of_two + 1] = thrust::count_if(segment_sizes.begin(),
                                                      segment_sizes.end(),
                                                      LessThan(bin_boundary));
  }

  thrust::adjacent_difference(bin_counters.begin(),
                              bin_counters.end(),
                              bin_counters.begin());

  for (std::size_t i = 0; i < bin_counters.size() - 1; i++)
  {
    std::cout << "bin[" << i << "]: " << bin_counters[i + 1] << "\n";
  }
}

int main(int argc, char *argv[])
{
  CommandLineArgs args(argc, argv);

  args.DeviceInit();

  curandGenerator_t gen;

  // Create a new generator
  CURandCall(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

  // Set the generator options
  CURandCall(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

  // Default values
  int total_segments_begin = 8; // 256 segments
  int total_segments_end = 31;  // 2147483648 segments
  int segments_step = 2;

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    std::cout << argv[0]
              << "\t [--seg-begin]\n"
              << "\t [--seg-end]\n"
              << "\t [--seg-step]\n"
              << "\t [--pattern=(0 - ConstantSmall; 1 - ConstantBig;\n"
              << "\t             2 - Ratio; 3 - Uniform;\n"
              << "\t             4 - NormalSmall; 5 - NormalMid;\n"
              << "\t             6 - NormalBig; ]\n"
              << "\t [--group-size] \n"
              << "\t [--big-segments-in-group] \n"
              << "\t [--small-segment-size] \n"
              << "\t [--big-segment-size] \n"
              << "\t [--histogram] \n"
              << "\t [--segment-sizes-file] \n"
              << std::endl;
    return 0;
  }

  args.GetCmdLineArgument("seg-begin", total_segments_begin);
  args.GetCmdLineArgument("seg-end", total_segments_end);
  args.GetCmdLineArgument("seg-step", segments_step);

  std::string segment_sizes_filename;
  args.GetCmdLineArgument("segment-sizes-file", segment_sizes_filename);

  bool print_histogram = args.CheckCmdLineFlag("histogram");

  if (segment_sizes_filename.empty())
  {
    for (int power_of_two = total_segments_begin;
         power_of_two < total_segments_end;
         power_of_two += segments_step)
    {
      const int total_segments = 1 << power_of_two;
      const auto d_segment_sizes = GenWorkload(args, gen, total_segments);

      if (print_histogram)
      {
        CalculateAndPrintHistogram(d_segment_sizes);
      }
      else
      {
        CompareLRB(d_segment_sizes);
      }
    }
  }
  else
  {
    const auto d_segment_sizes = ReadWorkload(segment_sizes_filename);

    if (print_histogram)
    {
      CalculateAndPrintHistogram(d_segment_sizes);
    }
    else
    {
      ComparisonResult result = CompareLRB(d_segment_sizes);
      std::cout << result.base_sorting_time << " - "
                << result.partition_balancing_time << std::endl;
    }
  }

  CURandCall(curandDestroyGenerator(gen));
}
