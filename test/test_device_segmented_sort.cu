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
 * Test of DeviceMergeSort utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_segmented_sort.cuh>
#include <test_util.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/reduce.h>

using namespace cub;

class SizeGroupDescription
{
public:
  SizeGroupDescription(const int segments,
                       const int segment_size)
      : segments(segments)
      , segment_size(segment_size)
  {}

  int segments {};
  int segment_size {};
};

template <typename KeyT,
          typename OffsetT>
struct SegmentChecker
{
  const KeyT *sorted_keys {};
  const OffsetT *offsets {};

  explicit SegmentChecker(const KeyT *sorted_keys,
                          const OffsetT *offsets)
    : sorted_keys(sorted_keys)
    , offsets(offsets)
  {}

  __device__ bool operator()(unsigned int segment_id)
  {
    const unsigned int segment_begin = offsets[segment_id];
    const unsigned int segment_end = offsets[segment_id + 1];

    unsigned int counter = 0;
    for (unsigned int i = segment_begin; i < segment_end; i++)
    {
      if (sorted_keys[i] != counter++)
      {
        return false;
      }
    }

    return true;
  }
};

template <typename KeyT,
          typename OffsetT>
struct ReversedIOTA
{
  KeyT *data {};
  const OffsetT *offsets {};

  ReversedIOTA(KeyT *data,
               const OffsetT *offsets)
    : data(data)
    , offsets(offsets)
  {}

  __device__ void operator()(unsigned int segment_id) const
  {
    const unsigned int segment_begin = offsets[segment_id];
    const unsigned int segment_end = offsets[segment_id + 1];
    const unsigned int segment_size = segment_end - segment_begin;

    unsigned int count = 0;
    for (unsigned int i = segment_begin; i < segment_end; i++)
    {
      data[i] = segment_size - 1 - count++;
    }
  }
};


template <typename KeyT,
  typename OffsetT>
class Input
{
  thrust::default_random_engine random_engine;
  thrust::device_vector<OffsetT> d_segment_sizes;
  thrust::device_vector<OffsetT> d_offsets;

  unsigned int num_items {};
  thrust::device_vector<KeyT> d_keys;

public:
  explicit Input(const thrust::host_vector<OffsetT> &h_segment_sizes)
    : d_segment_sizes(h_segment_sizes)
    , d_offsets(d_segment_sizes.size() + 1)
    , num_items(thrust::reduce(d_segment_sizes.begin(), d_segment_sizes.end()))
    , d_keys(num_items)
  {
    update();
  }

  void shuffle()
  {
    thrust::shuffle(d_segment_sizes.begin(), d_segment_sizes.end(), random_engine);

    update();
  }

  std::size_t get_num_items() const
  {
    return num_items;
  }

  std::size_t get_num_segments() const
  {
    return d_segment_sizes.size();
  }

  const KeyT *get_d_keys() const
  {
    return thrust::raw_pointer_cast(d_keys.data());
  }

  KeyT *get_d_keys()
  {
    return thrust::raw_pointer_cast(d_keys.data());
  }

  const OffsetT *get_d_offsets() const
  {
    return thrust::raw_pointer_cast(d_offsets.data());
  }

  bool check_output(const KeyT *keys_output)
  {
    thrust::device_vector<bool> is_segment_sorted(get_num_segments(), true);

    thrust::transform(
      thrust::counting_iterator<unsigned int>(0),
      thrust::counting_iterator<unsigned int>(get_num_segments()),
      is_segment_sorted.begin(),
      SegmentChecker<KeyT, OffsetT>(keys_output, get_d_offsets()));

    return thrust::reduce(is_segment_sorted.begin(),
                          is_segment_sorted.end(),
                          true,
                          thrust::logical_and<bool>());
  }

private:
  void update()
  {
    fill_offsets();
    gen_keys();
  }

  void fill_offsets()
  {
    thrust::copy(d_segment_sizes.begin(), d_segment_sizes.end(), d_offsets.begin());
    thrust::exclusive_scan(d_offsets.begin(), d_offsets.end(), d_offsets.begin(), 0u);
  }

  void gen_keys()
  {
    const unsigned int total_segments = get_num_segments();

    thrust::for_each(thrust::counting_iterator<unsigned int>(0),
                     thrust::counting_iterator<unsigned int>(total_segments),
                     ReversedIOTA<KeyT, OffsetT>(get_d_keys(),
                                                 get_d_offsets()));
  }
};

template <typename KeyT, typename OffsetT>
class InputDescription
{
  thrust::host_vector<OffsetT> segment_sizes;

public:
  InputDescription& add(const SizeGroupDescription &group)
  {
    if (static_cast<std::size_t>(group.segment_size) <
        static_cast<std::size_t>(std::numeric_limits<KeyT>::max()))
    {
      for (int i = 0; i < group.segments; i++)
      {
        segment_sizes.push_back(group.segment_size);
      }
    }

    return *this;
  }

  Input<KeyT, OffsetT> gen()
  {
    return Input<KeyT, OffsetT>(segment_sizes);
  }
};

template <typename KeyT,
          typename OffsetT>
void TestZeroSegments()
{
  const OffsetT *d_offsets = nullptr;
  const KeyT *keys_input = nullptr;
  KeyT *keys_output      = nullptr;

  std::size_t temp_storage_bytes = 42ul;
  CubDebugExit(cub::DeviceSegmentedSort::SortKeys(nullptr,
                                                  temp_storage_bytes,
                                                  keys_input,
                                                  keys_output,
                                                  OffsetT{},
                                                  OffsetT{},
                                                  d_offsets,
                                                  d_offsets + 1,
                                                  0,
                                                  true));

  AssertEquals(temp_storage_bytes, 0ul);

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(cub::DeviceSegmentedSort::SortKeys(d_tmp_storage,
                                                  temp_storage_bytes,
                                                  keys_input,
                                                  keys_output,
                                                  OffsetT{},
                                                  OffsetT{},
                                                  d_offsets,
                                                  d_offsets + 1,
                                                  0,
                                                  true));
}

template <typename KeyT,
          typename OffsetT>
void TestEmptySegments(OffsetT segments)
{
  thrust::device_vector<OffsetT> offsets(segments + 1, OffsetT{});
  const OffsetT *d_offsets = thrust::raw_pointer_cast(offsets.data());

  const KeyT *keys_input = nullptr;
  KeyT *keys_output      = nullptr;

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortKeys(nullptr,
                                                  temp_storage_bytes,
                                                  keys_input,
                                                  keys_output,
                                                  OffsetT{},
                                                  segments,
                                                  d_offsets,
                                                  d_offsets + 1,
                                                  0,
                                                  true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(cub::DeviceSegmentedSort::SortKeys(d_tmp_storage,
                                                  temp_storage_bytes,
                                                  keys_input,
                                                  keys_output,
                                                  OffsetT{},
                                                  segments,
                                                  d_offsets,
                                                  d_offsets + 1,
                                                  0,
                                                  true));
}

template <typename KeyT, typename OffsetT>
void TestSameSizeSegments(OffsetT segment_size, OffsetT segments)
{
  thrust::device_vector<OffsetT> offsets(segments + 1);
  thrust::sequence(offsets.begin(), offsets.end(), OffsetT{}, OffsetT{segment_size});

  const OffsetT *d_offsets = thrust::raw_pointer_cast(offsets.data());

  thrust::device_vector<KeyT> keys_input(segments * segment_size, KeyT{42});
  thrust::device_vector<KeyT> keys_output(segments * segment_size, KeyT{24});

  const KeyT *d_keys_input = thrust::raw_pointer_cast(keys_input.data());
  KeyT *d_keys_output      = thrust::raw_pointer_cast(keys_output.data());

  const OffsetT num_items = segment_size * segments;

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortKeys(nullptr,
                                                  temp_storage_bytes,
                                                  d_keys_input,
                                                  d_keys_output,
                                                  num_items,
                                                  segments,
                                                  d_offsets,
                                                  d_offsets + 1,
                                                  0,
                                                  true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(cub::DeviceSegmentedSort::SortKeys(d_tmp_storage,
                                                  temp_storage_bytes,
                                                  d_keys_input,
                                                  d_keys_output,
                                                  num_items,
                                                  segments,
                                                  d_offsets,
                                                  d_offsets + 1,
                                                  0,
                                                  true));

  AssertEquals(keys_input, keys_output);
}

template <typename KeyT,
          typename OffsetT>
void TestSingleItemSegments(OffsetT segments)
{
  TestSameSizeSegments<KeyT, OffsetT>(OffsetT{1}, segments);
}

template <typename KeyT,
          typename OffsetT>
void IndependentTest()
{
  TestZeroSegments<KeyT, OffsetT>();
}

template <typename KeyT,
          typename OffsetT>
void DependentTest(OffsetT segments)
{
  TestEmptySegments<KeyT, OffsetT>(segments);
  TestSingleItemSegments<KeyT, OffsetT>(segments);
  TestSameSizeSegments<KeyT, OffsetT>(42 * 1024, segments);
}

template <typename KeyT,
          typename OffsetT>
void DependentTest()
{
  DependentTest<KeyT, OffsetT>(42);
}

template <typename KeyT,
          typename OffsetT>
void InputTest(Input<KeyT, OffsetT> &input)
{
  int max_iterations = 100;

  thrust::device_vector<KeyT> keys_output(input.get_num_items());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortKeys(nullptr,
                                                  temp_storage_bytes,
                                                  input.get_d_keys(),
                                                  d_keys_output,
                                                  input.get_num_items(),
                                                  input.get_num_segments(),
                                                  input.get_d_offsets(),
                                                  input.get_d_offsets() + 1,
                                                  0,
                                                  true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});

    CubDebugExit(cub::DeviceSegmentedSort::SortKeys(d_tmp_storage,
                                                    temp_storage_bytes,
                                                    input.get_d_keys(),
                                                    d_keys_output,
                                                    input.get_num_items(),
                                                    input.get_num_segments(),
                                                    input.get_d_offsets(),
                                                    input.get_d_offsets() + 1,
                                                    0,
                                                    true));

    AssertTrue(input.check_output(d_keys_output));

    input.shuffle();
  }
}

template <typename KeyT,
          typename OffsetT>
void EdgePatternsTest()
{
  Input<KeyT, OffsetT> edge_cases = InputDescription<KeyT, OffsetT>()
                                      .add({420, 0})
                                      .add({420, 1})
                                      .add({420, 2})
                                      .add({420, 8})
                                      .add({420, 9})
                                      .add({420, 10})
                                      .add({420, 17})
                                      .add({42,  18})
                                      .add({42,  19})
                                      .add({42,  26})
                                      .add({42,  27})
                                      .add({42,  28})
                                      .add({42,  35})
                                      .add({42,  36})
                                      .add({42,  37})
                                      .add({42,  286})
                                      .add({42,  287})
                                      .add({42,  288})
                                      .add({42,  5887})
                                      .add({42,  5888})
                                      .add({42,  5889})
                                      .add({2,  23552})
                                      .gen();

  InputTest(edge_cases);
}

template <typename KeyT,
          typename OffsetT>
void Test()
{
  IndependentTest<KeyT, OffsetT>();
  DependentTest<KeyT, OffsetT>();
  EdgePatternsTest<KeyT, OffsetT>();
}

// TODO Test SortKeys
//      Patterns
//      Random
// TODO Test SortPairs
//      Patterns
//      Random

int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  // Test<std::uint8_t,  std::uint32_t>();
  // Test<std::uint16_t, std::uint32_t>();
  Test<std::uint32_t, std::uint32_t>();
  // Test<std::uint64_t, std::uint32_t>();
  // Test<std::uint32_t, std::int64_t>();

  return 0;
}
