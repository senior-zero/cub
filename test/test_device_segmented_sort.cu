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

#include <cub/device/device_segmented_sort.cuh>
#include <test_util.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/reduce.h>

using namespace cub;


constexpr static int MAX_ITERATIONS = 10;


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
      if (sorted_keys[i] != static_cast<KeyT>(counter++))
      {
        return false;
      }
    }

    return true;
  }
};

template <typename KeyT,
          typename OffsetT>
struct DescendingSegmentChecker
{
  const KeyT *sorted_keys{};
  const OffsetT *offsets{};

  explicit DescendingSegmentChecker(const KeyT *sorted_keys,
                                    const OffsetT *offsets)
      : sorted_keys(sorted_keys)
      , offsets(offsets)
  {}

  __device__ bool operator()(unsigned int segment_id)
  {
    const int segment_begin = static_cast<int>(offsets[segment_id]);
    const int segment_end   = static_cast<int>(offsets[segment_id + 1]);

    unsigned int counter = 0;
    for (int i = segment_end - 1; i >= segment_begin; i--)
    {
      if (sorted_keys[i] != static_cast<KeyT>(counter++))
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
      data[i] = static_cast<KeyT>(segment_size - 1 - count++);
    }
  }
};


template <typename KeyT,
          typename OffsetT>
struct IOTA
{
  KeyT *data{};
  const OffsetT *offsets{};

  IOTA(KeyT *data, const OffsetT *offsets)
      : data(data)
      , offsets(offsets)
  {}

  __device__ void operator()(unsigned int segment_id) const
  {
    const unsigned int segment_begin = offsets[segment_id];
    const unsigned int segment_end   = offsets[segment_id + 1];

    unsigned int count = 0;
    for (unsigned int i = segment_begin; i < segment_end; i++)
    {
      data[i] = static_cast<KeyT>(count++);
    }
  }
};


template <typename KeyT,
          typename OffsetT,
          typename ValueT = cub::NullType>
class Input
{
  thrust::default_random_engine random_engine;
  thrust::device_vector<OffsetT> d_segment_sizes;
  thrust::device_vector<OffsetT> d_offsets;
  thrust::host_vector<OffsetT> h_offsets;

  using MaskedValueT = typename std::conditional<
    std::is_same<ValueT, cub::NullType>::value,
    KeyT,
    ValueT>::type;

  bool reverse {};
  unsigned int num_items {};
  thrust::device_vector<KeyT> d_keys;
  thrust::device_vector<MaskedValueT> d_values;

public:
  Input(
    bool reverse,
    const thrust::host_vector<OffsetT> &h_segment_sizes)
    : d_segment_sizes(h_segment_sizes)
    , d_offsets(d_segment_sizes.size() + 1)
    , h_offsets(d_segment_sizes.size() + 1)
    , reverse(reverse)
    , num_items(thrust::reduce(d_segment_sizes.begin(), d_segment_sizes.end()))
    , d_keys(num_items)
    , d_values(num_items)
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

  thrust::device_vector<KeyT> &get_d_keys_vec()
  {
    return d_keys;
  }

  thrust::device_vector<MaskedValueT> &get_d_values_vec()
  {
    return d_values;
  }

  KeyT *get_d_keys()
  {
    return thrust::raw_pointer_cast(d_keys.data());
  }

  const thrust::host_vector<OffsetT>& get_h_offsets()
  {
    return h_offsets;
  }

  MaskedValueT *get_d_values()
  {
    return thrust::raw_pointer_cast(d_values.data());
  }

  const OffsetT *get_d_offsets() const
  {
    return thrust::raw_pointer_cast(d_offsets.data());
  }

  template <typename T>
  bool check_output_implementation(const T *keys_output)
  {
    thrust::device_vector<bool> is_segment_sorted(get_num_segments(), true);

    if (reverse)
    {
      thrust::transform(
        thrust::counting_iterator<unsigned int>(0),
                        thrust::counting_iterator<unsigned int>(
                          static_cast<unsigned int>(get_num_segments())),
                        is_segment_sorted.begin(),
                        DescendingSegmentChecker<T, OffsetT>(keys_output,
                                                             get_d_offsets()));
    }
    else
    {
      thrust::transform(
        thrust::counting_iterator<unsigned int>(0),
                        thrust::counting_iterator<unsigned int>(
                          static_cast<unsigned int>(get_num_segments())),
                        is_segment_sorted.begin(),
                        SegmentChecker<T, OffsetT>(keys_output,
                                                   get_d_offsets()));
    }

    return thrust::reduce(is_segment_sorted.begin(),
                          is_segment_sorted.end(),
                          true,
                          thrust::logical_and<bool>());
  }

  bool check_output(const KeyT *keys_output,
                    const MaskedValueT *values_output = nullptr)
  {
    const bool keys_ok = check_output_implementation(keys_output);
    const bool values_ok = std::is_same<ValueT, cub::NullType>::value
                         ? true
                         : check_output_implementation(values_output);

    return keys_ok && values_ok;
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
    thrust::copy(d_offsets.begin(), d_offsets.end(), h_offsets.begin());
  }

  void gen_keys()
  {
    const unsigned int total_segments =
      static_cast<unsigned int>(get_num_segments());

    if (reverse)
    {
      thrust::for_each(thrust::counting_iterator<unsigned int>(0),
                       thrust::counting_iterator<unsigned int>(total_segments),
                       IOTA<KeyT, OffsetT>(get_d_keys(),
                                           get_d_offsets()));
    }
    else
    {
      thrust::for_each(thrust::counting_iterator<unsigned int>(0),
                       thrust::counting_iterator<unsigned int>(total_segments),
                       ReversedIOTA<KeyT, OffsetT>(get_d_keys(),
                                                   get_d_offsets()));
    }

    thrust::copy(d_keys.begin(), d_keys.end(), d_values.begin());
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

  template <typename ValueT = cub::NullType>
  Input<KeyT, OffsetT, ValueT> gen(bool reverse)
  {
    return Input<KeyT, OffsetT, ValueT>(reverse, segment_sizes);
  }
};

template <typename KeyT,
          typename OffsetT>
void TestZeroSegmentsBuffer()
{
  const OffsetT *d_offsets = nullptr;
  cub::DoubleBuffer<KeyT> buffer(nullptr, nullptr);

  std::size_t temp_storage_bytes = 42ul;
  CubDebugExit(cub::DeviceSegmentedSort::SortKeys(nullptr,
                                                  temp_storage_bytes,
                                                  buffer,
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
                                                  buffer,
                                                  OffsetT{},
                                                  OffsetT{},
                                                  d_offsets,
                                                  d_offsets + 1,
                                                  0,
                                                  true));
  }

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
void TestZeroSegmentsDescending()
{
  const OffsetT *d_offsets = nullptr;
  const KeyT *keys_input   = nullptr;
  KeyT *keys_output        = nullptr;

  std::size_t temp_storage_bytes = 42ul;
  CubDebugExit(cub::DeviceSegmentedSort::SortKeysDescending(nullptr,
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

  CubDebugExit(cub::DeviceSegmentedSort::SortKeysDescending(d_tmp_storage,
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
void TestZeroSegmentsDescendingBuffer()
{
  const OffsetT *d_offsets = nullptr;
  cub::DoubleBuffer<KeyT> buffer(nullptr, nullptr);

  std::size_t temp_storage_bytes = 42ul;
  CubDebugExit(cub::DeviceSegmentedSort::SortKeysDescending(nullptr,
                                                            temp_storage_bytes,
                                                            buffer,
                                                            OffsetT{},
                                                            OffsetT{},
                                                            d_offsets,
                                                            d_offsets + 1,
                                                            0,
                                                            true));

  AssertEquals(temp_storage_bytes, 0ul);

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(cub::DeviceSegmentedSort::SortKeysDescending(d_tmp_storage,
                                                            temp_storage_bytes,
                                                            buffer,
                                                            OffsetT{},
                                                            OffsetT{},
                                                            d_offsets,
                                                            d_offsets + 1,
                                                            0,
                                                            true));
  }

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void TestZeroSegmentsPairs()
{
  const OffsetT *d_offsets = nullptr;
  const KeyT *keys_input   = nullptr;
  KeyT *keys_output        = nullptr;

  const ValueT *values_input = nullptr;
  ValueT *values_output      = nullptr;

  std::size_t temp_storage_bytes = 42ul;
  CubDebugExit(cub::DeviceSegmentedSort::SortPairs(nullptr,
                                                   temp_storage_bytes,
                                                   keys_input,
                                                   keys_output,
                                                   values_input,
                                                   values_output,
                                                   OffsetT{},
                                                   OffsetT{},
                                                   d_offsets,
                                                   d_offsets + 1,
                                                   0,
                                                   true));

  AssertEquals(temp_storage_bytes, 0ul);

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(cub::DeviceSegmentedSort::SortPairs(d_tmp_storage,
                                                   temp_storage_bytes,
                                                   keys_input,
                                                   keys_output,
                                                   values_input,
                                                   values_output,
                                                   OffsetT{},
                                                   OffsetT{},
                                                   d_offsets,
                                                   d_offsets + 1,
                                                   0,
                                                   true));
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void TestZeroSegmentsPairsBuffer()
{
  const OffsetT *d_offsets = nullptr;

  cub::DoubleBuffer<KeyT> keys(nullptr, nullptr);
  cub::DoubleBuffer<ValueT> values(nullptr, nullptr);

  std::size_t temp_storage_bytes = 42ul;
  CubDebugExit(cub::DeviceSegmentedSort::SortPairs(nullptr,
                                                   temp_storage_bytes,
                                                   keys,
                                                   values,
                                                   OffsetT{},
                                                   OffsetT{},
                                                   d_offsets,
                                                   d_offsets + 1,
                                                   0,
                                                   true));

  AssertEquals(temp_storage_bytes, 0ul);

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(cub::DeviceSegmentedSort::SortPairs(d_tmp_storage,
                                                   temp_storage_bytes,
                                                   keys,
                                                   values,
                                                   OffsetT{},
                                                   OffsetT{},
                                                   d_offsets,
                                                   d_offsets + 1,
                                                   0,
                                                   true));
  }

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void TestZeroSegmentsDescendingPairs()
{
  const OffsetT *d_offsets = nullptr;
  const KeyT *keys_input   = nullptr;
  KeyT *keys_output        = nullptr;

  const ValueT *values_input = nullptr;
  ValueT *values_output      = nullptr;

  std::size_t temp_storage_bytes = 42ul;
  CubDebugExit(cub::DeviceSegmentedSort::SortPairsDescending(nullptr,
                                                             temp_storage_bytes,
                                                             keys_input,
                                                             keys_output,
                                                             values_input,
                                                             values_output,
                                                             OffsetT{},
                                                             OffsetT{},
                                                             d_offsets,
                                                             d_offsets + 1,
                                                             0,
                                                             true));

  AssertEquals(temp_storage_bytes, 0ul);

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(cub::DeviceSegmentedSort::SortPairsDescending(d_tmp_storage,
                                                             temp_storage_bytes,
                                                             keys_input,
                                                             keys_output,
                                                             values_input,
                                                             values_output,
                                                             OffsetT{},
                                                             OffsetT{},
                                                             d_offsets,
                                                             d_offsets + 1,
                                                             0,
                                                             true));
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void TestZeroSegmentsDescendingPairsBuffer()
{
  const OffsetT *d_offsets = nullptr;
  cub::DoubleBuffer<KeyT> keys(nullptr, nullptr);
  cub::DoubleBuffer<ValueT> values(nullptr, nullptr);

  std::size_t temp_storage_bytes = 42ul;
  CubDebugExit(cub::DeviceSegmentedSort::SortPairsDescending(nullptr,
                                                             temp_storage_bytes,
                                                             keys,
                                                             values,
                                                             OffsetT{},
                                                             OffsetT{},
                                                             d_offsets,
                                                             d_offsets + 1,
                                                             0,
                                                             true));

  AssertEquals(temp_storage_bytes, 0ul);

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(cub::DeviceSegmentedSort::SortPairsDescending(d_tmp_storage,
                                                             temp_storage_bytes,
                                                             keys,
                                                             values,
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

template <typename KeyT,
          typename OffsetT>
void TestEmptySegmentsBuffer(OffsetT segments)
{
  thrust::device_vector<OffsetT> offsets(segments + 1, OffsetT{});
  const OffsetT *d_offsets = thrust::raw_pointer_cast(offsets.data());

  cub::DoubleBuffer<KeyT> keys(nullptr, nullptr);

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortKeys(nullptr,
                                                  temp_storage_bytes,
                                                  keys,
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
                                                  keys,
                                                  OffsetT{},
                                                  segments,
                                                  d_offsets,
                                                  d_offsets + 1,
                                                  0,
                                                  true));
  }

template <typename KeyT,
          typename OffsetT>
void TestEmptySegmentsDescending(OffsetT segments)
{
  thrust::device_vector<OffsetT> offsets(segments + 1, OffsetT{});
  const OffsetT *d_offsets = thrust::raw_pointer_cast(offsets.data());

  const KeyT *keys_input = nullptr;
  KeyT *keys_output      = nullptr;

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortKeysDescending(nullptr,
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

  CubDebugExit(cub::DeviceSegmentedSort::SortKeysDescending(d_tmp_storage,
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

template <typename KeyT,
          typename OffsetT>
void TestEmptySegmentsDescendingBuffer(OffsetT segments)
{
  thrust::device_vector<OffsetT> offsets(segments + 1, OffsetT{});
  const OffsetT *d_offsets = thrust::raw_pointer_cast(offsets.data());

  cub::DoubleBuffer<KeyT> keys(nullptr, nullptr);

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortKeysDescending(nullptr,
                                                            temp_storage_bytes,
                                                            keys,
                                                            OffsetT{},
                                                            segments,
                                                            d_offsets,
                                                            d_offsets + 1,
                                                            0,
                                                            true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(cub::DeviceSegmentedSort::SortKeysDescending(d_tmp_storage,
                                                            temp_storage_bytes,
                                                            keys,
                                                            OffsetT{},
                                                            segments,
                                                            d_offsets,
                                                            d_offsets + 1,
                                                            0,
                                                            true));
  }

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void TestEmptySegmentsPairs(OffsetT segments)
{
  thrust::device_vector<OffsetT> offsets(segments + 1, OffsetT{});
  const OffsetT *d_offsets = thrust::raw_pointer_cast(offsets.data());

  const KeyT *keys_input = nullptr;
  KeyT *keys_output      = nullptr;

  const ValueT *values_input = nullptr;
  ValueT *values_output      = nullptr;

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortPairs(nullptr,
                                                   temp_storage_bytes,
                                                   keys_input,
                                                   keys_output,
                                                   values_input,
                                                   values_output,
                                                   OffsetT{},
                                                   segments,
                                                   d_offsets,
                                                   d_offsets + 1,
                                                   0,
                                                   true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(cub::DeviceSegmentedSort::SortPairs(d_tmp_storage,
                                                   temp_storage_bytes,
                                                   keys_input,
                                                   keys_output,
                                                   values_input,
                                                   values_output,
                                                   OffsetT{},
                                                   segments,
                                                   d_offsets,
                                                   d_offsets + 1,
                                                   0,
                                                   true));
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void TestEmptySegmentsPairsBuffer(OffsetT segments)
{
  thrust::device_vector<OffsetT> offsets(segments + 1, OffsetT{});
  const OffsetT *d_offsets = thrust::raw_pointer_cast(offsets.data());

  KeyT *keys_input  = nullptr;
  KeyT *keys_output = nullptr;

  ValueT *values_input  = nullptr;
  ValueT *values_output = nullptr;

  cub::DoubleBuffer<KeyT> keys(keys_input, keys_output);
  cub::DoubleBuffer<ValueT> values(values_input, values_output);

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortPairs(nullptr,
                                                   temp_storage_bytes,
                                                   keys,
                                                   values,
                                                   OffsetT{},
                                                   segments,
                                                   d_offsets,
                                                   d_offsets + 1,
                                                   0,
                                                   true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(cub::DeviceSegmentedSort::SortPairs(d_tmp_storage,
                                                   temp_storage_bytes,
                                                   keys,
                                                   values,
                                                   OffsetT{},
                                                   segments,
                                                   d_offsets,
                                                   d_offsets + 1,
                                                   0,
                                                   true));
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void TestEmptySegmentsDescendingPairs(OffsetT segments)
{
  thrust::device_vector<OffsetT> offsets(segments + 1, OffsetT{});
  const OffsetT *d_offsets = thrust::raw_pointer_cast(offsets.data());

  const KeyT *keys_input = nullptr;
  KeyT *keys_output      = nullptr;

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortKeysDescending(nullptr,
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

  CubDebugExit(cub::DeviceSegmentedSort::SortKeysDescending(d_tmp_storage,
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

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void TestEmptySegmentsDescendingPairsBuffer(OffsetT segments)
{
  thrust::device_vector<OffsetT> offsets(segments + 1, OffsetT{});
  const OffsetT *d_offsets = thrust::raw_pointer_cast(offsets.data());

  KeyT *keys_input  = nullptr;
  KeyT *keys_output = nullptr;

  ValueT *values_input  = nullptr;
  ValueT *values_output = nullptr;

  cub::DoubleBuffer<KeyT> keys(keys_input, keys_output);
  cub::DoubleBuffer<ValueT> values(values_input, values_output);

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortPairsDescending(nullptr,
                                                             temp_storage_bytes,
                                                             keys,
                                                             values,
                                                             OffsetT{},
                                                             segments,
                                                             d_offsets,
                                                             d_offsets + 1,
                                                             0,
                                                             true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(cub::DeviceSegmentedSort::SortPairsDescending(d_tmp_storage,
                                                             temp_storage_bytes,
                                                             keys,
                                                             values,
                                                             OffsetT{},
                                                             segments,
                                                             d_offsets,
                                                             d_offsets + 1,
                                                             0,
                                                             true));
}

template <typename KeyT,
          typename OffsetT>
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
void TestSameSizeSegmentsBuffer(OffsetT segment_size, OffsetT segments)
{
  thrust::device_vector<OffsetT> offsets(segments + 1);
  thrust::sequence(offsets.begin(), offsets.end(), OffsetT{}, OffsetT{segment_size});

  const OffsetT num_items = segment_size * segments;

  const OffsetT *d_offsets = thrust::raw_pointer_cast(offsets.data());

  thrust::device_vector<KeyT> keys_input(num_items, KeyT{42});
  thrust::device_vector<KeyT> keys_output(num_items, KeyT{24});

  KeyT *d_keys_input = thrust::raw_pointer_cast(keys_input.data());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  cub::DoubleBuffer<KeyT> keys(d_keys_input, d_keys_output);

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortKeys(nullptr,
                                                  temp_storage_bytes,
                                                  keys,
                                                  num_items,
                                                  segments,
                                                  d_offsets,
                                                  d_offsets + 1,
                                                  0,
                                                  true));

  // If temporary storage size is defined by extra keys storage
  if (2 * segments * sizeof(OffsetT) < num_items * sizeof(KeyT))
  {
    std::size_t extra_temp_storage_bytes{};
    CubDebugExit(cub::DeviceSegmentedSort::SortKeys(nullptr,
                                                    extra_temp_storage_bytes,
                                                    d_keys_input,
                                                    d_keys_output,
                                                    num_items,
                                                    segments,
                                                    d_offsets,
                                                    d_offsets + 1,
                                                    0,
                                                    true));

    AssertTrue(extra_temp_storage_bytes > temp_storage_bytes);
  }

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(cub::DeviceSegmentedSort::SortKeys(d_tmp_storage,
                                                  temp_storage_bytes,
                                                  keys,
                                                  num_items,
                                                  segments,
                                                  d_offsets,
                                                  d_offsets + 1,
                                                  0,
                                                  true));

  const std::size_t items_selected =
    keys.selector == 1
      ? thrust::count(keys_output.begin(), keys_output.end(), KeyT{42})
      : thrust::count(keys_input.begin(), keys_input.end(), KeyT{42});
  AssertEquals(items_selected, num_items);
}

template <typename KeyT,
          typename OffsetT>
void TestSameSizeSegmentsDescending(OffsetT segment_size, OffsetT segments)
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
  CubDebugExit(cub::DeviceSegmentedSort::SortKeysDescending(nullptr,
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

  CubDebugExit(cub::DeviceSegmentedSort::SortKeysDescending(d_tmp_storage,
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
void TestSameSizeSegmentsDescendingBuffer(
  OffsetT segment_size,
  OffsetT segments)
{
  thrust::device_vector<OffsetT> offsets(segments + 1);
  thrust::sequence(offsets.begin(),
                   offsets.end(),
                   OffsetT{},
                   OffsetT{segment_size});

  const OffsetT num_items = segment_size * segments;

  const OffsetT *d_offsets = thrust::raw_pointer_cast(offsets.data());

  thrust::device_vector<KeyT> keys_input(num_items, KeyT{42});
  thrust::device_vector<KeyT> keys_output(num_items, KeyT{24});

  KeyT *d_keys_input  = thrust::raw_pointer_cast(keys_input.data());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  cub::DoubleBuffer<KeyT> keys(d_keys_input, d_keys_output);

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortKeysDescending(nullptr,
                                                            temp_storage_bytes,
                                                            keys,
                                                            num_items,
                                                            segments,
                                                            d_offsets,
                                                            d_offsets + 1,
                                                            0,
                                                            true));

  // If temporary storage size is defined by extra keys storage
  if (2 * segments * sizeof(OffsetT) < num_items * sizeof(KeyT))
  {
    std::size_t extra_temp_storage_bytes{};
    CubDebugExit(
      cub::DeviceSegmentedSort::SortKeysDescending(nullptr,
                                                   extra_temp_storage_bytes,
                                                   d_keys_input,
                                                   d_keys_output,
                                                   num_items,
                                                   segments,
                                                   d_offsets,
                                                   d_offsets + 1,
                                                   0,
                                                   true));

    AssertTrue(extra_temp_storage_bytes > temp_storage_bytes);
  }

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(cub::DeviceSegmentedSort::SortKeysDescending(d_tmp_storage,
                                                            temp_storage_bytes,
                                                            keys,
                                                            num_items,
                                                            segments,
                                                            d_offsets,
                                                            d_offsets + 1,
                                                            0,
                                                            true));

  const std::size_t items_selected =
    keys.selector
      ? thrust::count(keys_output.begin(), keys_output.end(), KeyT{42})
      : thrust::count(keys_input.begin(), keys_input.end(), KeyT{42});
  AssertEquals(items_selected, num_items);
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void TestSameSizeSegmentsPairs(OffsetT segment_size, OffsetT segments)
{
  thrust::device_vector<OffsetT> offsets(segments + 1);
  thrust::sequence(offsets.begin(),
                   offsets.end(),
                   OffsetT{},
                   OffsetT{segment_size});

  const OffsetT *d_offsets = thrust::raw_pointer_cast(offsets.data());

  thrust::device_vector<KeyT> keys_input(segments * segment_size, KeyT{42});
  thrust::device_vector<KeyT> keys_output(segments * segment_size, KeyT{24});

  const KeyT *d_keys_input = thrust::raw_pointer_cast(keys_input.data());
  KeyT *d_keys_output      = thrust::raw_pointer_cast(keys_output.data());

  thrust::device_vector<ValueT> values_input(segments * segment_size, ValueT{42});
  thrust::device_vector<ValueT> values_output(segments * segment_size, ValueT{24});

  const ValueT *d_values_input = thrust::raw_pointer_cast(values_input.data());
  ValueT *d_values_output      = thrust::raw_pointer_cast(values_output.data());

  const OffsetT num_items = segment_size * segments;

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortPairs(nullptr,
                                                   temp_storage_bytes,
                                                   d_keys_input,
                                                   d_keys_output,
                                                   d_values_input,
                                                   d_values_output,
                                                   num_items,
                                                   segments,
                                                   d_offsets,
                                                   d_offsets + 1,
                                                   0,
                                                   true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(cub::DeviceSegmentedSort::SortPairs(d_tmp_storage,
                                                   temp_storage_bytes,
                                                   d_keys_input,
                                                   d_keys_output,
                                                   d_values_input,
                                                   d_values_output,
                                                   num_items,
                                                   segments,
                                                   d_offsets,
                                                   d_offsets + 1,
                                                   0,
                                                   true));

  AssertEquals(keys_input, keys_output);
  AssertEquals(values_input, values_output);
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void TestSameSizeSegmentsPairsBuffer(OffsetT segment_size, OffsetT segments)
{
  thrust::device_vector<OffsetT> offsets(segments + 1);
  thrust::sequence(offsets.begin(),
                   offsets.end(),
                   OffsetT{},
                   OffsetT{segment_size});

  const OffsetT *d_offsets = thrust::raw_pointer_cast(offsets.data());

  const OffsetT num_items = segment_size * segments;

  thrust::device_vector<KeyT> keys_input(num_items, KeyT{42});
  thrust::device_vector<KeyT> keys_output(num_items, KeyT{24});

  KeyT *d_keys_input = thrust::raw_pointer_cast(keys_input.data());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  thrust::device_vector<ValueT> values_input(num_items, ValueT{42});
  thrust::device_vector<ValueT> values_output(num_items, ValueT{24});

  ValueT *d_values_input = thrust::raw_pointer_cast(values_input.data());
  ValueT *d_values_output = thrust::raw_pointer_cast(values_output.data());

  cub::DoubleBuffer<KeyT> keys(d_keys_input, d_keys_output);
  cub::DoubleBuffer<ValueT> values(d_values_input, d_values_output);

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortPairs(nullptr,
                                                   temp_storage_bytes,
                                                   keys,
                                                   values,
                                                   num_items,
                                                   segments,
                                                   d_offsets,
                                                   d_offsets + 1,
                                                   0,
                                                   true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(cub::DeviceSegmentedSort::SortPairs(d_tmp_storage,
                                                   temp_storage_bytes,
                                                   keys,
                                                   values,
                                                   num_items,
                                                   segments,
                                                   d_offsets,
                                                   d_offsets + 1,
                                                   0,
                                                   true));

  {
    const std::size_t items_selected =
      keys.selector
      ? thrust::count(keys_output.begin(), keys_output.end(), KeyT{42})
      : thrust::count(keys_input.begin(), keys_input.end(), KeyT{42});
    AssertEquals(items_selected, num_items);
  }

  {
    const std::size_t items_selected =
      values.selector
      ? thrust::count(values_output.begin(), values_output.end(), ValueT{42})
      : thrust::count(values_input.begin(), values_input.end(), ValueT{42});
    AssertEquals(items_selected, num_items);
  }
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void TestSameSizeSegmentsDescendingPairs(OffsetT segment_size, OffsetT segments)
{
  thrust::device_vector<OffsetT> offsets(segments + 1);
  thrust::sequence(offsets.begin(),
                   offsets.end(),
                   OffsetT{},
                   OffsetT{segment_size});

  const OffsetT *d_offsets = thrust::raw_pointer_cast(offsets.data());

  thrust::device_vector<KeyT> keys_input(segments * segment_size, KeyT{42});
  thrust::device_vector<KeyT> keys_output(segments * segment_size, KeyT{24});

  const KeyT *d_keys_input = thrust::raw_pointer_cast(keys_input.data());
  KeyT *d_keys_output      = thrust::raw_pointer_cast(keys_output.data());

  thrust::device_vector<ValueT> values_input(segments * segment_size, ValueT{42});
  thrust::device_vector<ValueT> values_output(segments * segment_size, ValueT{24});

  const ValueT *d_values_input = thrust::raw_pointer_cast(values_input.data());
  ValueT *d_values_output      = thrust::raw_pointer_cast(values_output.data());

  const OffsetT num_items = segment_size * segments;

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortPairsDescending(nullptr,
                                                             temp_storage_bytes,
                                                             d_keys_input,
                                                             d_keys_output,
                                                             d_values_input,
                                                             d_values_output,
                                                             num_items,
                                                             segments,
                                                             d_offsets,
                                                             d_offsets + 1,
                                                             0,
                                                             true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(cub::DeviceSegmentedSort::SortPairsDescending(d_tmp_storage,
                                                             temp_storage_bytes,
                                                             d_keys_input,
                                                             d_keys_output,
                                                             d_values_input,
                                                             d_values_output,
                                                             num_items,
                                                             segments,
                                                             d_offsets,
                                                             d_offsets + 1,
                                                             0,
                                                             true));

  AssertEquals(keys_input, keys_output);
  AssertEquals(values_input, values_output);
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void TestSameSizeSegmentsDescendingPairsBuffer(
  OffsetT segment_size,
  OffsetT segments)
{
  thrust::device_vector<OffsetT> offsets(segments + 1);
  thrust::sequence(offsets.begin(),
                   offsets.end(),
                   OffsetT{},
                   OffsetT{segment_size});

  const OffsetT *d_offsets = thrust::raw_pointer_cast(offsets.data());

  const OffsetT num_items = segment_size * segments;

  thrust::device_vector<KeyT> keys_input(num_items, KeyT{42});
  thrust::device_vector<KeyT> keys_output(num_items, KeyT{24});

  KeyT *d_keys_input  = thrust::raw_pointer_cast(keys_input.data());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  thrust::device_vector<ValueT> values_input(num_items, ValueT{42});
  thrust::device_vector<ValueT> values_output(num_items, ValueT{24});

  ValueT *d_values_input  = thrust::raw_pointer_cast(values_input.data());
  ValueT *d_values_output = thrust::raw_pointer_cast(values_output.data());

  cub::DoubleBuffer<KeyT> keys(d_keys_input, d_keys_output);
  cub::DoubleBuffer<ValueT> values(d_values_output, d_values_input);
  values.selector = 1;

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortPairsDescending(nullptr,
                                                             temp_storage_bytes,
                                                             keys,
                                                             values,
                                                             num_items,
                                                             segments,
                                                             d_offsets,
                                                             d_offsets + 1,
                                                             0,
                                                             true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(cub::DeviceSegmentedSort::SortPairsDescending(d_tmp_storage,
                                                             temp_storage_bytes,
                                                             keys,
                                                             values,
                                                             num_items,
                                                             segments,
                                                             d_offsets,
                                                             d_offsets + 1,
                                                             0,
                                                             true));

  {
    const std::size_t items_selected =
      keys.selector
        ? thrust::count(keys_output.begin(), keys_output.end(), KeyT{42})
        : thrust::count(keys_input.begin(), keys_input.end(), KeyT{42});
    AssertEquals(items_selected, num_items);
  }

  {
    const std::size_t items_selected =
      values.selector
        ? thrust::count(values_input.begin(), values_input.end(), ValueT{42})
        : thrust::count(values_output.begin(), values_output.end(), ValueT{42});
    AssertEquals(items_selected, num_items);
  }
}

template <typename KeyT,
          typename OffsetT>
void TestSingleItemSegments(OffsetT segments)
{
  TestSameSizeSegments<KeyT, OffsetT>(OffsetT{1}, segments);
  TestSameSizeSegmentsBuffer<KeyT, OffsetT>(OffsetT{1}, segments);
  TestSameSizeSegmentsDescending<KeyT, OffsetT>(OffsetT{1}, segments);
  TestSameSizeSegmentsDescendingBuffer<KeyT, OffsetT>(OffsetT{1}, segments);
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void TestSingleItemSegmentsPairs(OffsetT segments)
{
  TestSameSizeSegmentsPairs<KeyT, ValueT, OffsetT>(OffsetT{1}, segments);
  TestSameSizeSegmentsPairsBuffer<KeyT, ValueT, OffsetT>(OffsetT{1}, segments);
  TestSameSizeSegmentsDescendingPairs<KeyT, ValueT, OffsetT>(OffsetT{1}, segments);
  TestSameSizeSegmentsDescendingPairsBuffer<KeyT, ValueT, OffsetT>(OffsetT{1}, segments);
}


template <typename KeyT,
          typename OffsetT>
void IndependentTest()
{
  TestZeroSegments<KeyT, OffsetT>();
  TestZeroSegmentsBuffer<KeyT, OffsetT>();
  TestZeroSegmentsDescending<KeyT, OffsetT>();
  TestZeroSegmentsDescendingBuffer<KeyT, OffsetT>();
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void IndependentTestPairs()
{
  TestZeroSegmentsPairs<KeyT, ValueT, OffsetT>();
  TestZeroSegmentsPairsBuffer<KeyT, ValueT, OffsetT>();
  TestZeroSegmentsDescendingPairs<KeyT, ValueT, OffsetT>();
  TestZeroSegmentsDescendingPairsBuffer<KeyT, ValueT, OffsetT>();
}

template <typename KeyT,
          typename OffsetT>
void DependentTest(OffsetT segments)
{
  TestEmptySegments<KeyT, OffsetT>(segments);
  TestEmptySegmentsBuffer<KeyT, OffsetT>(segments);
  TestEmptySegmentsDescending<KeyT, OffsetT>(segments);
  TestEmptySegmentsDescendingBuffer<KeyT, OffsetT>(segments);

  TestSingleItemSegments<KeyT, OffsetT>(segments);

  TestSameSizeSegments<KeyT, OffsetT>(42 * 1024, segments);
  TestSameSizeSegmentsBuffer<KeyT, OffsetT>(42 * 1024, segments);
  TestSameSizeSegmentsDescending<KeyT, OffsetT>(42 * 1024, segments);
  TestSameSizeSegmentsDescendingBuffer<KeyT, OffsetT>(42 * 1024, segments);
}

template <typename KeyT,
          typename OffsetT>
void DependentTest()
{
  DependentTest<KeyT, OffsetT>(42);
  DependentTest<KeyT, OffsetT>(1024);
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void DependentTestPairs(OffsetT segments)
{
  TestEmptySegmentsPairs<KeyT, ValueT, OffsetT>(segments);
  TestEmptySegmentsPairsBuffer<KeyT, ValueT, OffsetT>(segments);
  TestEmptySegmentsDescendingPairs<KeyT, ValueT, OffsetT>(segments);
  TestEmptySegmentsDescendingPairsBuffer<KeyT, ValueT, OffsetT>(segments);

  TestSingleItemSegmentsPairs<KeyT, ValueT, OffsetT>(segments);

  TestSameSizeSegmentsPairs<KeyT, ValueT, OffsetT>(42 * 1024, segments);
  TestSameSizeSegmentsPairsBuffer<KeyT, ValueT, OffsetT>(42 * 1024, segments);
  TestSameSizeSegmentsDescendingPairs<KeyT, ValueT, OffsetT>(42 * 1024, segments);
  TestSameSizeSegmentsDescendingPairsBuffer<KeyT, ValueT, OffsetT>(42 * 1024, segments);
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void DependentTestPairs()
{
  DependentTestPairs<KeyT, ValueT, OffsetT>(42);
  DependentTestPairs<KeyT, ValueT, OffsetT>(1024);
}

template <typename KeyT,
          typename OffsetT>
void InputTest(Input<KeyT, OffsetT> &input)
{
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

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
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
void InputTestBuffer(Input<KeyT, OffsetT> &input)
{
  thrust::device_vector<KeyT> keys_output(input.get_num_items());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  cub::DoubleBuffer<KeyT> empty_keys(nullptr, nullptr);

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortKeys(nullptr,
                                                  temp_storage_bytes,
                                                  empty_keys,
                                                  input.get_num_items(),
                                                  input.get_num_segments(),
                                                  input.get_d_offsets(),
                                                  input.get_d_offsets() + 1,
                                                  0,
                                                  true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});

    cub::DoubleBuffer<KeyT> keys(input.get_d_keys(), d_keys_output);

    CubDebugExit(cub::DeviceSegmentedSort::SortKeys(d_tmp_storage,
                                                    temp_storage_bytes,
                                                    keys,
                                                    input.get_num_items(),
                                                    input.get_num_segments(),
                                                    input.get_d_offsets(),
                                                    input.get_d_offsets() + 1,
                                                    0,
                                                    true));

    AssertTrue(input.check_output(keys.Current()));

    input.shuffle();
  }
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void InputTestPairs(Input<KeyT, OffsetT, ValueT> &input)
{
  thrust::device_vector<KeyT> keys_output(input.get_num_items());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  thrust::device_vector<ValueT> values_output(input.get_num_items());
  ValueT *d_values_output = thrust::raw_pointer_cast(values_output.data());

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortPairs(nullptr,
                                                   temp_storage_bytes,
                                                   input.get_d_keys(),
                                                   d_keys_output,
                                                   input.get_d_values(),
                                                   d_values_output,
                                                   input.get_num_items(),
                                                   input.get_num_segments(),
                                                   input.get_d_offsets(),
                                                   input.get_d_offsets() + 1,
                                                   0,
                                                   true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});

    CubDebugExit(cub::DeviceSegmentedSort::SortPairs(d_tmp_storage,
                                                     temp_storage_bytes,
                                                     input.get_d_keys(),
                                                     d_keys_output,
                                                     input.get_d_values(),
                                                     d_values_output,
                                                     input.get_num_items(),
                                                     input.get_num_segments(),
                                                     input.get_d_offsets(),
                                                     input.get_d_offsets() + 1,
                                                     0,
                                                     true));

    AssertTrue(input.check_output(d_keys_output, d_values_output));

    input.shuffle();
  }
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void InputTestPairsBuffer(Input<KeyT, OffsetT, ValueT> &input)
{
  thrust::device_vector<KeyT> keys_output(input.get_num_items());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  thrust::device_vector<ValueT> values_output(input.get_num_items());
  ValueT *d_values_output = thrust::raw_pointer_cast(values_output.data());

  cub::DoubleBuffer<KeyT> empty_keys(nullptr, nullptr);
  cub::DoubleBuffer<ValueT> empty_values(nullptr, nullptr);

  std::size_t temp_storage_bytes{};
  CubDebugExit(
    cub::DeviceSegmentedSort::SortPairs(nullptr,
                                        temp_storage_bytes,
                                        empty_keys,
                                        empty_values,
                                        input.get_num_items(),
                                        input.get_num_segments(),
                                        input.get_d_offsets(),
                                        input.get_d_offsets() + 1,
                                        0,
                                        true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});
    thrust::fill(values_output.begin(), values_output.end(), ValueT{});

    cub::DoubleBuffer<KeyT> keys(input.get_d_keys(), d_keys_output);
    cub::DoubleBuffer<ValueT> values(input.get_d_values(), d_values_output);

    CubDebugExit(
      cub::DeviceSegmentedSort::SortPairs(d_tmp_storage,
                                          temp_storage_bytes,
                                          keys,
                                          values,
                                          input.get_num_items(),
                                          input.get_num_segments(),
                                          input.get_d_offsets(),
                                          input.get_d_offsets() + 1,
                                          0,
                                          true));

    AssertTrue(input.check_output(keys.Current(), values.Current()));

    input.shuffle();
  }
}

template <typename KeyT,
          typename OffsetT>
void InputTestDescending(Input<KeyT, OffsetT> &input)
{
  thrust::device_vector<KeyT> keys_output(input.get_num_items());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  std::size_t temp_storage_bytes{};
  CubDebugExit(
    cub::DeviceSegmentedSort::SortKeysDescending(nullptr,
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

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});

    CubDebugExit(
      cub::DeviceSegmentedSort::SortKeysDescending(d_tmp_storage,
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
void InputTestDescendingBuffer(Input<KeyT, OffsetT> &input)
{
  thrust::device_vector<KeyT> keys_output(input.get_num_items());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  cub::DoubleBuffer<KeyT> empty_keys(nullptr, nullptr);

  std::size_t temp_storage_bytes{};
  CubDebugExit(
    cub::DeviceSegmentedSort::SortKeysDescending(nullptr,
                                                 temp_storage_bytes,
                                                 empty_keys,
                                                 input.get_num_items(),
                                                 input.get_num_segments(),
                                                 input.get_d_offsets(),
                                                 input.get_d_offsets() + 1,
                                                 0,
                                                 true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});

    cub::DoubleBuffer<KeyT> keys(input.get_d_keys(), d_keys_output);

    CubDebugExit(
      cub::DeviceSegmentedSort::SortKeysDescending(d_tmp_storage,
                                                   temp_storage_bytes,
                                                   keys,
                                                   input.get_num_items(),
                                                   input.get_num_segments(),
                                                   input.get_d_offsets(),
                                                   input.get_d_offsets() + 1,
                                                   0,
                                                   true));

    AssertTrue(input.check_output(keys.Current()));

    input.shuffle();
  }
}


template <typename T,
          typename OffsetT>
bool compare_two_outputs(const thrust::host_vector<OffsetT> &offsets,
                         const thrust::host_vector<T> &lhs,
                         const thrust::host_vector<T> &rhs)
{
  const std::size_t num_segments = offsets.size() - 1;

  for (std::size_t segment_id = 0; segment_id < num_segments; segment_id++)
  {
    auto lhs_begin = lhs.cbegin() + offsets[segment_id];
    auto lhs_end = lhs.cbegin() + offsets[segment_id + 1];
    auto rhs_begin = rhs.cbegin() + offsets[segment_id];

    auto err = thrust::mismatch(lhs_begin, lhs_end, rhs_begin);

    if (err.first != lhs_end)
    {
      const auto idx = thrust::distance(lhs_begin, err.first);
      const auto segment_size = std::distance(lhs_begin, lhs_end);

      std::cerr << "Mismatch in segment " << segment_id
                << " at position " << idx << " / " << segment_size
                << ": "
                << static_cast<std::int64_t>(lhs_begin[idx]) << " vs "
                << static_cast<std::int64_t>(rhs_begin[idx]) << std::endl;

      return false;
    }
  }

  return true;
}

template <typename KeyT,
          typename OffsetT>
void InputTestDescendingRandom(Input<KeyT, OffsetT> &input)
{
  thrust::host_vector<KeyT> h_keys_output(input.get_num_items());
  thrust::device_vector<KeyT> keys_output(input.get_num_items());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  std::size_t temp_storage_bytes{};
  CubDebugExit(
    cub::DeviceSegmentedSort::SortKeysDescending(nullptr,
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

  const thrust::host_vector<OffsetT> &h_offsets = input.get_h_offsets();
  thrust::host_vector<KeyT> h_keys(input.get_num_items());

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});

    for (std::size_t i = 0; i < input.get_num_items(); i++)
    {
      h_keys[i] = RandomValue(std::numeric_limits<KeyT>::max());
    }
    thrust::copy(h_keys.begin(), h_keys.end(), input.get_d_keys_vec().begin());

    CubDebugExit(
      cub::DeviceSegmentedSort::SortKeysDescending(d_tmp_storage,
                                                   temp_storage_bytes,
                                                   input.get_d_keys(),
                                                   d_keys_output,
                                                   input.get_num_items(),
                                                   input.get_num_segments(),
                                                   input.get_d_offsets(),
                                                   input.get_d_offsets() + 1,
                                                   0,
                                                   true));

    for (std::size_t segment_i = 0;
         segment_i < input.get_num_segments();
         segment_i++)
    {
      const OffsetT segment_begin = h_offsets[segment_i];
      const OffsetT segment_end = h_offsets[segment_i + 1];

      thrust::sort(h_keys.begin() + segment_begin,
                   h_keys.begin() + segment_end,
                   thrust::greater<KeyT>{});
    }

    h_keys_output = keys_output;

    AssertTrue(compare_two_outputs(h_offsets, h_keys, h_keys_output));

    input.shuffle();
  }
}

template <typename KeyT,
          typename OffsetT>
void InputTestDescendingRandomBuffer(Input<KeyT, OffsetT> &input)
{
  thrust::host_vector<KeyT> h_keys_output(input.get_num_items());
  thrust::device_vector<KeyT> keys_output(input.get_num_items());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  cub::DoubleBuffer<KeyT> empty_keys(nullptr, nullptr);

  std::size_t temp_storage_bytes{};
  CubDebugExit(
    cub::DeviceSegmentedSort::SortKeysDescending(nullptr,
                                                 temp_storage_bytes,
                                                 empty_keys,
                                                 input.get_num_items(),
                                                 input.get_num_segments(),
                                                 input.get_d_offsets(),
                                                 input.get_d_offsets() + 1,
                                                 0,
                                                 true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  const thrust::host_vector<OffsetT> &h_offsets = input.get_h_offsets();
  thrust::host_vector<KeyT> h_keys(input.get_num_items());

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});

    cub::DoubleBuffer<KeyT> keys(input.get_d_keys(), d_keys_output);

    for (std::size_t i = 0; i < input.get_num_items(); i++)
    {
      h_keys[i] = RandomValue(std::numeric_limits<KeyT>::max());
    }
    thrust::copy(h_keys.begin(), h_keys.end(), input.get_d_keys_vec().begin());

    CubDebugExit(
      cub::DeviceSegmentedSort::SortKeysDescending(d_tmp_storage,
                                                   temp_storage_bytes,
                                                   keys,
                                                   input.get_num_items(),
                                                   input.get_num_segments(),
                                                   input.get_d_offsets(),
                                                   input.get_d_offsets() + 1,
                                                   0,
                                                   true));

    for (std::size_t segment_i = 0;
         segment_i < input.get_num_segments();
         segment_i++)
    {
      const OffsetT segment_begin = h_offsets[segment_i];
      const OffsetT segment_end = h_offsets[segment_i + 1];

      thrust::sort(h_keys.begin() + segment_begin,
                   h_keys.begin() + segment_end,
                   thrust::greater<KeyT>{});
    }

    if (keys.selector)
    {
      h_keys_output = keys_output;
    }
    else
    {
      h_keys_output = input.get_d_keys_vec();
    }

    AssertTrue(compare_two_outputs(h_offsets, h_keys, h_keys_output));

    input.shuffle();
  }
}

template <typename KeyT,
          typename OffsetT>
void InputTestRandomBuffer(Input<KeyT, OffsetT> &input)
{
  thrust::host_vector<KeyT> h_keys_output(input.get_num_items());
  thrust::device_vector<KeyT> keys_output(input.get_num_items());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  cub::DoubleBuffer<KeyT> empty_keys(nullptr, nullptr);

  std::size_t temp_storage_bytes{};
  CubDebugExit(
    cub::DeviceSegmentedSort::SortKeys(nullptr,
                                       temp_storage_bytes,
                                       empty_keys,
                                       input.get_num_items(),
                                       input.get_num_segments(),
                                       input.get_d_offsets(),
                                       input.get_d_offsets() + 1,
                                       0,
                                       true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  const thrust::host_vector<OffsetT> &h_offsets = input.get_h_offsets();
  thrust::host_vector<KeyT> h_keys(input.get_num_items());

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});

    cub::DoubleBuffer<KeyT> keys(input.get_d_keys(), d_keys_output);

    for (std::size_t i = 0; i < input.get_num_items(); i++)
    {
      h_keys[i] = RandomValue(std::numeric_limits<KeyT>::max());
    }
    thrust::copy(h_keys.begin(), h_keys.end(), input.get_d_keys_vec().begin());

    CubDebugExit(
      cub::DeviceSegmentedSort::SortKeys(d_tmp_storage,
                                         temp_storage_bytes,
                                         keys,
                                         input.get_num_items(),
                                         input.get_num_segments(),
                                         input.get_d_offsets(),
                                         input.get_d_offsets() + 1,
                                         0,
                                         true));

    for (std::size_t segment_i = 0;
         segment_i < input.get_num_segments();
         segment_i++)
    {
      const OffsetT segment_begin = h_offsets[segment_i];
      const OffsetT segment_end = h_offsets[segment_i + 1];

      thrust::sort(h_keys.begin() + segment_begin,
                   h_keys.begin() + segment_end);
    }

    if (keys.selector)
    {
      h_keys_output = keys_output;
    }
    else
    {
      h_keys_output = input.get_d_keys_vec();
    }

    AssertTrue(compare_two_outputs(h_offsets, h_keys, h_keys_output));

    input.shuffle();
  }
}

template <typename KeyT,
          typename OffsetT>
void InputTestRandom(Input<KeyT, OffsetT> &input)
{
  thrust::host_vector<KeyT> h_keys_output(input.get_num_items());
  thrust::device_vector<KeyT> keys_output(input.get_num_items());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  std::size_t temp_storage_bytes{};
  CubDebugExit(
    cub::DeviceSegmentedSort::SortKeys(nullptr,
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

  const thrust::host_vector<OffsetT> &h_offsets = input.get_h_offsets();
  thrust::host_vector<KeyT> h_keys(input.get_num_items());

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});

    for (std::size_t i = 0; i < input.get_num_items(); i++)
    {
      h_keys[i] = RandomValue(std::numeric_limits<KeyT>::max());
    }
    thrust::copy(h_keys.begin(), h_keys.end(), input.get_d_keys_vec().begin());

    CubDebugExit(
      cub::DeviceSegmentedSort::SortKeys(d_tmp_storage,
                                         temp_storage_bytes,
                                         input.get_d_keys(),
                                         d_keys_output,
                                         input.get_num_items(),
                                         input.get_num_segments(),
                                         input.get_d_offsets(),
                                         input.get_d_offsets() + 1,
                                         0,
                                         true));

    for (std::size_t segment_i = 0; segment_i < input.get_num_segments();
         segment_i++)
    {
      const OffsetT segment_begin = h_offsets[segment_i];
      const OffsetT segment_end   = h_offsets[segment_i + 1];

      thrust::sort(h_keys.begin() + segment_begin,
                   h_keys.begin() + segment_end);
    }

    h_keys_output = keys_output;

    AssertTrue(compare_two_outputs(h_offsets, h_keys, h_keys_output));

    input.shuffle();
  }
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void InputTestPairsRandom(Input<KeyT, OffsetT, ValueT> &input)
{
  thrust::host_vector<KeyT> h_keys_output(input.get_num_items());
  thrust::device_vector<KeyT> keys_output(input.get_num_items());

  thrust::host_vector<ValueT> h_values_output(input.get_num_items());
  thrust::device_vector<ValueT> values_output(input.get_num_items());

  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());
  ValueT *d_values_output = thrust::raw_pointer_cast(values_output.data());

  std::size_t temp_storage_bytes{};
  CubDebugExit(cub::DeviceSegmentedSort::SortPairs(nullptr,
                                                   temp_storage_bytes,
                                                   input.get_d_keys(),
                                                   d_keys_output,
                                                   input.get_d_values(),
                                                   d_values_output,
                                                   input.get_num_items(),
                                                   input.get_num_segments(),
                                                   input.get_d_offsets(),
                                                   input.get_d_offsets() + 1,
                                                   0,
                                                   true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  const thrust::host_vector<OffsetT> &h_offsets = input.get_h_offsets();
  thrust::host_vector<KeyT> h_keys(input.get_num_items());
  thrust::host_vector<ValueT> h_values(input.get_num_items());

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});

    for (std::size_t i = 0; i < input.get_num_items(); i++)
    {
      h_keys[i] = RandomValue(std::numeric_limits<KeyT>::max());
      h_values[i] = RandomValue(std::numeric_limits<ValueT>::max());
    }
    thrust::copy(h_keys.begin(), h_keys.end(), input.get_d_keys_vec().begin());
    thrust::copy(h_values.begin(), h_values.end(), input.get_d_values_vec().begin());

    CubDebugExit(cub::DeviceSegmentedSort::SortPairs(d_tmp_storage,
                                                     temp_storage_bytes,
                                                     input.get_d_keys(),
                                                     d_keys_output,
                                                     input.get_d_values(),
                                                     d_values_output,
                                                     input.get_num_items(),
                                                     input.get_num_segments(),
                                                     input.get_d_offsets(),
                                                     input.get_d_offsets() + 1,
                                                     0,
                                                     true));

    for (std::size_t segment_i = 0;
         segment_i < input.get_num_segments();
         segment_i++)
    {
      const OffsetT segment_begin = h_offsets[segment_i];
      const OffsetT segment_end   = h_offsets[segment_i + 1];

      thrust::sort_by_key(h_keys.begin() + segment_begin,
                          h_keys.begin() + segment_end,
                          h_values.begin() + segment_begin);
    }

    h_keys_output = keys_output;
    h_values_output = values_output;

    AssertTrue(compare_two_outputs(h_offsets, h_keys, h_keys_output));
    AssertTrue(compare_two_outputs(h_offsets, h_values, h_values_output));

    input.shuffle();
  }
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void InputTestPairsDescendingRandom(Input<KeyT, OffsetT, ValueT> &input)
{
  thrust::host_vector<KeyT> h_keys_output(input.get_num_items());
  thrust::device_vector<KeyT> keys_output(input.get_num_items());

  thrust::host_vector<ValueT> h_values_output(input.get_num_items());
  thrust::device_vector<ValueT> values_output(input.get_num_items());

  KeyT *d_keys_output   = thrust::raw_pointer_cast(keys_output.data());
  ValueT *d_values_output = thrust::raw_pointer_cast(values_output.data());

  std::size_t temp_storage_bytes{};
  CubDebugExit(
    cub::DeviceSegmentedSort::SortPairsDescending(nullptr,
                                                  temp_storage_bytes,
                                                  input.get_d_keys(),
                                                  d_keys_output,
                                                  input.get_d_values(),
                                                  d_values_output,
                                                  input.get_num_items(),
                                                  input.get_num_segments(),
                                                  input.get_d_offsets(),
                                                  input.get_d_offsets() + 1,
                                                  0,
                                                  true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  const thrust::host_vector<OffsetT> &h_offsets = input.get_h_offsets();
  thrust::host_vector<KeyT> h_keys(input.get_num_items());
  thrust::host_vector<ValueT> h_values(input.get_num_items());

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});
    thrust::fill(values_output.begin(), values_output.end(), ValueT{});

    for (std::size_t i = 0; i < input.get_num_items(); i++)
    {
      h_keys[i]   = RandomValue(std::numeric_limits<KeyT>::max());
      h_values[i] = RandomValue(std::numeric_limits<ValueT>::max());
    }
    thrust::copy(h_keys.begin(), h_keys.end(), input.get_d_keys_vec().begin());
    thrust::copy(h_values.begin(),
                 h_values.end(),
                 input.get_d_values_vec().begin());

    CubDebugExit(
      cub::DeviceSegmentedSort::SortPairsDescending(d_tmp_storage,
                                                    temp_storage_bytes,
                                                    input.get_d_keys(),
                                                    d_keys_output,
                                                    input.get_d_values(),
                                                    d_values_output,
                                                    input.get_num_items(),
                                                    input.get_num_segments(),
                                                    input.get_d_offsets(),
                                                    input.get_d_offsets() + 1,
                                                    0,
                                                    true));

    for (std::size_t segment_i = 0; segment_i < input.get_num_segments();
         segment_i++)
    {
      const OffsetT segment_begin = h_offsets[segment_i];
      const OffsetT segment_end   = h_offsets[segment_i + 1];

      thrust::sort_by_key(h_keys.begin() + segment_begin,
                          h_keys.begin() + segment_end,
                          h_values.begin() + segment_begin,
                          thrust::greater<KeyT>{});
    }

    thrust::copy(keys_output.begin(), keys_output.end(), h_keys_output.begin());
    thrust::copy(values_output.begin(), values_output.end(), h_values_output.begin());

    AssertTrue(compare_two_outputs(h_offsets, h_keys, h_keys_output));
    AssertTrue(compare_two_outputs(h_offsets, h_values, h_values_output));

    input.shuffle();
  }
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void InputTestPairsDescendingRandomBuffer(Input<KeyT, OffsetT, ValueT> &input)
{
  thrust::host_vector<KeyT> h_keys_output(input.get_num_items());
  thrust::device_vector<KeyT> keys_output(input.get_num_items());

  thrust::host_vector<ValueT> h_values_output(input.get_num_items());
  thrust::device_vector<ValueT> values_output(input.get_num_items());

  KeyT *d_keys_output   = thrust::raw_pointer_cast(keys_output.data());
  ValueT *d_values_output = thrust::raw_pointer_cast(values_output.data());

  cub::DoubleBuffer<KeyT> empty_keys(nullptr, nullptr);
  cub::DoubleBuffer<ValueT> empty_values(nullptr, nullptr);

  std::size_t temp_storage_bytes{};
  CubDebugExit(
    cub::DeviceSegmentedSort::SortPairsDescending(nullptr,
                                                  temp_storage_bytes,
                                                  empty_keys,
                                                  empty_values,
                                                  input.get_num_items(),
                                                  input.get_num_segments(),
                                                  input.get_d_offsets(),
                                                  input.get_d_offsets() + 1,
                                                  0,
                                                  true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  const thrust::host_vector<OffsetT> &h_offsets = input.get_h_offsets();
  thrust::host_vector<KeyT> h_keys(input.get_num_items());
  thrust::host_vector<ValueT> h_values(input.get_num_items());

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});
    thrust::fill(values_output.begin(), values_output.end(), ValueT{});

    cub::DoubleBuffer<KeyT> keys(input.get_d_keys(), d_keys_output);
    cub::DoubleBuffer<ValueT> values(d_values_output, input.get_d_values());
    values.selector = 1;

    for (std::size_t i = 0; i < input.get_num_items(); i++)
    {
      h_keys[i]   = RandomValue(std::numeric_limits<KeyT>::max());
      h_values[i] = RandomValue(std::numeric_limits<ValueT>::max());
    }
    thrust::copy(h_keys.begin(), h_keys.end(), input.get_d_keys_vec().begin());
    thrust::copy(h_values.begin(),
                 h_values.end(),
                 input.get_d_values_vec().begin());

    CubDebugExit(
      cub::DeviceSegmentedSort::SortPairsDescending(d_tmp_storage,
                                                    temp_storage_bytes,
                                                    keys,
                                                    values,
                                                    input.get_num_items(),
                                                    input.get_num_segments(),
                                                    input.get_d_offsets(),
                                                    input.get_d_offsets() + 1,
                                                    0,
                                                    true));

    for (std::size_t segment_i = 0;
         segment_i < input.get_num_segments();
         segment_i++)
    {
      const OffsetT segment_begin = h_offsets[segment_i];
      const OffsetT segment_end   = h_offsets[segment_i + 1];

      thrust::sort_by_key(h_keys.begin() + segment_begin,
                          h_keys.begin() + segment_end,
                          h_values.begin() + segment_begin,
                          thrust::greater<KeyT>{});
    }

    if (keys.selector)
    {
      h_keys_output = keys_output;
    }
    else
    {
      h_keys_output = input.get_d_keys_vec();
    }

    if (values.selector)
    {
      h_values_output = input.get_d_values_vec();
    }
    else
    {
      h_values_output = values_output;
    }

    AssertTrue(compare_two_outputs(h_offsets, h_keys, h_keys_output));
    AssertTrue(compare_two_outputs(h_offsets, h_values, h_values_output));

    input.shuffle();
  }
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void InputTestPairsRandomBuffer(Input<KeyT, OffsetT, ValueT> &input)
{
  thrust::host_vector<KeyT> h_keys_output(input.get_num_items());
  thrust::device_vector<KeyT> keys_output(input.get_num_items());

  thrust::host_vector<ValueT> h_values_output(input.get_num_items());
  thrust::device_vector<ValueT> values_output(input.get_num_items());

  KeyT *d_keys_output   = thrust::raw_pointer_cast(keys_output.data());
  ValueT *d_values_output = thrust::raw_pointer_cast(values_output.data());

  cub::DoubleBuffer<KeyT> empty_keys(nullptr, nullptr);
  cub::DoubleBuffer<ValueT> empty_values(nullptr, nullptr);

  std::size_t temp_storage_bytes{};
  CubDebugExit(
    cub::DeviceSegmentedSort::SortPairs(nullptr,
                                        temp_storage_bytes,
                                        empty_keys,
                                        empty_values,
                                        input.get_num_items(),
                                        input.get_num_segments(),
                                        input.get_d_offsets(),
                                        input.get_d_offsets() + 1,
                                        0,
                                        true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  const thrust::host_vector<OffsetT> &h_offsets = input.get_h_offsets();
  thrust::host_vector<KeyT> h_keys(input.get_num_items());
  thrust::host_vector<ValueT> h_values(input.get_num_items());

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});
    thrust::fill(values_output.begin(), values_output.end(), ValueT{});

    cub::DoubleBuffer<KeyT> keys(input.get_d_keys(), d_keys_output);
    cub::DoubleBuffer<ValueT> values(d_values_output, input.get_d_values());
    values.selector = 1;

    for (std::size_t i = 0; i < input.get_num_items(); i++)
    {
      h_keys[i]   = RandomValue(std::numeric_limits<KeyT>::max());
      h_values[i] = RandomValue(std::numeric_limits<ValueT>::max());
    }
    thrust::copy(h_keys.begin(), h_keys.end(), input.get_d_keys_vec().begin());
    thrust::copy(h_values.begin(),
                 h_values.end(),
                 input.get_d_values_vec().begin());

    CubDebugExit(
      cub::DeviceSegmentedSort::SortPairs(d_tmp_storage,
                                          temp_storage_bytes,
                                          keys,
                                          values,
                                          input.get_num_items(),
                                          input.get_num_segments(),
                                          input.get_d_offsets(),
                                          input.get_d_offsets() + 1,
                                          0,
                                          true));

    for (std::size_t segment_i = 0;
         segment_i < input.get_num_segments();
         segment_i++)
    {
      const OffsetT segment_begin = h_offsets[segment_i];
      const OffsetT segment_end   = h_offsets[segment_i + 1];

      thrust::sort_by_key(h_keys.begin() + segment_begin,
                          h_keys.begin() + segment_end,
                          h_values.begin() + segment_begin);
    }

    if (keys.selector)
    {
      h_keys_output = keys_output;
    }
    else
    {
      h_keys_output = input.get_d_keys_vec();
    }

    if (values.selector)
    {
      h_values_output = input.get_d_values_vec();
    }
    else
    {
      h_values_output = values_output;
    }

    AssertTrue(compare_two_outputs(h_offsets, h_keys, h_keys_output));
    AssertTrue(compare_two_outputs(h_offsets, h_values, h_values_output));

    input.shuffle();
  }
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void InputTestDescendingPairs(Input<KeyT, OffsetT, ValueT> &input)
{
  thrust::device_vector<KeyT> keys_output(input.get_num_items());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  thrust::device_vector<ValueT> values_output(input.get_num_items());
  ValueT *d_values_output = thrust::raw_pointer_cast(values_output.data());

  std::size_t temp_storage_bytes{};
  CubDebugExit(
    cub::DeviceSegmentedSort::SortPairsDescending(nullptr,
                                                  temp_storage_bytes,
                                                  input.get_d_keys(),
                                                  d_keys_output,
                                                  input.get_d_values(),
                                                  d_values_output,
                                                  input.get_num_items(),
                                                  input.get_num_segments(),
                                                  input.get_d_offsets(),
                                                  input.get_d_offsets() + 1,
                                                  0,
                                                  true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});

    CubDebugExit(
      cub::DeviceSegmentedSort::SortPairsDescending(d_tmp_storage,
                                                    temp_storage_bytes,
                                                    input.get_d_keys(),
                                                    d_keys_output,
                                                    input.get_d_values(),
                                                    d_values_output,
                                                    input.get_num_items(),
                                                    input.get_num_segments(),
                                                    input.get_d_offsets(),
                                                    input.get_d_offsets() + 1,
                                                    0,
                                                    true));

    AssertTrue(input.check_output(d_keys_output, d_values_output));

    input.shuffle();
  }
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void InputTestDescendingPairsBuffer(Input<KeyT, OffsetT, ValueT> &input)
{
  thrust::device_vector<KeyT> keys_output(input.get_num_items());
  KeyT *d_keys_output = thrust::raw_pointer_cast(keys_output.data());

  thrust::device_vector<ValueT> values_output(input.get_num_items());
  ValueT *d_values_output = thrust::raw_pointer_cast(values_output.data());

  cub::DoubleBuffer<KeyT> empty_keys(nullptr, nullptr);
  cub::DoubleBuffer<ValueT> empty_values(nullptr, nullptr);

  std::size_t temp_storage_bytes{};
  CubDebugExit(
    cub::DeviceSegmentedSort::SortPairsDescending(nullptr,
                                                  temp_storage_bytes,
                                                  empty_keys,
                                                  empty_values,
                                                  input.get_num_items(),
                                                  input.get_num_segments(),
                                                  input.get_d_offsets(),
                                                  input.get_d_offsets() + 1,
                                                  0,
                                                  true));

  thrust::device_vector<std::uint8_t> tmp_storage(temp_storage_bytes);
  std::uint8_t *d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++)
  {
    thrust::fill(keys_output.begin(), keys_output.end(), KeyT{});
    thrust::fill(values_output.begin(), values_output.end(), KeyT{});

    cub::DoubleBuffer<KeyT> keys(input.get_d_keys(), d_keys_output);
    cub::DoubleBuffer<ValueT> values(input.get_d_values(), d_values_output);

    CubDebugExit(
      cub::DeviceSegmentedSort::SortPairsDescending(d_tmp_storage,
                                                    temp_storage_bytes,
                                                    keys,
                                                    values,
                                                    input.get_num_items(),
                                                    input.get_num_segments(),
                                                    input.get_d_offsets(),
                                                    input.get_d_offsets() + 1,
                                                    0,
                                                    true));

    AssertTrue(input.check_output(keys.Current(), values.Current()));

    input.shuffle();
  }
}

template <typename KeyT,
          typename OffsetT>
void EdgePatternsTest(bool descending)
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
                                      .gen(descending);

  if (descending)
  {
    InputTestDescending(edge_cases);
    InputTestDescendingBuffer(edge_cases);
  }
  else
  {
    InputTest(edge_cases);
    InputTestBuffer(edge_cases);
  }
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void EdgePatternsTestPairs(bool descending)
{
  Input<KeyT, OffsetT, ValueT> edge_cases = InputDescription<KeyT, OffsetT>()
                                      .add({420, 0})
                                      .add({420, 1})
                                      .add({420, 2})
                                      .add({420, 8})
                                      .add({420, 9})
                                      .add({420, 10})
                                      .add({420, 17})
                                      .add({42, 18})
                                      .add({42, 19})
                                      .add({42, 26})
                                      .add({42, 27})
                                      .add({42, 28})
                                      .add({42, 35})
                                      .add({42, 36})
                                      .add({42, 37})
                                      .add({42, 286})
                                      .add({42, 287})
                                      .add({42, 288})
                                      .add({42, 5887})
                                      .add({42, 5888})
                                      .add({42, 5889})
                                      .add({2, 23552})
                                      .template gen<ValueT>(descending);

  if (descending)
  {
    InputTestDescendingPairs<KeyT, ValueT, OffsetT>(edge_cases);
    InputTestDescendingPairsBuffer<KeyT, ValueT, OffsetT>(edge_cases);
  }
  else
  {
    InputTestPairs<KeyT, ValueT, OffsetT>(edge_cases);
    InputTestPairsBuffer<KeyT, ValueT, OffsetT>(edge_cases);
  }
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
Input<KeyT, OffsetT, ValueT> GenRandomInput(OffsetT max_items,
                                            OffsetT max_segments,
                                            bool descending)
{
  std::size_t items_generated {};
  const std::size_t segments_num = RandomValue(max_segments) + 1;

  thrust::host_vector<OffsetT> segment_sizes;
  segment_sizes.reserve(segments_num);

  for (std::size_t segment_id = 0; segment_id < segments_num; segment_id++)
  {
    const OffsetT segment_size_raw = RandomValue(max_items / 100);
    const OffsetT segment_size = segment_size_raw > OffsetT{0} ? segment_size_raw
                                                               : OffsetT{0};

    if (segment_size + items_generated > max_items)
    {
      break;
    }

    items_generated += segment_size;
    segment_sizes.push_back(segment_size);
  }

  return Input<KeyT, OffsetT, ValueT>{descending, segment_sizes};
}

template <typename KeyT,
          typename OffsetT>
void RandomTest(bool descending)
{
  const OffsetT max_items = 1000000;
  const OffsetT max_segments = 42000;

  Input<KeyT, OffsetT> random_input =
    GenRandomInput<KeyT, cub::NullType, OffsetT>(max_items,
                                                 max_segments,
                                                 descending);

  if (descending)
  {
    InputTestDescendingRandom(random_input);
    InputTestDescendingRandomBuffer(random_input);
  }
  else
  {
    InputTestRandom(random_input);
    InputTestRandomBuffer(random_input);
  }
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void RandomPairsTest(bool descending)
{
  const OffsetT max_items = 1000000;
  const OffsetT max_segments = 42000;

  Input<KeyT, OffsetT, ValueT> edge_cases =
    GenRandomInput<KeyT, ValueT, OffsetT>(max_items, max_segments, descending);

  if (descending)
  {
    InputTestPairsDescendingRandom<KeyT, ValueT, OffsetT>(edge_cases);
    InputTestPairsDescendingRandomBuffer<KeyT, ValueT, OffsetT>(edge_cases);
  }
  else
  {
    InputTestPairsRandom<KeyT, ValueT, OffsetT>(edge_cases);
    InputTestPairsRandomBuffer<KeyT, ValueT, OffsetT>(edge_cases);
  }
}

template <typename KeyT,
          typename OffsetT>
void TestKeys()
{
  IndependentTest<KeyT, OffsetT>();
  DependentTest<KeyT, OffsetT>();

  const bool basic = false;
  EdgePatternsTest<KeyT, OffsetT>(basic);
  RandomTest<KeyT, OffsetT>(basic);

  const bool descending = true;
  EdgePatternsTest<KeyT, OffsetT>(descending);
  RandomTest<KeyT, OffsetT>(descending);
}

template <typename KeyT,
          typename ValueT,
          typename OffsetT>
void TestPairs()
{
  IndependentTestPairs<KeyT, ValueT, OffsetT>();
  DependentTestPairs<KeyT, ValueT, OffsetT>();

  const bool basic = false;
  EdgePatternsTestPairs<KeyT, ValueT, OffsetT>(basic);
  RandomPairsTest<KeyT, ValueT, OffsetT>(basic);

  const bool descending = true;
  EdgePatternsTestPairs<KeyT, ValueT, OffsetT>(descending);
  RandomPairsTest<KeyT, ValueT, OffsetT>(descending);
}

template <typename T,
          typename OffsetT>
void TestKeysAndPairs()
{
  TestKeys<T, OffsetT>();
  TestPairs<T, T, OffsetT>();
}

int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  TestKeysAndPairs<std::uint8_t,  std::uint32_t>();
  TestKeysAndPairs<std::uint16_t, std::uint32_t>();
  TestKeysAndPairs<std::uint32_t, std::uint32_t>();
  TestKeysAndPairs<std::uint64_t, std::uint32_t>();
  TestKeysAndPairs<std::uint64_t, std::uint64_t>();
  TestPairs<std::uint8_t, std::uint64_t, std::uint32_t>();
  TestPairs<std::int64_t, std::uint64_t, std::uint32_t>();

  return 0;
}
