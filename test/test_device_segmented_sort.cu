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

class InputDescription
{
  std::vector<SizeGroupDescription> groups;

public:
  InputDescription& add(const SizeGroupDescription &group)
  {
    groups.push_back(group);
    return *this;
  }
};

template <typename KeyT,
          typename OffsetT>
void TestZeroSegments()
{
  const OffsetT *d_offsets = nullptr;
  const KeyT *keys_input = nullptr;
  KeyT *keys_output      = nullptr;

  std::size_t temp_storage_bytes{};
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

template <typename KeyT,
          typename OffsetT>
void TestSingleItemSegments(OffsetT segments)
{
  thrust::device_vector<OffsetT> offsets(segments + 1);
  thrust::sequence(offsets.begin(), offsets.end(), OffsetT{}, OffsetT{1});

  const OffsetT *d_offsets = thrust::raw_pointer_cast(offsets.data());

  thrust::device_vector<OffsetT> keys_input(segments, KeyT{42});
  thrust::device_vector<OffsetT> keys_output(segments, KeyT{24});

  const KeyT *d_keys_input = thrust::raw_pointer_cast(keys_input.data());
  KeyT *d_keys_output      = thrust::raw_pointer_cast(keys_output.data());

  const OffsetT num_items = segments; // There's only one item in each segment

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
}

template <typename KeyT,
          typename OffsetT>
void DependentTest()
{
  DependentTest<KeyT, OffsetT>(42);
}

template <typename KeyT,
          typename OffsetT>
void Test()
{
  IndependentTest<KeyT, OffsetT>();
  DependentTest<KeyT, OffsetT>();
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

  Test<std::uint32_t, std::uint32_t>();

  return 0;
}
