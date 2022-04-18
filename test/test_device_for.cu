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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
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

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/random.h>

#include "test_util.h"

template <typename OffsetT>
struct Incrementer
{
  int *d_counts{};

  __device__ void operator()(OffsetT i)
  {
    // Check if some `i` were served more than once
    atomicAdd(d_counts + static_cast<int>(i), 1);
  }
};

template <typename OffsetT>
class OffsetProxy
{
  OffsetT m_offset;

public:
  __host__ __device__ OffsetProxy(OffsetT offset)
      : m_offset(offset)
  {}

  __host__ __device__ operator OffsetT() const { return m_offset; }
};

struct ItemOverwriter
{
  const std::size_t *d_input;
  const std::size_t magic_value;

  __device__ void operator()(const std::size_t &i) const
  {
    if (i == magic_value)
    {
      const std::size_t *d_ptr     = &i;
      const std::size_t offset     = static_cast<std::size_t>(d_ptr - d_input);
      const_cast<std::size_t &>(i) = offset;
    }
  }
};

template <typename OffsetT>
void TestBulkDefault(OffsetT num_items)
{
  thrust::device_vector<int> counts(num_items);
  int *d_counts = thrust::raw_pointer_cast(counts.data());

  cub::DeviceFor::Bulk(num_items, Incrementer<OffsetT>{d_counts}, {}, true);

  const OffsetT num_of_once_marked_items =
    static_cast<OffsetT>(thrust::count(counts.begin(), counts.end(), 1));

  AssertEquals(num_items, num_of_once_marked_items);
}

template <typename OffsetT,
          cub::ForEachAlgorithm Algorithm,
          unsigned BlockThreads,
          unsigned ItemsPerThread>
void TestBulkTuned(OffsetT num_items)
{
  auto tuning = cub::TuneForEach<Algorithm>(
    cub::ForEachConfigurationSpace{}.Add<BlockThreads, ItemsPerThread>());

  thrust::device_vector<int> counts(num_items);
  int *d_counts = thrust::raw_pointer_cast(counts.data());

  cub::DeviceFor::Bulk(num_items,
                       Incrementer<OffsetT>{d_counts},
                       {},
                       true,
                       tuning);

  const OffsetT num_of_once_marked_items =
    static_cast<OffsetT>(thrust::count(counts.begin(), counts.end(), 1));

  AssertEquals(num_items, num_of_once_marked_items);
}

template <typename OffsetT>
void TestBulkTuned(OffsetT num_items)
{
  constexpr auto block_striped = cub::ForEachAlgorithm::BLOCK_STRIPED;

  TestBulkTuned<OffsetT, block_striped, 32, 28>(num_items);
  TestBulkTuned<OffsetT, block_striped, 128, 8>(num_items);
  TestBulkTuned<OffsetT, block_striped, 256, 2>(num_items);
  TestBulkTuned<OffsetT, block_striped, 512, 3>(num_items);
  TestBulkTuned<OffsetT, block_striped, 1024, 1>(num_items);
}

template <typename OffsetT>
void TestBulk(OffsetT num_items)
{
  TestBulkDefault<OffsetT>(num_items);
  TestBulkTuned<OffsetT>(num_items);
}

template <typename OffsetT>
void TestBulkRandom(thrust::default_random_engine &rng)
{
  const int num_iterations = 8;
  const OffsetT max_items  = 2 << 26; // Up to 512 MB
  thrust::uniform_int_distribution<OffsetT> dist(0, max_items);

  for (int iteration = 0; iteration < num_iterations; iteration++)
  {
    OffsetT num_items = dist(rng);
    TestBulk<OffsetT>(num_items);
  }
}

template <typename OffsetT>
void TestBulkEdgeCases()
{
  TestBulk<OffsetT>(0);

  for (int power_of_two = 0; power_of_two < 26; power_of_two += 2)
  {
    TestBulk<OffsetT>(static_cast<OffsetT>(2 << power_of_two) - 1);
    TestBulk<OffsetT>(static_cast<OffsetT>(2 << power_of_two));
    TestBulk<OffsetT>(static_cast<OffsetT>(2 << power_of_two) + 1);
  }
}

template <typename OffsetT>
void TestBulk(thrust::default_random_engine &rng)
{
  TestBulkRandom<OffsetT>(rng);
  TestBulkEdgeCases<OffsetT>();
}

void TestBulk(thrust::default_random_engine &rng)
{
  TestBulk<int>(rng);
  TestBulk<std::size_t>(rng);
}

template <typename OffsetT>
void TestForEachDefault(OffsetT num_items)
{
  thrust::device_vector<int> counts(num_items);
  thrust::device_vector<OffsetProxy<OffsetT>> input(num_items, OffsetT{});
  thrust::sequence(input.begin(), input.end(), OffsetT{});

  int *d_counts                 = thrust::raw_pointer_cast(counts.data());
  OffsetProxy<OffsetT> *d_input = thrust::raw_pointer_cast(input.data());

  cub::DeviceFor::ForEachN(d_input,
                           num_items,
                           Incrementer<OffsetProxy<OffsetT>>{d_counts},
                           {},
                           true);

  const OffsetT num_of_once_marked_items =
    static_cast<OffsetT>(thrust::count(counts.begin(), counts.end(), 1));

  AssertEquals(num_items, num_of_once_marked_items);
}

template <typename OffsetT,
          cub::ForEachAlgorithm Algorithm,
          cub::CacheLoadModifier LoadModifier,
          unsigned BlockThreads,
          unsigned ItemsPerThread>
void TestForEachTuned(OffsetT num_items)
{
  auto tuning = cub::TuneForEach<Algorithm, LoadModifier>(
    cub::ForEachConfigurationSpace{}.Add<BlockThreads, ItemsPerThread>());

  thrust::device_vector<int> counts(num_items);
  thrust::device_vector<OffsetProxy<OffsetT>> input(num_items, OffsetT{});
  thrust::sequence(input.begin(), input.end(), OffsetT{});

  int *d_counts                 = thrust::raw_pointer_cast(counts.data());
  OffsetProxy<OffsetT> *d_input = thrust::raw_pointer_cast(input.data());

  cub::DeviceFor::ForEachN(d_input,
                           num_items,
                           Incrementer<OffsetProxy<OffsetT>>{d_counts},
                           {},
                           true,
                           tuning);

  const OffsetT num_of_once_marked_items =
    static_cast<OffsetT>(thrust::count(counts.begin(), counts.end(), 1));

  AssertEquals(num_items, num_of_once_marked_items);
}

template <typename OffsetT, cub::CacheLoadModifier LoadModifier>
void TestForEachTuned(OffsetT num_items)
{
  constexpr auto block_striped = cub::ForEachAlgorithm::BLOCK_STRIPED;

  TestForEachTuned<OffsetT, block_striped, LoadModifier, 32, 28>(num_items);
  TestForEachTuned<OffsetT, block_striped, LoadModifier, 128, 8>(num_items);
  TestForEachTuned<OffsetT, block_striped, LoadModifier, 256, 7>(num_items);
  TestForEachTuned<OffsetT, block_striped, LoadModifier, 512, 3>(num_items);
  TestForEachTuned<OffsetT, block_striped, LoadModifier, 1024, 1>(num_items);
}

template <typename OffsetT>
void TestForEachTuned(OffsetT num_items)
{
  TestForEachTuned<OffsetT, cub::CacheLoadModifier::LOAD_DEFAULT>(num_items);
  TestForEachTuned<OffsetT, cub::CacheLoadModifier::LOAD_CA>(num_items);
  TestForEachTuned<OffsetT, cub::CacheLoadModifier::LOAD_CS>(num_items);
}

template <typename OffsetT>
void TestForEach(OffsetT num_items)
{
  // TODO Return once BLOCK_STRIPED_VECTORIZED is tested
  // TestForEachDefault<OffsetT>(num_items);
  TestForEachTuned<OffsetT>(num_items);
}

template <typename OffsetT>
void TestForEachRandom(thrust::default_random_engine &rng)
{
  const int num_iterations = 8;
  const OffsetT max_items  = 2 << 26; // Up to 512 MB
  thrust::uniform_int_distribution<OffsetT> dist(0, max_items);

  for (int iteration = 0; iteration < num_iterations; iteration++)
  {
    OffsetT num_items = dist(rng);
    TestForEach<OffsetT>(num_items);
  }
}

template <typename OffsetT>
void TestForEachEdgeCases()
{
  TestForEach<OffsetT>(0);

  for (int power_of_two = 0; power_of_two < 26; power_of_two += 4)
  {
    TestForEach<OffsetT>(static_cast<OffsetT>(2 << power_of_two) - 1);
    TestForEach<OffsetT>(static_cast<OffsetT>(2 << power_of_two));
    TestForEach<OffsetT>(static_cast<OffsetT>(2 << power_of_two) + 1);
  }
}

template <typename OffsetT>
void TestForEach(thrust::default_random_engine &rng)
{
  TestForEachRandom<OffsetT>(rng);
  TestForEachEdgeCases<OffsetT>();
}

void TestForEachIterator()
{
  const int num_items = 42 * 1024;
  thrust::device_vector<int> counts(num_items);
  int *d_counts = thrust::raw_pointer_cast(counts.data());
  auto begin    = thrust::make_counting_iterator(0);

  cub::DeviceFor::ForEachN(begin,
                           num_items,
                           Incrementer<int>{d_counts},
                           {},
                           true);

  const int num_of_once_marked_items =
    static_cast<int>(thrust::count(counts.begin(), counts.end(), 1));

  AssertEquals(num_items, num_of_once_marked_items);
}

template <cub::CacheLoadModifier LoadModifier>
void TestForEachOverwrite()
{
  const std::size_t num_items   = 42;
  const std::size_t magic_value = num_items + 1; // expected in ItemOverwriter

  thrust::device_vector<std::size_t> input(num_items, magic_value);

  const std::size_t *d_input = thrust::raw_pointer_cast(input.data());

  // Load modifier can restrict the ability to take the address of an element
  auto tuning =
    cub::TuneForEach<cub::ForEachAlgorithm::BLOCK_STRIPED, LoadModifier>(
      cub::ForEachConfigurationSpace{}.Add<256, 2>());

  cub::DeviceFor::ForEachN(d_input,
                           num_items,
                           ItemOverwriter{d_input, magic_value},
                           {},
                           true, 
                           tuning);

  if (LoadModifier == cub::CacheLoadModifier::LOAD_DEFAULT)
  {
    AssertTrue(thrust::equal(input.begin(),
                             input.end(),
                             thrust::make_counting_iterator(std::size_t{})));
  }
  else
  {
    const std::size_t num_magic_values = static_cast<std::size_t>(
      thrust::count(input.begin(), input.end(), magic_value));

    AssertEquals(num_items, num_magic_values);
  }
}

void TestForEach(thrust::default_random_engine &rng)
{
  TestForEach<int>(rng);
  TestForEach<std::size_t>(rng);
  TestForEachIterator();
  TestForEachOverwrite<cub::CacheLoadModifier::LOAD_DEFAULT>();
  TestForEachOverwrite<cub::CacheLoadModifier::LOAD_CA>();
}

int main(int argc, char **argv)
{
  // Initialize command line
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  thrust::default_random_engine rng;

  TestBulk(rng);
  TestForEach(rng);
}
