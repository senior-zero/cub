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

#include <limits>
#include <memory>

#include "cub/util_allocator.cuh"
#include "cub/thread/thread_operators.cuh"
#include "cub/device/device_adjacent_difference.cuh"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>

#include "test_util.h"


using namespace cub;


/**
 * \brief Generates integer sequence \f$S_n=i(i-1)/2\f$.
 *
 * The adjacent difference of this sequence produce consecutive numbers:
 * \f[
 *   p = \frac{i(i - 1)}{2} \\
 *   n = \frac{(i + 1) i}{2} \\
 *   n - p = i \\
 *   \frac{(i + 1) i}{2} - \frac{i (i - 1)}{2} = i \\
 *   (i + 1) i - i (i - 1) = 2 i \\
 *   (i + 1) - (i - 1) = 2 \\
 *   2 = 2
 * \f]
 */
template <typename DestT>
struct TestSequenceGenerator
{
  template <typename SourceT>
  __device__ __host__ DestT operator()(SourceT index) const
  {
    return static_cast<DestT>(index * (index - 1) / SourceT(2));
  }
};


template <typename OutputT>
struct CustomDifference
{
  template <typename InputT>
  __device__ OutputT operator()(const InputT &lhs, const InputT &rhs)
  {
    return static_cast<OutputT>(lhs - rhs);
  }
};

template <bool ReadLeft,
          typename IteratorT,
          typename DifferenceOpT>
void AdjacentDifference(void *temp_storage,
                        std::size_t &temp_storage_bytes,
                        IteratorT it,
                        DifferenceOpT difference_op,
                        std::size_t num_items)
{
  const bool is_default_op_in_use =
    std::is_same<DifferenceOpT, cub::Difference>::value;

  if (ReadLeft)
  {
    if (is_default_op_in_use)
    {
      CubDebugExit(
        cub::DeviceAdjacentDifference::SubtractLeft(temp_storage,
                                                    temp_storage_bytes,
                                                    it,
                                                    num_items,
                                                    0,
                                                    true));
    }
    else
    {
      CubDebugExit(
        cub::DeviceAdjacentDifference::SubtractLeft(temp_storage,
                                                    temp_storage_bytes,
                                                    it,
                                                    num_items,
                                                    difference_op,
                                                    0,
                                                    true));
    }
  }
  else
  {
    if (is_default_op_in_use)
    {
      CubDebugExit(
        cub::DeviceAdjacentDifference::SubtractRight(temp_storage,
                                                     temp_storage_bytes,
                                                     it,
                                                     num_items,
                                                     0,
                                                     true));
    }
    else
    {
      CubDebugExit(
        cub::DeviceAdjacentDifference::SubtractRight(temp_storage,
                                                     temp_storage_bytes,
                                                     it,
                                                     num_items,
                                                     difference_op,
                                                     0,
                                                     true));
    }
  }
}


template <bool ReadLeft,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename DifferenceOpT>
void AdjacentDifferenceCopy(void *temp_storage,
                            std::size_t &temp_storage_bytes,
                            InputIteratorT input,
                            OutputIteratorT output,
                            DifferenceOpT difference_op,
                            std::size_t num_items)
{
  const bool is_default_op_in_use =
    std::is_same<DifferenceOpT, cub::Difference>::value;

  if (ReadLeft)
  {
    if (is_default_op_in_use)
    {
      CubDebugExit(
        cub::DeviceAdjacentDifference::SubtractLeftCopy(temp_storage,
                                                        temp_storage_bytes,
                                                        input,
                                                        output,
                                                        num_items,
                                                        0,
                                                        true));
    }
    else
    {
      CubDebugExit(
        cub::DeviceAdjacentDifference::SubtractLeftCopy(temp_storage,
                                                        temp_storage_bytes,
                                                        input,
                                                        output,
                                                        num_items,
                                                        difference_op,
                                                        0,
                                                        true));
    }
  }
  else
  {
    if (is_default_op_in_use)
    {
      CubDebugExit(
        cub::DeviceAdjacentDifference::SubtractRightCopy(temp_storage,
                                                         temp_storage_bytes,
                                                         input,
                                                         output,
                                                         num_items,
                                                         0,
                                                         true));
    }
    else
    {
      CubDebugExit(
        cub::DeviceAdjacentDifference::SubtractRightCopy(temp_storage,
                                                         temp_storage_bytes,
                                                         input,
                                                         output,
                                                         num_items,
                                                         difference_op,
                                                         0,
                                                         true));
    }
  }
}

template <bool ReadLeft,
          typename IteratorT,
          typename DifferenceOpT>
void AdjacentDifference(IteratorT it,
                        DifferenceOpT difference_op,
                        std::size_t num_items)
{
  std::size_t temp_storage_bytes {};

  AdjacentDifference<ReadLeft>(nullptr,
                               temp_storage_bytes,
                               it,
                               difference_op,
                               num_items);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  AdjacentDifference<ReadLeft>(thrust::raw_pointer_cast(temp_storage.data()),
                               temp_storage_bytes,
                               it,
                               difference_op,
                               num_items);
}


template <bool ReadLeft,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename DifferenceOpT>
void AdjacentDifferenceCopy(InputIteratorT input,
                            OutputIteratorT output,
                            DifferenceOpT difference_op,
                            std::size_t num_items)
{
  std::size_t temp_storage_bytes{};

  AdjacentDifferenceCopy<ReadLeft>(nullptr,
                                   temp_storage_bytes,
                                   input,
                                   output,
                                   difference_op,
                                   num_items);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  AdjacentDifferenceCopy<ReadLeft>(thrust::raw_pointer_cast(
                                     temp_storage.data()),
                                   temp_storage_bytes,
                                   input,
                                   output,
                                   difference_op,
                                   num_items);
}

template <typename FirstIteratorT,
          typename SecondOperatorT>
bool CheckResult(FirstIteratorT first_begin,
                 FirstIteratorT first_end,
                 SecondOperatorT second_begin)
{
  auto err = thrust::mismatch(first_begin, first_end, second_begin);

  if (err.first != first_end)
  {
    return false;
  }

  return true;
}


template <typename InputT,
          typename OutputT,
          typename DifferenceOpT>
void TestCopy(std::size_t elements, DifferenceOpT difference_op)
{
  thrust::device_vector<InputT> input(elements);
  thrust::tabulate(input.begin(),
                   input.end(),
                   TestSequenceGenerator<InputT>{});

  thrust::device_vector<OutputT> output(elements, OutputT{42});

  InputT *d_input = thrust::raw_pointer_cast(input.data());
  OutputT *d_output = thrust::raw_pointer_cast(output.data());

  using CountingIteratorT =
    typename thrust::counting_iterator<OutputT,
                                       thrust::use_default,
                                       std::size_t,
                                       std::size_t>;

  constexpr bool read_left = true;
  constexpr bool read_right = false;

  AdjacentDifferenceCopy<read_left>(d_input,
                                    d_output,
                                    difference_op,
                                    elements);

  AssertTrue(CheckResult(output.begin() + 1,
                         output.end(),
                         CountingIteratorT(OutputT{0})));

  thrust::fill(output.begin(), output.end(), OutputT{42});

  AdjacentDifferenceCopy<read_right>(d_input,
                                     d_output,
                                     difference_op,
                                     elements);

  thrust::device_vector<OutputT> reference(input.size());
  thrust::sequence(reference.begin(),
                   reference.end(),
                   static_cast<OutputT>(0),
                   static_cast<OutputT>(-1));
  AssertTrue(CheckResult(output.begin(),
                         output.end() - 1,
                         reference.begin()));
}


template <typename InputT,
          typename OutputT,
          typename DifferenceOpT>
void TestIteratorCopy(std::size_t elements, DifferenceOpT difference_op)
{
  thrust::device_vector<InputT> input(elements);
  thrust::tabulate(input.begin(),
                   input.end(),
                   TestSequenceGenerator<InputT>{});

  thrust::device_vector<OutputT> output(elements, OutputT{42});

  using CountingIteratorT =
  typename thrust::counting_iterator<OutputT,
    thrust::use_default,
    std::size_t,
    std::size_t>;

  constexpr bool read_left = true;
  constexpr bool read_right = false;

  AdjacentDifferenceCopy<read_left>(input.cbegin(),
                                    output.begin(),
                                    difference_op,
                                    elements);

  AssertTrue(CheckResult(output.begin() + 1,
                         output.end(),
                         CountingIteratorT(OutputT{0})));

  thrust::fill(output.begin(), output.end(), OutputT{42});

  AdjacentDifferenceCopy<read_right>(input.cbegin(),
                                     output.begin(),
                                     difference_op,
                                     elements);

  thrust::device_vector<OutputT> reference(input.size());
  thrust::sequence(reference.begin(),
                   reference.end(),
                   static_cast<OutputT>(0),
                   static_cast<OutputT>(-1));
  AssertTrue(CheckResult(output.begin(),
                         output.end() - 1,
                         reference.begin()));
}


template <typename InputT,
          typename OutputT>
void TestCopy(std::size_t elements)
{
  TestCopy<InputT, OutputT>(elements, cub::Difference{});
  TestCopy<InputT, OutputT>(elements, CustomDifference<OutputT>{});

  TestIteratorCopy<InputT, OutputT>(elements, cub::Difference{});
  TestIteratorCopy<InputT, OutputT>(elements, CustomDifference<OutputT>{});
}


void TestCopy(std::size_t elements)
{
  TestCopy<std::uint64_t, std::int64_t >(elements);
  TestCopy<std::uint32_t, std::int32_t>(elements);
}


template <typename T,
          typename DifferenceOpT>
void Test(std::size_t elements, DifferenceOpT difference_op)
{
  thrust::device_vector<T> data(elements);
  thrust::tabulate(data.begin(),
                   data.end(),
                   TestSequenceGenerator<T>{});

  T *d_data = thrust::raw_pointer_cast(data.data());

  using CountingIteratorT =
    typename thrust::counting_iterator<T,
      thrust::use_default,
      std::size_t,
      std::size_t>;

  constexpr bool read_left = true;
  constexpr bool read_right = false;

  AdjacentDifference<read_left>(d_data,
                                difference_op,
                                elements);

  AssertTrue(CheckResult(data.begin() + 1,
                         data.end(),
                         CountingIteratorT(T{0})));


  thrust::tabulate(data.begin(),
                   data.end(),
                   TestSequenceGenerator<T>{});

  AdjacentDifference<read_right>(d_data,
                                 difference_op,
                                 elements);

  thrust::device_vector<T> reference(data.size());
  thrust::sequence(reference.begin(),
                   reference.end(),
                   static_cast<T>(0),
                   static_cast<T>(-1));
  AssertTrue(CheckResult(data.begin(),
                         data.end() - 1,
                         reference.begin()));
}


template <typename T,
          typename DifferenceOpT>
void TestIterators(std::size_t elements, DifferenceOpT difference_op)
{
  thrust::device_vector<T> data(elements);
  thrust::tabulate(data.begin(),
                   data.end(),
                   TestSequenceGenerator<T>{});

  using CountingIteratorT =
  typename thrust::counting_iterator<T,
    thrust::use_default,
    std::size_t,
    std::size_t>;

  constexpr bool read_left = true;
  constexpr bool read_right = false;

  AdjacentDifference<read_left>(data.begin(),
                                difference_op,
                                elements);

  AssertTrue(CheckResult(data.begin() + 1,
                         data.end(),
                         CountingIteratorT(T{0})));


  thrust::tabulate(data.begin(),
                   data.end(),
                   TestSequenceGenerator<T>{});

  AdjacentDifference<read_right>(data.begin(),
                                 difference_op,
                                 elements);

  thrust::device_vector<T> reference(data.size());
  thrust::sequence(reference.begin(),
                   reference.end(),
                   static_cast<T>(0),
                   static_cast<T>(-1));

  AssertTrue(CheckResult(data.begin(), data.end() - 1, reference.begin()));
}


template <typename T>
void Test(std::size_t elements)
{
  Test<T>(elements, cub::Difference{});
  Test<T>(elements, CustomDifference<T>{});

  TestIterators<T>(elements, cub::Difference{});
  TestIterators<T>(elements, CustomDifference<T>{});
}


void Test(std::size_t elements)
{
  Test<std::int32_t>(elements);
  Test<std::uint32_t>(elements);
  Test<std::uint64_t>(elements);
}


template <typename ValueT>
void TestFancyIterators(std::size_t elements)
{
  thrust::counting_iterator<ValueT> count_iter(ValueT{1});
  thrust::device_vector<ValueT> output(elements, ValueT{42});

  constexpr bool read_left  = true;
  constexpr bool read_right = false;

  AdjacentDifferenceCopy<read_left>(count_iter,
                                    output.begin(),
                                    cub::Difference{},
                                    elements);
  AssertEquals(elements,
               static_cast<std::size_t>(
                 thrust::count(output.begin(), output.end(), ValueT(1))));

  thrust::fill(output.begin(), output.end(), ValueT{});
  AdjacentDifferenceCopy<read_right>(count_iter,
                                     output.begin(),
                                     cub::Difference{},
                                     elements);
  AssertEquals(elements - 1,
               static_cast<std::size_t>(
                 thrust::count(output.begin(),
                               output.end() - 1,
                               static_cast<ValueT>(-1))));
  AssertEquals(output.back(), static_cast<ValueT>(elements));

  thrust::constant_iterator<ValueT> const_iter(ValueT{});

  AdjacentDifferenceCopy<read_left>(const_iter,
                                    output.begin(),
                                    cub::Difference{},
                                    elements);
  AssertEquals(elements,
               static_cast<std::size_t>(
                 thrust::count(output.begin(), output.end(), ValueT{})));

  thrust::fill(output.begin(), output.end(), ValueT{});
  AdjacentDifferenceCopy<read_right>(const_iter,
                                     output.begin(),
                                     cub::Difference{},
                                     elements);
  AssertEquals(elements,
               static_cast<std::size_t>(
                 thrust::count(output.begin(), output.end(), ValueT{})));

  AdjacentDifferenceCopy<read_left>(const_iter,
                                    thrust::discard_iterator{},
                                    cub::Difference{},
                                    elements);

  AdjacentDifferenceCopy<read_right>(const_iter,
                                     thrust::discard_iterator{},
                                     cub::Difference{},
                                     elements);
}


void TestFancyIterators(std::size_t elements)
{
  TestFancyIterators<std::uint64_t>(elements);
}


int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  for (int power_of_two = 2; power_of_two < 20; power_of_two += 2)
  {
    unsigned int elements = 1 << power_of_two;

    Test(elements);
    TestCopy(elements);
    TestFancyIterators(elements);
  }

  return 0;
}
