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

#pragma once

#include "../config.cuh"
#include "../util_namespace.cuh"
#include "dispatch/dispatch_segmented_sort.cuh"

CUB_NAMESPACE_BEGIN


/**
 * @brief DeviceSegmentedSort provides device-wide, parallel operations for
 *        computing a batched sort across multiple, non-overlapping sequences of
 *        data items residing within device-accessible memory.
 *        ![](segmented_sorting_logo.png)
 * @ingroup SegmentedModule
 *
 * @par Overview
 * The algorithm arranges items into ascending (or descending) order.
 * The underlying sorting algorithm is undefined. Depending on the segment size,
 * it might be radix sort, merge sort or something else. Therefore, no
 * assumptions on the underlying implementation should be made.
 *
 * @par Differences from DeviceSegmentedRadixSort
 * DeviceSegmentedRadixSort is optimized for significantly large segments (tens
 * of thousands of items and more). Nevertheless, some domains produce a wide
 * range of segment sizes. DeviceSegmentedSort partitions segments into size
 * groups and specialize sorting algorithms for each group. This approach leads
 * to better resource utilization in the presence of segment size imbalance or
 * moderate segment sizes (up to thousands of items).
 *
 * @par Supported Types
 * The algorithm has to satisfy the underlying algorithms restrictions. Radix
 * sort usage restricts the list of supported types. Therefore,
 * DeviceSegmentedSort can sort all of the built-in C++ numeric primitive types
 * (`unsigned char`, `int`, `double`, etc.) as well as CUDA's `__half` and
 * `__nv_bfloat16` 16-bit floating-point types.
 *
 * @par Floating-Point Special Cases
 * - Positive and negative zeros are considered equivalent, and will be treated
 *   as such in the output.
 * - No special handling is implemented for NaN values; these are sorted
 *   according to their bit representations after any transformations.
 *
 * @par A simple example
 * @code
 * #include <cub/cub.cuh>   // or equivalently <cub/device/device_segmented_sort.cuh>
 *
 * // Declare, allocate, and initialize device-accessible pointers for sorting data
 * int  num_items;          // e.g., 7
 * int  num_segments;       // e.g., 3
 * int  *d_offsets;         // e.g., [0, 3, 3, 7]
 * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
 * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
 * int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
 * int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
 * ...
 *
 * // Determine temporary device storage requirements
 * void     *d_temp_storage = NULL;
 * size_t   temp_storage_bytes = 0;
 * cub::DeviceSegmentedSort::SortPairs(
 *     d_temp_storage, temp_storage_bytes,
 *     d_keys_in, d_keys_out, d_values_in, d_values_out,
 *     num_items, num_segments, d_offsets, d_offsets + 1);
 *
 * // Allocate temporary storage
 * cudaMalloc(&d_temp_storage, temp_storage_bytes);
 *
 * // Run sorting operation
 * cub::DeviceSegmentedSort::SortPairs(
 *     d_temp_storage, temp_storage_bytes,
 *     d_keys_in, d_keys_out, d_values_in, d_values_out,
 *     num_items, num_segments, d_offsets, d_offsets + 1);
 *
 * // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
 * // d_values_out          <-- [1, 2, 0, 5, 4, 3, 6]
 * @endcode
 */
struct DeviceSegmentedSort
{

  /*************************************************************************//**
   * @name Keys-only
   ****************************************************************************/
  //@{

  /**
   * @brief Sorts segments of keys into ascending order. Approximately
   *        <tt>num_items + 2*num_segments</tt> auxiliary storage required.
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation
   * - When input a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as <tt>segment_offsets+1</tt>).
   * - SortKeys is not guaranteed to be stable. That is, suppose that @p i and
   *   @p j are equivalent: neither one is less than the other. It is not
   *   guaranteed that the relative order of these two elements will be
   *   preserved by sort.
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Determine temporary device storage requirements
   * void    *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::SortKeys(
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::SortKeys(
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
   * @endcode
   *
   * @tparam KeyT                  <b>[inferred]</b> Key type
   * @tparam OffsetT               <b>[inferred]</b> Integer type for global offsets
   * @tparam BeginOffsetIteratorT  <b>[inferred]</b> Random-access input iterator type for reading segment beginning offsets \iterator
   * @tparam EndOffsetIteratorT    <b>[inferred]</b> Random-access input iterator type for reading segment ending offsets \iterator
   */
  template <typename KeyT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           std::size_t &temp_storage_bytes,
           const KeyT *d_keys_in,
           KeyT *d_keys_out,
           OffsetT num_items,
           unsigned int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           cudaStream_t stream    = 0,
           bool debug_synchronous = false)
  {
    constexpr bool is_descending = false;
    constexpr bool is_overwrite_okay = false;
    using DispatchT = DispatchSegmentedSort<is_descending,
                                            KeyT,
                                            cub::NullType,
                                            OffsetT,
                                            BeginOffsetIteratorT,
                                            EndOffsetIteratorT>;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<NullType> d_values;

    return DispatchT::Dispatch(d_temp_storage,
                               temp_storage_bytes,
                               d_keys,
                               d_values,
                               num_items,
                               num_segments,
                               d_begin_offsets,
                               d_end_offsets,
                               is_overwrite_okay,
                               stream,
                               debug_synchronous);
  }

  /**
   * @brief Sorts segments of keys into descending order. Approximately
   *        <tt>num_items + 2*num_segments</tt> auxiliary storage required.
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation
   * - When input a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as <tt>segment_offsets+1</tt>).
   * - SortKeysDescending is not guaranteed to be stable. That is, suppose that
   *   @p i and @p j are equivalent: neither one is less than the other. It is
   *   not guaranteed that the relative order of these two elements will be
   *   preserved by sort.
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Determine temporary device storage requirements
   * void    *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::SortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::SortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
   * @endcode
   *
   * @tparam KeyT                  <b>[inferred]</b> Key type
   * @tparam OffsetT               <b>[inferred]</b> Integer type for global offsets
   * @tparam BeginOffsetIteratorT  <b>[inferred]</b> Random-access input iterator type for reading segment beginning offsets \iterator
   * @tparam EndOffsetIteratorT    <b>[inferred]</b> Random-access input iterator type for reading segment ending offsets \iterator
   */
  template <typename KeyT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysDescending(void *d_temp_storage,
                     std::size_t &temp_storage_bytes,
                     const KeyT *d_keys_in,
                     KeyT *d_keys_out,
                     OffsetT num_items,
                     unsigned int num_segments,
                     BeginOffsetIteratorT d_begin_offsets,
                     EndOffsetIteratorT d_end_offsets,
                     cudaStream_t stream    = 0,
                     bool debug_synchronous = false)
  {
    constexpr bool is_descending = true;
    constexpr bool is_overwrite_okay = false;
    using DispatchT = DispatchSegmentedSort<is_descending,
                                            KeyT,
                                            cub::NullType,
                                            OffsetT,
                                            BeginOffsetIteratorT,
                                            EndOffsetIteratorT>;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<NullType> d_values;

    return DispatchT::Dispatch(d_temp_storage,
                               temp_storage_bytes,
                               d_keys,
                               d_values,
                               num_items,
                               num_segments,
                               d_begin_offsets,
                               d_end_offsets,
                               is_overwrite_okay,
                               stream,
                               debug_synchronous);
  }

  /**
   * @brief Sorts segments of keys into ascending order. Approximately
   *        <tt>2*num_segments</tt> auxiliary storage required.
   *
   * @par
   * - The sorting operation is given a pair of key buffers managed by a
   *   DoubleBuffer structure that indicates which of the two buffers is
   *   "current" (and thus contains the input data to be sorted).
   * - The contents of both buffers may be altered by the sorting operation.
   * - Upon completion, the sorting operation will update the "current"
   *   indicator within the DoubleBuffer wrapper to reference which of the two
   *   buffers now contains the sorted output sequence (a function of the number
   *   of key bits and the targeted device architecture).
   * - When input a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as <tt>segment_offsets+1</tt>).
   * - SortKeys is not guaranteed to be stable. That is, suppose that
   *   @p i and @p j are equivalent: neither one is less than the other. It is
   *   not guaranteed that the relative order of these two elements will be
   *   preserved by sort.
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of \p int keys.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Create a DoubleBuffer to wrap the pair of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::SortKeys(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::SortKeys(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
   * @endcode
   *
   * @tparam KeyT                  <b>[inferred]</b> Key type
   * @tparam OffsetT               <b>[inferred]</b> Integer type for global offsets
   * @tparam BeginOffsetIteratorT  <b>[inferred]</b> Random-access input iterator type for reading segment beginning offsets \iterator
   * @tparam EndOffsetIteratorT    <b>[inferred]</b> Random-access input iterator type for reading segment ending offsets \iterator
   */
  template <typename KeyT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           std::size_t &temp_storage_bytes,
           DoubleBuffer<KeyT> &d_keys,
           OffsetT num_items,
           unsigned int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           cudaStream_t stream    = 0,
           bool debug_synchronous = false)
  {
    constexpr bool is_descending     = false;
    constexpr bool is_overwrite_okay = true;

    using DispatchT = DispatchSegmentedSort<is_descending,
                                            KeyT,
                                            cub::NullType,
                                            OffsetT,
                                            BeginOffsetIteratorT,
                                            EndOffsetIteratorT>;

    DoubleBuffer<NullType> d_values;

    return DispatchT::Dispatch(d_temp_storage,
                               temp_storage_bytes,
                               d_keys,
                               d_values,
                               num_items,
                               num_segments,
                               d_begin_offsets,
                               d_end_offsets,
                               is_overwrite_okay,
                               stream,
                               debug_synchronous);
  }

  /**
   * @brief Sorts segments of keys into descending order. Approximately
   *        <tt>2*num_segments</tt> auxiliary storage required.
   *
   * @par
   * - The sorting operation is given a pair of key buffers managed by a
   *   DoubleBuffer structure that indicates which of the two buffers is
   *   "current" (and thus contains the input data to be sorted).
   * - The contents of both buffers may be altered by the sorting operation.
   * - Upon completion, the sorting operation will update the "current"
   *   indicator within the DoubleBuffer wrapper to reference which of the two
   *   buffers now contains the sorted output sequence (a function of the number
   *   of key bits and the targeted device architecture).
   * - When input a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as <tt>segment_offsets+1</tt>).
   * - SortKeysDescending is not guaranteed to be stable. That is, suppose that
   *   @p i and @p j are equivalent: neither one is less than the other. It is
   *   not guaranteed that the relative order of these two elements will be
   *   preserved by sort.
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of \p int keys.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Create a DoubleBuffer to wrap the pair of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::SortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::SortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
   * @endcode
   *
   * @tparam KeyT                  <b>[inferred]</b> Key type
   * @tparam OffsetT               <b>[inferred]</b> Integer type for global offsets
   * @tparam BeginOffsetIteratorT  <b>[inferred]</b> Random-access input iterator type for reading segment beginning offsets \iterator
   * @tparam EndOffsetIteratorT    <b>[inferred]</b> Random-access input iterator type for reading segment ending offsets \iterator
   */
  template <typename KeyT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysDescending(void *d_temp_storage,
                     std::size_t &temp_storage_bytes,
                     DoubleBuffer<KeyT> &d_keys,
                     OffsetT num_items,
                     unsigned int num_segments,
                     BeginOffsetIteratorT d_begin_offsets,
                     EndOffsetIteratorT d_end_offsets,
                     cudaStream_t stream    = 0,
                     bool debug_synchronous = false)
  {
    constexpr bool is_descending = true;
    constexpr bool is_overwrite_okay = true;

    using DispatchT = DispatchSegmentedSort<is_descending,
                                            KeyT,
                                            cub::NullType,
                                            OffsetT,
                                            BeginOffsetIteratorT,
                                            EndOffsetIteratorT>;

    DoubleBuffer<NullType> d_values;

    return DispatchT::Dispatch(d_temp_storage,
                               temp_storage_bytes,
                               d_keys,
                               d_values,
                               num_items,
                               num_segments,
                               d_begin_offsets,
                               d_end_offsets,
                               is_overwrite_okay,
                               stream,
                               debug_synchronous);
  }

  /**
   * @brief Sorts segments of keys into ascending order. Approximately
   *        <tt>num_items + 2*num_segments</tt> auxiliary storage required.
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation
   * - When input a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as <tt>segment_offsets+1</tt>).
   * - StableSortKeys is stable: it preserves the relative ordering of
   *   equivalent elements. That is, if @p x and @p y are elements such that
   *   @p x precedes @p y, and if the two elements are equivalent (neither
   *   @p x < @p y nor @p y < @p x) then a postcondition of stable sort is that
   *   @p x still precedes @p y.
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Determine temporary device storage requirements
   * void    *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::StableSortKeys(
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::StableSortKeys(
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
   * @endcode
   *
   * @tparam KeyT                  <b>[inferred]</b> Key type
   * @tparam OffsetT               <b>[inferred]</b> Integer type for global offsets
   * @tparam BeginOffsetIteratorT  <b>[inferred]</b> Random-access input iterator type for reading segment beginning offsets \iterator
   * @tparam EndOffsetIteratorT    <b>[inferred]</b> Random-access input iterator type for reading segment ending offsets \iterator
   */
  template <typename KeyT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortKeys(void *d_temp_storage,
                 std::size_t &temp_storage_bytes,
                 const KeyT *d_keys_in,
                 KeyT *d_keys_out,
                 OffsetT num_items,
                 unsigned int num_segments,
                 BeginOffsetIteratorT d_begin_offsets,
                 EndOffsetIteratorT d_end_offsets,
                 cudaStream_t stream    = 0,
                 bool debug_synchronous = false)
  {
    return SortKeys<KeyT, OffsetT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      stream,
      debug_synchronous);
  }

  /**
   * @brief Sorts segments of keys into descending order. Approximately
   *        <tt>num_items + 2*num_segments</tt> auxiliary storage required.
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation
   * - When input a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as <tt>segment_offsets+1</tt>).
   * - StableSortKeysDescending is stable: it preserves the relative ordering of
   *   equivalent elements. That is, if @p x and @p y are elements such that
   *   @p x precedes @p y, and if the two elements are equivalent (neither
   *   @p x < @p y nor @p y < @p x) then a postcondition of stable sort is that
   *   @p x still precedes @p y.
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Determine temporary device storage requirements
   * void    *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::StableSortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::StableSortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
   * @endcode
   *
   * @tparam KeyT                  <b>[inferred]</b> Key type
   * @tparam OffsetT               <b>[inferred]</b> Integer type for global offsets
   * @tparam BeginOffsetIteratorT  <b>[inferred]</b> Random-access input iterator type for reading segment beginning offsets \iterator
   * @tparam EndOffsetIteratorT    <b>[inferred]</b> Random-access input iterator type for reading segment ending offsets \iterator
   */
  template <typename KeyT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortKeysDescending(void *d_temp_storage,
                           std::size_t &temp_storage_bytes,
                           const KeyT *d_keys_in,
                           KeyT *d_keys_out,
                           OffsetT num_items,
                           unsigned int num_segments,
                           BeginOffsetIteratorT d_begin_offsets,
                           EndOffsetIteratorT d_end_offsets,
                           cudaStream_t stream    = 0,
                           bool debug_synchronous = false)
  {
    return SortKeysDescending<KeyT,
                              OffsetT,
                              BeginOffsetIteratorT,
                              EndOffsetIteratorT>(d_temp_storage,
                                                  temp_storage_bytes,
                                                  d_keys_in,
                                                  d_keys_out,
                                                  num_items,
                                                  num_segments,
                                                  d_begin_offsets,
                                                  d_end_offsets,
                                                  stream,
                                                  debug_synchronous);
  }

  /**
   * @brief Sorts segments of keys into ascending order. Approximately
   *        <tt>2*num_segments</tt> auxiliary storage required.
   *
   * @par
   * - The sorting operation is given a pair of key buffers managed by a
   *   DoubleBuffer structure that indicates which of the two buffers is
   *   "current" (and thus contains the input data to be sorted).
   * - The contents of both buffers may be altered by the sorting operation.
   * - Upon completion, the sorting operation will update the "current"
   *   indicator within the DoubleBuffer wrapper to reference which of the two
   *   buffers now contains the sorted output sequence (a function of the number
   *   of key bits and the targeted device architecture).
   * - When input a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as <tt>segment_offsets+1</tt>).
   * - StableSortKeys is stable: it preserves the relative ordering of
   *   equivalent elements. That is, if @p x and @p y are elements such that
   *   @p x precedes @p y, and if the two elements are equivalent (neither
   *   @p x < @p y nor @p y < @p x) then a postcondition of stable sort is that
   *   @p x still precedes @p y.
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of \p int keys.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Create a DoubleBuffer to wrap the pair of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::StableSortKeys(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::StableSortKeys(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
   * @endcode
   *
   * @tparam KeyT                  <b>[inferred]</b> Key type
   * @tparam OffsetT               <b>[inferred]</b> Integer type for global offsets
   * @tparam BeginOffsetIteratorT  <b>[inferred]</b> Random-access input iterator type for reading segment beginning offsets \iterator
   * @tparam EndOffsetIteratorT    <b>[inferred]</b> Random-access input iterator type for reading segment ending offsets \iterator
   */
  template <typename KeyT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortKeys(void *d_temp_storage,
                 std::size_t &temp_storage_bytes,
                 DoubleBuffer<KeyT> &d_keys,
                 OffsetT num_items,
                 unsigned int num_segments,
                 BeginOffsetIteratorT d_begin_offsets,
                 EndOffsetIteratorT d_end_offsets,
                 cudaStream_t stream    = 0,
                 bool debug_synchronous = false)
  {
    return SortKeys<KeyT, OffsetT, BeginOffsetIteratorT, EndOffsetIteratorT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys,
      num_items,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      stream,
      debug_synchronous);
  }

  /**
   * @brief Sorts segments of keys into descending order. Approximately
   *        <tt>2*num_segments</tt> auxiliary storage required.
   *
   * @par
   * - The sorting operation is given a pair of key buffers managed by a
   *   DoubleBuffer structure that indicates which of the two buffers is
   *   "current" (and thus contains the input data to be sorted).
   * - The contents of both buffers may be altered by the sorting operation.
   * - Upon completion, the sorting operation will update the "current"
   *   indicator within the DoubleBuffer wrapper to reference which of the two
   *   buffers now contains the sorted output sequence (a function of the number
   *   of key bits and the targeted device architecture).
   * - When input a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as <tt>segment_offsets+1</tt>).
   * - StableSortKeysDescending is stable: it preserves the relative ordering of
   *   equivalent elements. That is, if @p x and @p y are elements such that
   *   @p x precedes @p y, and if the two elements are equivalent (neither
   *   @p x < @p y nor @p y < @p x) then a postcondition of stable sort is that
   *   @p x still precedes @p y.
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of \p int keys.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Create a DoubleBuffer to wrap the pair of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::StableSortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::StableSortKeysDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
   * @endcode
   *
   * @tparam KeyT                  <b>[inferred]</b> Key type
   * @tparam OffsetT               <b>[inferred]</b> Integer type for global offsets
   * @tparam BeginOffsetIteratorT  <b>[inferred]</b> Random-access input iterator type for reading segment beginning offsets \iterator
   * @tparam EndOffsetIteratorT    <b>[inferred]</b> Random-access input iterator type for reading segment ending offsets \iterator
   */
  template <typename KeyT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortKeysDescending(void *d_temp_storage,
                           std::size_t &temp_storage_bytes,
                           DoubleBuffer<KeyT> &d_keys,
                           OffsetT num_items,
                           unsigned int num_segments,
                           BeginOffsetIteratorT d_begin_offsets,
                           EndOffsetIteratorT d_end_offsets,
                           cudaStream_t stream    = 0,
                           bool debug_synchronous = false)
  {
    return SortKeysDescending<KeyT,
                              OffsetT,
                              BeginOffsetIteratorT,
                              EndOffsetIteratorT>(d_temp_storage,
                                                  temp_storage_bytes,
                                                  d_keys,
                                                  num_items,
                                                  num_segments,
                                                  d_begin_offsets,
                                                  d_end_offsets,
                                                  stream,
                                                  debug_synchronous);
  }

  //@}  end member group
  /*************************************************************************//**
   * @name Key-value pairs
   ****************************************************************************/
  //@{

  /**
   * @brief Sorts segments of key-value pairs into ascending order. Approximately
   *        <tt>2*num_items + 2*num_segments</tt> auxiliary storage required.
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation
   * - When input a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as <tt>segment_offsets+1</tt>).
   * - SortPairs is not guaranteed to be stable. That is, suppose that @p i and
   *   @p j are equivalent: neither one is less than the other. It is not
   *   guaranteed that the relative order of these two elements will be
   *   preserved by sort.
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys with associated vector of
   * @p int values.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
   * int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::SortPairs(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::SortPairs(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
   * // d_values_out          <-- [1, 2, 0, 5, 4, 3, 6]
   * @endcode
   *
   * @tparam KeyT                  <b>[inferred]</b> Key type
   * @tparam ValueT                <b>[inferred]</b> Value type
   * @tparam OffsetT               <b>[inferred]</b> Integer type for global offsets
   * @tparam BeginOffsetIteratorT  <b>[inferred]</b> Random-access input iterator type for reading segment beginning offsets \iterator
   * @tparam EndOffsetIteratorT    <b>[inferred]</b> Random-access input iterator type for reading segment ending offsets \iterator
   */
  template <typename KeyT,
            typename ValueT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairs(void *d_temp_storage,
           std::size_t &temp_storage_bytes,
           const KeyT *d_keys_in,
           KeyT *d_keys_out,
           const ValueT *d_values_in,
           ValueT *d_values_out,
           OffsetT num_items,
           unsigned int num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           cudaStream_t stream = 0,
           bool debug_synchronous = false)
  {
    constexpr bool is_descending = false;
    constexpr bool is_overwrite_okay = false;
    using DispatchT = DispatchSegmentedSort<is_descending,
                                            KeyT,
                                            ValueT,
                                            OffsetT,
                                            BeginOffsetIteratorT,
                                            EndOffsetIteratorT>;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<ValueT> d_values(const_cast<ValueT *>(d_values_in), d_values_out);

    return DispatchT::Dispatch(d_temp_storage,
                               temp_storage_bytes,
                               d_keys,
                               d_values,
                               num_items,
                               num_segments,
                               d_begin_offsets,
                               d_end_offsets,
                               is_overwrite_okay,
                               stream,
                               debug_synchronous);
  }

  /**
   * @brief Sorts segments of key-value pairs into descending order. Approximately
   *        <tt>2*num_items + 2*num_segments</tt> auxiliary storage required.
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation
   * - When input a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as <tt>segment_offsets+1</tt>).
   * - SortPairs is not guaranteed to be stable. That is, suppose that @p i and
   *   @p j are equivalent: neither one is less than the other. It is not
   *   guaranteed that the relative order of these two elements will be
   *   preserved by sort.
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys with associated vector of
   * @p int values.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
   * int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Determine temporary device storage requirements
   * void    *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::SortPairsDescending(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::SortPairsDescending(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
   * // d_values_out          <-- [1, 2, 0, 5, 4, 3, 6]
   * @endcode
   *
   * @tparam KeyT                  <b>[inferred]</b> Key type
   * @tparam ValueT                <b>[inferred]</b> Value type
   * @tparam OffsetT               <b>[inferred]</b> Integer type for global offsets
   * @tparam BeginOffsetIteratorT  <b>[inferred]</b> Random-access input iterator type for reading segment beginning offsets \iterator
   * @tparam EndOffsetIteratorT    <b>[inferred]</b> Random-access input iterator type for reading segment ending offsets \iterator
   */
  template <typename KeyT,
            typename ValueT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairsDescending(void *d_temp_storage,
                      std::size_t &temp_storage_bytes,
                      const KeyT *d_keys_in,
                      KeyT *d_keys_out,
                      const ValueT *d_values_in,
                      ValueT *d_values_out,
                      OffsetT num_items,
                      unsigned int num_segments,
                      BeginOffsetIteratorT d_begin_offsets,
                      EndOffsetIteratorT d_end_offsets,
                      cudaStream_t stream    = 0,
                      bool debug_synchronous = false)
  {
    constexpr bool is_descending = true;
    constexpr bool is_overwrite_okay = false;
    using DispatchT = DispatchSegmentedSort<is_descending,
                                            KeyT,
                                            ValueT,
                                            OffsetT,
                                            BeginOffsetIteratorT,
                                            EndOffsetIteratorT>;

    DoubleBuffer<KeyT> d_keys(const_cast<KeyT *>(d_keys_in), d_keys_out);
    DoubleBuffer<ValueT> d_values(const_cast<ValueT *>(d_values_in), d_values_out);

    return DispatchT::Dispatch(d_temp_storage,
                               temp_storage_bytes,
                               d_keys,
                               d_values,
                               num_items,
                               num_segments,
                               d_begin_offsets,
                               d_end_offsets,
                               is_overwrite_okay,
                               stream,
                               debug_synchronous);
  }

  /**
   * @brief Sorts segments of key-value pairs into ascending order. Approximately
   *        <tt>2*num_segments</tt> auxiliary storage required.
   *
   * @par
   * - The sorting operation is given a pair of key buffers and a corresponding
   *   pair of associated value buffers.  Each pair is managed by a DoubleBuffer
   *   structure that indicates which of the two buffers is "current" (and thus
   *   contains the input data to be sorted).
   * - The contents of both buffers within each pair may be altered by the sorting
   *   operation.
   * - Upon completion, the sorting operation will update the "current" indicator
   *   within each DoubleBuffer wrapper to reference which of the two buffers
   *   now contains the sorted output sequence (a function of the number of key bits
   *   specified and the targeted device architecture).
   * - When input a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as <tt>segment_offsets+1</tt>).
   * - SortPairs is not guaranteed to be stable. That is, suppose that @p i and
   *   @p j are equivalent: neither one is less than the other. It is not
   *   guaranteed that the relative order of these two elements will be
   *   preserved by sort.
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of \p int keys with associated vector of
   * @p int values.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
   * int  *d_value_buf;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_value_alt_buf;   // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Create a set of DoubleBuffers to wrap pairs of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   * cub::DoubleBuffer<int> d_values(d_value_buf, d_value_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::SortPairs(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::SortPairs(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
   * // d_values.Current()    <-- [5, 4, 3, 1, 2, 0, 6]
   *
   * @endcode
   *
   * @tparam KeyT                  <b>[inferred]</b> Key type
   * @tparam ValueT                <b>[inferred]</b> Value type
   * @tparam OffsetT               <b>[inferred]</b> Integer type for global offsets
   * @tparam BeginOffsetIteratorT  <b>[inferred]</b> Random-access input iterator type for reading segment beginning offsets \iterator
   * @tparam EndOffsetIteratorT    <b>[inferred]</b> Random-access input iterator type for reading segment ending offsets \iterator
   */
  template <typename KeyT,
            typename ValueT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairs(void *d_temp_storage,
            std::size_t &temp_storage_bytes,
            DoubleBuffer<KeyT> &d_keys,
            DoubleBuffer<ValueT> &d_values,
            OffsetT num_items,
            unsigned int num_segments,
            BeginOffsetIteratorT d_begin_offsets,
            EndOffsetIteratorT d_end_offsets,
            cudaStream_t stream    = 0,
            bool debug_synchronous = false)
  {
    constexpr bool is_descending = false;
    constexpr bool is_overwrite_okay = true;
    using DispatchT = DispatchSegmentedSort<is_descending,
                                            KeyT,
                                            ValueT,
                                            OffsetT,
                                            BeginOffsetIteratorT,
                                            EndOffsetIteratorT>;

    return DispatchT::Dispatch(d_temp_storage,
                               temp_storage_bytes,
                               d_keys,
                               d_values,
                               num_items,
                               num_segments,
                               d_begin_offsets,
                               d_end_offsets,
                               is_overwrite_okay,
                               stream,
                               debug_synchronous);
  }

  /**
   * @brief Sorts segments of key-value pairs into descending order.
   *        Approximately <tt>2*num_segments</tt> auxiliary storage required.
   *
   * @par
   * - The sorting operation is given a pair of key buffers and a corresponding
   *   pair of associated value buffers. Each pair is managed by a DoubleBuffer
   *   structure that indicates which of the two buffers is "current" (and thus
   *   contains the input data to be sorted).
   * - The contents of both buffers within each pair may be altered by the sorting
   *   operation.
   * - Upon completion, the sorting operation will update the "current" indicator
   *   within each DoubleBuffer wrapper to reference which of the two buffers
   *   now contains the sorted output sequence (a function of the number of key bits
   *   specified and the targeted device architecture).
   * - When input a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as <tt>segment_offsets+1</tt>).
   * - SortPairsDescending is not guaranteed to be stable. That is, suppose that @p i and
   *   @p j are equivalent: neither one is less than the other. It is not
   *   guaranteed that the relative order of these two elements will be
   *   preserved by sort.
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of \p int keys with associated vector of
   * @p int values.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
   * int  *d_value_buf;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_value_alt_buf;   // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Create a set of DoubleBuffers to wrap pairs of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   * cub::DoubleBuffer<int> d_values(d_value_buf, d_value_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::SortPairsDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::SortPairsDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
   * // d_values.Current()    <-- [5, 4, 3, 1, 2, 0, 6]
   *
   * @endcode
   *
   * @tparam KeyT                  <b>[inferred]</b> Key type
   * @tparam ValueT                <b>[inferred]</b> Value type
   * @tparam OffsetT               <b>[inferred]</b> Integer type for global offsets
   * @tparam BeginOffsetIteratorT  <b>[inferred]</b> Random-access input iterator type for reading segment beginning offsets \iterator
   * @tparam EndOffsetIteratorT    <b>[inferred]</b> Random-access input iterator type for reading segment ending offsets \iterator
   */
  template <typename KeyT,
            typename ValueT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortPairsDescending(void *d_temp_storage,
                      std::size_t &temp_storage_bytes,
                      DoubleBuffer<KeyT> &d_keys,
                      DoubleBuffer<ValueT> &d_values,
                      OffsetT num_items,
                      unsigned int num_segments,
                      BeginOffsetIteratorT d_begin_offsets,
                      EndOffsetIteratorT d_end_offsets,
                      cudaStream_t stream    = 0,
                      bool debug_synchronous = false)
  {
    constexpr bool is_descending = true;
    constexpr bool is_overwrite_okay = true;
    using DispatchT = DispatchSegmentedSort<is_descending,
                                            KeyT,
                                            ValueT,
                                            OffsetT,
                                            BeginOffsetIteratorT,
                                            EndOffsetIteratorT>;

    return DispatchT::Dispatch(d_temp_storage,
                               temp_storage_bytes,
                               d_keys,
                               d_values,
                               num_items,
                               num_segments,
                               d_begin_offsets,
                               d_end_offsets,
                               is_overwrite_okay,
                               stream,
                               debug_synchronous);
  }

  /**
   * @brief Sorts segments of key-value pairs into ascending order. Approximately
   *        <tt>2*num_items + 2*num_segments</tt> auxiliary storage required.
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation
   * - When input a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as <tt>segment_offsets+1</tt>).
   * - StableSortPairs is stable: it preserves the relative ordering of
   *   equivalent elements. That is, if @p x and @p y are elements such that
   *   @p x precedes @p y, and if the two elements are equivalent (neither
   *   @p x < @p y nor @p y < @p x) then a postcondition of stable sort is that
   *   @p x still precedes @p y.
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys with associated vector of
   * @p int values.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
   * int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::StableSortPairs(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::StableSortPairs(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
   * // d_values_out          <-- [1, 2, 0, 5, 4, 3, 6]
   * @endcode
   *
   * @tparam KeyT                  <b>[inferred]</b> Key type
   * @tparam ValueT                <b>[inferred]</b> Value type
   * @tparam OffsetT               <b>[inferred]</b> Integer type for global offsets
   * @tparam BeginOffsetIteratorT  <b>[inferred]</b> Random-access input iterator type for reading segment beginning offsets \iterator
   * @tparam EndOffsetIteratorT    <b>[inferred]</b> Random-access input iterator type for reading segment ending offsets \iterator
   */
  template <typename KeyT,
            typename ValueT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortPairs(void *d_temp_storage,
                  std::size_t &temp_storage_bytes,
                  const KeyT *d_keys_in,
                  KeyT *d_keys_out,
                  const ValueT *d_values_in,
                  ValueT *d_values_out,
                  OffsetT num_items,
                  unsigned int num_segments,
                  BeginOffsetIteratorT d_begin_offsets,
                  EndOffsetIteratorT d_end_offsets,
                  cudaStream_t stream    = 0,
                  bool debug_synchronous = false)
  {
    return SortPairs<KeyT,
                     ValueT,
                     OffsetT,
                     BeginOffsetIteratorT,
                     EndOffsetIteratorT>(d_temp_storage,
                                         temp_storage_bytes,
                                         d_keys_in,
                                         d_keys_out,
                                         d_values_in,
                                         d_values_out,
                                         num_items,
                                         num_segments,
                                         d_begin_offsets,
                                         d_end_offsets,
                                         stream,
                                         debug_synchronous);
  }

  /**
   * @brief Sorts segments of key-value pairs into descending order.
   *        Approximately <tt>2*num_items + 2*num_segments</tt> auxiliary
   *        storage required.
   *
   * @par
   * - The contents of the input data are not altered by the sorting operation
   * - When input a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as <tt>segment_offsets+1</tt>).
   * - StableSortPairsDescending is stable: it preserves the relative ordering
   *   of equivalent elements. That is, if @p x and @p y are elements such that
   *   @p x precedes @p y, and if the two elements are equivalent (neither
   *   @p x < @p y nor @p y < @p x) then a postcondition of stable sort is that
   *   @p x still precedes @p y.
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of @p int keys with associated vector of
   * @p int values.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
   * int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::StableSortPairsDescending(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::StableSortPairsDescending(
   *     d_temp_storage, temp_storage_bytes,
   *     d_keys_in, d_keys_out, d_values_in, d_values_out,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
   * // d_values_out          <-- [1, 2, 0, 5, 4, 3, 6]
   * @endcode
   *
   * @tparam KeyT                  <b>[inferred]</b> Key type
   * @tparam ValueT                <b>[inferred]</b> Value type
   * @tparam OffsetT               <b>[inferred]</b> Integer type for global offsets
   * @tparam BeginOffsetIteratorT  <b>[inferred]</b> Random-access input iterator type for reading segment beginning offsets \iterator
   * @tparam EndOffsetIteratorT    <b>[inferred]</b> Random-access input iterator type for reading segment ending offsets \iterator
   */
  template <typename KeyT,
            typename ValueT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortPairsDescending(void *d_temp_storage,
                            std::size_t &temp_storage_bytes,
                            const KeyT *d_keys_in,
                            KeyT *d_keys_out,
                            const ValueT *d_values_in,
                            ValueT *d_values_out,
                            OffsetT num_items,
                            unsigned int num_segments,
                            BeginOffsetIteratorT d_begin_offsets,
                            EndOffsetIteratorT d_end_offsets,
                            cudaStream_t stream    = 0,
                            bool debug_synchronous = false)
  {
    return SortPairsDescending<KeyT,
                               ValueT,
                               OffsetT,
                               BeginOffsetIteratorT,
                               EndOffsetIteratorT>(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_keys_in,
                                                   d_keys_out,
                                                   d_values_in,
                                                   d_values_out,
                                                   num_items,
                                                   num_segments,
                                                   d_begin_offsets,
                                                   d_end_offsets,
                                                   stream,
                                                   debug_synchronous);
  }

  /**
   * @brief Sorts segments of key-value pairs into ascending order. Approximately
   *        <tt>2*num_segments</tt> auxiliary storage required.
   *
   * @par
   * - The sorting operation is given a pair of key buffers and a corresponding
   *   pair of associated value buffers.  Each pair is managed by a DoubleBuffer
   *   structure that indicates which of the two buffers is "current" (and thus
   *   contains the input data to be sorted).
   * - The contents of both buffers within each pair may be altered by the sorting
   *   operation.
   * - Upon completion, the sorting operation will update the "current" indicator
   *   within each DoubleBuffer wrapper to reference which of the two buffers
   *   now contains the sorted output sequence (a function of the number of key bits
   *   specified and the targeted device architecture).
   * - When input a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as <tt>segment_offsets+1</tt>).
   * - StableSortPairs is stable: it preserves the relative ordering
   *   of equivalent elements. That is, if @p x and @p y are elements such that
   *   @p x precedes @p y, and if the two elements are equivalent (neither
   *   @p x < @p y nor @p y < @p x) then a postcondition of stable sort is that
   *   @p x still precedes @p y.
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of \p int keys with associated vector of
   * @p int values.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
   * int  *d_value_buf;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_value_alt_buf;   // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Create a set of DoubleBuffers to wrap pairs of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   * cub::DoubleBuffer<int> d_values(d_value_buf, d_value_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::StableSortPairs(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::StableSortPairs(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
   * // d_values.Current()    <-- [5, 4, 3, 1, 2, 0, 6]
   *
   * @endcode
   *
   * @tparam KeyT                  <b>[inferred]</b> Key type
   * @tparam ValueT                <b>[inferred]</b> Value type
   * @tparam OffsetT               <b>[inferred]</b> Integer type for global offsets
   * @tparam BeginOffsetIteratorT  <b>[inferred]</b> Random-access input iterator type for reading segment beginning offsets \iterator
   * @tparam EndOffsetIteratorT    <b>[inferred]</b> Random-access input iterator type for reading segment ending offsets \iterator
   */
  template <typename KeyT,
            typename ValueT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortPairs(void *d_temp_storage,
                  std::size_t &temp_storage_bytes,
                  DoubleBuffer<KeyT> &d_keys,
                  DoubleBuffer<ValueT> &d_values,
                  OffsetT num_items,
                  unsigned int num_segments,
                  BeginOffsetIteratorT d_begin_offsets,
                  EndOffsetIteratorT d_end_offsets,
                  cudaStream_t stream    = 0,
                  bool debug_synchronous = false)
  {
    return SortPairs<KeyT,
                     ValueT,
                     OffsetT,
                     BeginOffsetIteratorT,
                     EndOffsetIteratorT>(d_temp_storage,
                                         temp_storage_bytes,
                                         d_keys,
                                         d_values,
                                         num_items,
                                         num_segments,
                                         d_begin_offsets,
                                         d_end_offsets,
                                         stream,
                                         debug_synchronous);
  }

  /**
   * @brief Sorts segments of key-value pairs into descending order.
   *        Approximately <tt>2*num_segments</tt> auxiliary storage required.
   *
   * @par
   * - The sorting operation is given a pair of key buffers and a corresponding
   *   pair of associated value buffers.  Each pair is managed by a DoubleBuffer
   *   structure that indicates which of the two buffers is "current" (and thus
   *   contains the input data to be sorted).
   * - The contents of both buffers within each pair may be altered by the sorting
   *   operation.
   * - Upon completion, the sorting operation will update the "current" indicator
   *   within each DoubleBuffer wrapper to reference which of the two buffers
   *   now contains the sorted output sequence (a function of the number of key bits
   *   specified and the targeted device architecture).
   * - When input a contiguous sequence of segments, a single sequence
   *   @p segment_offsets (of length <tt>num_segments+1</tt>) can be aliased
   *   for both the @p d_begin_offsets and @p d_end_offsets parameters (where
   *   the latter is specified as <tt>segment_offsets+1</tt>).
   * - StableSortPairsDescending is stable: it preserves the relative ordering
   *   of equivalent elements. That is, if @p x and @p y are elements such that
   *   @p x precedes @p y, and if the two elements are equivalent (neither
   *   @p x < @p y nor @p y < @p x) then a postcondition of stable sort is that
   *   @p x still precedes @p y.
   *
   * @par Snippet
   * The code snippet below illustrates the batched sorting of three segments
   * (with one zero-length segment) of \p int keys with associated vector of
   * @p int values.
   *
   * @par
   * @code
   * #include <cub/cub.cuh>   // or equivalently <cub/device/device_segmented_sort.cuh>
   *
   * // Declare, allocate, and initialize device-accessible pointers for sorting data
   * int  num_items;          // e.g., 7
   * int  num_segments;       // e.g., 3
   * int  *d_offsets;         // e.g., [0, 3, 3, 7]
   * int  *d_key_buf;         // e.g., [8, 6, 7, 5, 3, 0, 9]
   * int  *d_key_alt_buf;     // e.g., [-, -, -, -, -, -, -]
   * int  *d_value_buf;       // e.g., [0, 1, 2, 3, 4, 5, 6]
   * int  *d_value_alt_buf;   // e.g., [-, -, -, -, -, -, -]
   * ...
   *
   * // Create a set of DoubleBuffers to wrap pairs of device pointers
   * cub::DoubleBuffer<int> d_keys(d_key_buf, d_key_alt_buf);
   * cub::DoubleBuffer<int> d_values(d_value_buf, d_value_alt_buf);
   *
   * // Determine temporary device storage requirements
   * void     *d_temp_storage = NULL;
   * size_t   temp_storage_bytes = 0;
   * cub::DeviceSegmentedSort::StableSortPairsDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // Allocate temporary storage
   * cudaMalloc(&d_temp_storage, temp_storage_bytes);
   *
   * // Run sorting operation
   * cub::DeviceSegmentedSort::StableSortPairsDescending(
   *     d_temp_storage, temp_storage_bytes, d_keys, d_values,
   *     num_items, num_segments, d_offsets, d_offsets + 1);
   *
   * // d_keys.Current()      <-- [6, 7, 8, 0, 3, 5, 9]
   * // d_values.Current()    <-- [5, 4, 3, 1, 2, 0, 6]
   *
   * @endcode
   *
   * @tparam KeyT                  <b>[inferred]</b> Key type
   * @tparam ValueT                <b>[inferred]</b> Value type
   * @tparam OffsetT               <b>[inferred]</b> Integer type for global offsets
   * @tparam BeginOffsetIteratorT  <b>[inferred]</b> Random-access input iterator type for reading segment beginning offsets \iterator
   * @tparam EndOffsetIteratorT    <b>[inferred]</b> Random-access input iterator type for reading segment ending offsets \iterator
   */
  template <typename KeyT,
            typename ValueT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  StableSortPairsDescending(void *d_temp_storage,
                            std::size_t &temp_storage_bytes,
                            DoubleBuffer<KeyT> &d_keys,
                            DoubleBuffer<ValueT> &d_values,
                            OffsetT num_items,
                            unsigned int num_segments,
                            BeginOffsetIteratorT d_begin_offsets,
                            EndOffsetIteratorT d_end_offsets,
                            cudaStream_t stream    = 0,
                            bool debug_synchronous = false)
  {
    return SortPairsDescending<KeyT,
                               ValueT,
                               OffsetT,
                               BeginOffsetIteratorT,
                               EndOffsetIteratorT>(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_keys,
                                                   d_values,
                                                   num_items,
                                                   num_segments,
                                                   d_begin_offsets,
                                                   d_end_offsets,
                                                   stream,
                                                   debug_synchronous);
  }

  //@}  end member group

};


CUB_NAMESPACE_END
