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

struct DeviceSegmentedSort
{

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
           OffsetT num_segments,
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
                     OffsetT num_segments,
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
           OffsetT num_segments,
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
                      OffsetT num_segments,
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

  template <typename KeyT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeys(void *d_temp_storage,
           std::size_t &temp_storage_bytes,
           DoubleBuffer<KeyT> &d_keys,
           OffsetT num_items,
           OffsetT num_segments,
           BeginOffsetIteratorT d_begin_offsets,
           EndOffsetIteratorT d_end_offsets,
           cudaStream_t stream    = 0,
           bool debug_synchronous = false)
  {
    constexpr bool is_descending = false;
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

  template <typename KeyT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  SortKeysDescending(void *d_temp_storage,
                     std::size_t &temp_storage_bytes,
                     DoubleBuffer<KeyT> &d_keys,
                     OffsetT num_items,
                     OffsetT num_segments,
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
            OffsetT num_segments,
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
                      OffsetT num_segments,
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
};

CUB_NAMESPACE_END
