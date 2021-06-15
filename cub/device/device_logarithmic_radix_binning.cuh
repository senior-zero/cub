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
#include "dispatch/dispatch_logarithmic_radix_binning.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * J. Fox, A. Tripathy, O. Green "Improving Scheduling for Irregular Applications with Logarithmic Radix Binning", IEEE High Performance Extreme Computing Conference (HPEC), 2019
 */
struct DeviceLogarithmicRadixBinning
{
  template <typename SegmentHandlerT>
  static cudaError_t Bin(int num_segments,
                         SegmentHandlerT segment_handler,
                         void *d_storage,
                         size_t &storage_bytes,
                         LogarithmicRadixBinningResult &result)
  {
    return DispatchLogarithmicRadixBinning<SegmentHandlerT>::Dispatch(
      num_segments,
      segment_handler,
      d_storage,
      storage_bytes,
      result);
  }

  template <typename OffsetIteratorT, typename BalancedOffsetIteratorT>
  struct OffsetSegmentHandler
  {
    using offset_size_type = int;

    OffsetIteratorT m_begin_offsets;
    OffsetIteratorT m_end_offsets;
    BalancedOffsetIteratorT m_begin_balanced_offsets;
    BalancedOffsetIteratorT m_end_balanced_offsets;

    OffsetSegmentHandler(OffsetIteratorT begin_offsets,
                         OffsetIteratorT end_offsets,
                         BalancedOffsetIteratorT begin_balanced_offsets,
                         BalancedOffsetIteratorT end_balanced_offsets)
        : m_begin_offsets(begin_offsets)
        , m_end_offsets(end_offsets)
        , m_begin_balanced_offsets(begin_balanced_offsets)
        , m_end_balanced_offsets(end_balanced_offsets)
    {}

    __device__ offset_size_type get_segment_size(int segment_id) const
    {
      return m_end_offsets[segment_id] - m_begin_offsets[segment_id];
    }

    __device__ void set_balanced_position(int segment_id,
                                          int balanced_pos) const
    {
      m_begin_balanced_offsets[balanced_pos] = m_begin_offsets[segment_id];
      m_end_balanced_offsets[balanced_pos]   = m_end_offsets[segment_id];
    }
  };

  template <typename OffsetIteratorT, typename BalancedOffsetIteratorT>
  static cudaError_t
  BinOffsets(int num_segments,
             OffsetIteratorT d_begin_offsets,
             OffsetIteratorT d_end_offsets,
             BalancedOffsetIteratorT d_begin_balanced_offsets,
             BalancedOffsetIteratorT d_end_balanced_offsets,
             void *d_storage,
             size_t &storage_bytes,
             LogarithmicRadixBinningResult &result)
  {
    OffsetSegmentHandler<OffsetIteratorT, BalancedOffsetIteratorT>
      segment_handler(d_begin_offsets,
                      d_end_offsets,
                      d_begin_balanced_offsets,
                      d_end_balanced_offsets);

    return DeviceLogarithmicRadixBinning::Bin(num_segments,
                                              segment_handler,
                                              d_storage,
                                              storage_bytes,
                                              result);
  }
};

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
