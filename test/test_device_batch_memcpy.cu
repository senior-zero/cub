/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "test_util.h"
#include <cub/device/device_batch_memcpy.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <cub/util_ptx.cuh>

template <typename T>
void GenerateRandomData(T *rand_out,
                        const std::size_t num_items,
                        const T min_rand_val          = std::numeric_limits<T>::min(),
                        const T max_rand_val          = std::numeric_limits<T>::max(),
                        const std::uint_fast32_t seed = 320981U,
                        std::enable_if_t<std::is_integral<T>::value && (sizeof(T) >= 2)> * = nullptr)
{
  // initialize random number generator
  std::mt19937 rng(seed);
  std::uniform_int_distribution<T> uni_dist(min_rand_val, max_rand_val);

  // generate random numbers
  for (std::size_t i = 0; i < num_items; ++i)
  {
    rand_out[i] = uni_dist(rng);
  }
}

template <typename InputBufferIt, typename OutputBufferIt, typename BufferSizeIteratorT>
void __global__ BaselineBatchMemCpyKernel(InputBufferIt input_buffer_it,
                                          OutputBufferIt output_buffer_it,
                                          BufferSizeIteratorT buffer_sizes,
                                          uint32_t num_buffers)
{
  uint32_t gtid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gtid >= num_buffers)
    return;
  for (int i = 0; i < buffer_sizes[gtid]; i++)
  {
    reinterpret_cast<uint8_t *>(output_buffer_it[gtid])[i] = reinterpret_cast<uint8_t *>(input_buffer_it[gtid])[i];
  }
}

template <typename InputBufferIt, typename OutputBufferIt, typename BufferSizeIteratorT>
void InvokeBaselineBatchMemcpy(InputBufferIt input_buffer_it,
                               OutputBufferIt output_buffer_it,
                               BufferSizeIteratorT buffer_sizes,
                               uint32_t num_buffers)
{
  constexpr uint32_t block_threads = 128U;
  uint32_t num_blocks              = (num_buffers + block_threads - 1) / block_threads;
  BaselineBatchMemCpyKernel<<<num_blocks, block_threads>>>(input_buffer_it, output_buffer_it, buffer_sizes, num_buffers);
}

template <typename IteratorT>
struct OffsetToPtrOp
{
  template <typename T>
  __host__ __device__ __forceinline__ IteratorT operator()(T offset) const
  {
    return base_it + offset;
  }
  IteratorT base_it;
};

int main(int argc, char **argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  // Type used for indexing into the array of buffers
  using BufferOffsetT = uint32_t;

  // Type used for indexing into individual bytes of a buffer (large enough to cover the max buffer size)
  using BufferSizeT = uint32_t;

  // Type used for indexing into bytes over *all* the buffers' sizes
  using ByteOffsetT = uint32_t;

  using SrcPtrT = uint8_t *;

  BufferOffsetT data_size     = 10000000ULL;
  BufferOffsetT num_buffers   = 2000ULL;
  BufferSizeT min_buffer_size = 1ULL;
  BufferSizeT max_buffer_size = 20000ULL;

  // Buffer segment data (their offsets and sizes)
  std::vector<BufferSizeT> h_buffer_sizes(num_buffers);
  std::vector<ByteOffsetT> h_buffer_src_offsets(num_buffers);
  std::vector<ByteOffsetT> h_buffer_dst_offsets(num_buffers);

  // Device-side resources
  void *d_in                        = nullptr;
  void *d_out                       = nullptr;
  ByteOffsetT *d_buffer_src_offsets = nullptr;
  ByteOffsetT *d_buffer_dst_offsets = nullptr;
  BufferSizeT *d_buffer_sizes       = nullptr;
  void *d_temp_storage              = nullptr;
  size_t temp_storage_bytes         = 0;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Generate the buffer sizes
  GenerateRandomData(h_buffer_sizes.data(), h_buffer_sizes.size(), min_buffer_size, max_buffer_size);

  // Compute the total bytes to be copied
  ByteOffsetT num_total_bytes = 0;
  for (BufferOffsetT i = 0; i < num_buffers; i++)
  {
    h_buffer_dst_offsets[i] = num_total_bytes;
    num_total_bytes += h_buffer_sizes[i];
  }
  size_t buffers_mem_size = (2 * sizeof(ByteOffsetT) + sizeof(BufferSizeT)) * num_buffers;
  std::cout << "Total of " << num_buffers << " buffers (src_offsets + dst_offsets + sizes -> " << buffers_mem_size
            << " bytes), with a total of " << num_total_bytes
            << " bytes (total memory transfers: " << (buffers_mem_size + 2 * num_total_bytes) << " bytes)\n";

  // Allocate device memory
  CubDebugExit(cudaMalloc(&d_in, data_size));
  CubDebugExit(cudaMalloc(&d_out, num_total_bytes));
  CubDebugExit(cudaMalloc(&d_buffer_src_offsets, num_buffers * sizeof(d_buffer_src_offsets[0])));
  CubDebugExit(cudaMalloc(&d_buffer_dst_offsets, num_buffers * sizeof(d_buffer_dst_offsets[0])));
  CubDebugExit(cudaMalloc(&d_buffer_sizes, num_buffers * sizeof(d_buffer_sizes[0])));

  std::cout << "Device-side src data segment @" << static_cast<void *>(d_in) << " of " << data_size << " bytes\n";

  // Populate the data source with random data
  using RandomInitAliasT      = uint16_t;
  uint32_t num_aliased_factor = sizeof(RandomInitAliasT) / sizeof(uint8_t);
  uint32_t num_aliased_units  = (data_size + num_aliased_factor - 1ULL) / num_aliased_factor;
  uint8_t *h_in               = new uint8_t[num_aliased_units * num_aliased_factor];
  uint8_t *h_out              = new uint8_t[num_total_bytes];
  uint8_t *h_gpu_results      = new uint8_t[num_total_bytes];
  GenerateRandomData(reinterpret_cast<uint16_t *>(h_in), num_aliased_units);

  // Generate random offsets into the random-bits data buffer
  GenerateRandomData(h_buffer_src_offsets.data(),
                     num_buffers,
                     static_cast<ByteOffsetT>(0U),
                     data_size - 1 - max_buffer_size);

  // Data size has to be at least as large as the maximum buffer size
  assert(data_size - 1 - max_buffer_size);

  for (BufferOffsetT i = 0; i < num_buffers; i++)
  {
    if (i < 100)
    {
      std::cout
        << "Buffer #" << i                                                                                         //
        << ": [" << h_buffer_dst_offsets[i] << ", " << (h_buffer_dst_offsets[i] + h_buffer_sizes[i]) << ") "       //
        << " <- [" << h_buffer_src_offsets[i] << ", " << h_buffer_src_offsets[i] + h_buffer_sizes[i] << "] : "     //
        << "[" << static_cast<void *>(static_cast<char *>(d_out) + h_buffer_dst_offsets[i]) << ", "                //
        << static_cast<void *>(static_cast<char *>(d_out) + (h_buffer_dst_offsets[i] + h_buffer_sizes[i])) << ") " //
        << " <- [" << static_cast<void *>(static_cast<char *>(d_in) + h_buffer_src_offsets[i]) << ", "             //
        << static_cast<void *>(static_cast<char *>(d_in) + h_buffer_src_offsets[i] + h_buffer_sizes[i])            //
        << "), size: " << h_buffer_sizes[i] << "\n ";
    }
  }

  // Prepare d_buffer_srcs
  OffsetToPtrOp<SrcPtrT> src_transform_op{static_cast<SrcPtrT>(d_in)};
  cub::TransformInputIterator<SrcPtrT, OffsetToPtrOp<SrcPtrT>, ByteOffsetT *> d_buffer_srcs(d_buffer_src_offsets,
                                                                                            src_transform_op);

  // Prepare d_buffer_dsts
  OffsetToPtrOp<SrcPtrT> dst_transform_op{static_cast<SrcPtrT>(d_out)};
  cub::TransformInputIterator<SrcPtrT, OffsetToPtrOp<SrcPtrT>, ByteOffsetT *> d_buffer_dsts(d_buffer_dst_offsets,
                                                                                            dst_transform_op);

  // Get temporary storage requirements
  CubDebugExit(DeviceBatchMemcpy(d_temp_storage,
                                 temp_storage_bytes,
                                 d_buffer_srcs,
                                 d_buffer_dsts,
                                 d_buffer_sizes,
                                 num_buffers,
                                 stream));

  // Allocate temporary storage
  CubDebugExit(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // Prepare random data segment (which serves for the buffer sources)
  CubDebugExit(cudaMemcpyAsync(d_in, h_in, data_size, cudaMemcpyHostToDevice, stream));

  // Prepare d_buffer_src_offsets
  CubDebugExit(cudaMemcpyAsync(d_buffer_src_offsets,
                               h_buffer_src_offsets.data(),
                               h_buffer_src_offsets.size() * sizeof(h_buffer_src_offsets[0]),
                               cudaMemcpyHostToDevice,
                               stream));

  // Prepare d_buffer_dst_offsets
  CubDebugExit(cudaMemcpyAsync(d_buffer_dst_offsets,
                               h_buffer_dst_offsets.data(),
                               h_buffer_dst_offsets.size() * sizeof(h_buffer_dst_offsets[0]),
                               cudaMemcpyHostToDevice,
                               stream));

  // Prepare d_buffer_sizes
  CubDebugExit(cudaMemcpyAsync(d_buffer_sizes,
                               h_buffer_sizes.data(),
                               h_buffer_sizes.size() * sizeof(h_buffer_sizes[0]),
                               cudaMemcpyHostToDevice,
                               stream));

  // Invoke algorithm
  CubDebugExit(DeviceBatchMemcpy(d_temp_storage,
                                 temp_storage_bytes,
                                 d_buffer_srcs,
                                 d_buffer_dsts,
                                 d_buffer_sizes,
                                 num_buffers,
                                 stream));

  // Copy back the output buffer
  CubDebugExit(cudaMemcpyAsync(h_gpu_results, d_out, num_total_bytes, cudaMemcpyDeviceToHost, stream));

  CubDebugExit(cudaStreamSynchronize(stream));

  // CPU-side result generation for verification
  for (unsigned int i = 0; i < num_buffers; i++)
  {
    std::memcpy(&h_out[h_buffer_dst_offsets[i]], &h_in[h_buffer_src_offsets[i]], h_buffer_sizes[i]);
  }

  for (unsigned int i = 0; i < num_total_bytes; i++)
  {
    if (h_gpu_results[i] != h_out[i])
    {
      std::cout << "Mismatch at index " << i << ", CPU vs. GPU: " << static_cast<uint16_t>(h_gpu_results[i]) << ", "
                << static_cast<uint16_t>(h_out[i]) << "\n";
      break;
    }
    if (i < 1000)
    {
      if (i % 10 == 0)
        std::cout << "\n";
      std::cout << "[" << i << "]:"
                << "(" << static_cast<uint16_t>(h_gpu_results[i]) << ")/"
                << "(" << static_cast<uint16_t>(h_out[i]) << "); ";
    }
  }

  CubDebugExit(cudaFree(d_in));
  CubDebugExit(cudaFree(d_out));
  CubDebugExit(cudaFree(d_buffer_src_offsets));
  CubDebugExit(cudaFree(d_buffer_dst_offsets));
  CubDebugExit(cudaFree(d_buffer_sizes));
  CubDebugExit(cudaFree(d_temp_storage));
}