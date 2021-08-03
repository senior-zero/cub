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

#include <stdio.h>
#include <iterator>

#include "dispatch_scan.cuh"
#include "../../config.cuh"
#include "../../agent/agent_three_way_partition.cuh"
#include "../../thread/thread_operators.cuh"
#include "../../grid/grid_queue.cuh"
#include "../../util_device.cuh"
#include "../../util_math.cuh"

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

CUB_NAMESPACE_BEGIN


/******************************************************************************
 * Kernel entry points
 *****************************************************************************/


template <
  typename            AgentThreeWayPartitionPolicyT,
  typename            InputIteratorT,
  typename            FlagsInputIteratorT,
  typename            SelectedOutputIteratorT,
  typename            NumSelectedIteratorT,
  typename            ScanTileStateT,
  typename            SelectOp1T,
  typename            SelectOp2T,
  typename            OffsetT>
__launch_bounds__ (int(AgentThreeWayPartitionPolicyT::BLOCK_THREADS))
__global__ void DeviceSelectSweep3Kernel(
  InputIteratorT          d_in,
  FlagsInputIteratorT     d_flags,
  SelectedOutputIteratorT d_selected_out_1,
  SelectedOutputIteratorT d_selected_out_2,
  NumSelectedIteratorT    d_num_selected_out,
  ScanTileStateT          tile_status_1,
  ScanTileStateT          tile_status_2,
  SelectOp1T              select_op_1,
  SelectOp2T              select_op_2,
  OffsetT                 num_items,
  int                     num_tiles)
{
  // Thread block type for selecting data from input tiles
  typedef AgentThreeWayPartition<AgentThreeWayPartitionPolicyT,
                                 InputIteratorT,
                                 FlagsInputIteratorT,
                                 SelectedOutputIteratorT,
                                 SelectOp1T,
                                 SelectOp2T,
                                 OffsetT>
    AgentThreeWayPartitionT;

  // Shared memory for AgentSelectIf
  __shared__ typename AgentThreeWayPartitionT::TempStorage temp_storage;

  // Process tiles
  AgentThreeWayPartitionT(temp_storage,
                 d_in,
                 d_flags,
                 d_selected_out_1,
                 d_selected_out_2,
                 select_op_1,
                 select_op_2,
                 num_items)
    .ConsumeRange(num_tiles, tile_status_1, tile_status_2, d_num_selected_out);
}

/**
 * Initialization kernel for tile status initialization (multi-block)
 */
template <
  typename                ScanTileStateT,         ///< Tile status interface type
  typename                NumSelectedIteratorT>   ///< Output iterator type for recording the number of items selected
__global__ void DeviceCompactInit3Kernel(
  ScanTileStateT          tile_state_1,           ///< [in] Tile status interface
  ScanTileStateT          tile_state_2,           ///< [in] Tile status interface
  int                     num_tiles,              ///< [in] Number of tiles
  NumSelectedIteratorT    d_num_selected_out)     ///< [out] Pointer to the total number of items selected (i.e., length of \p d_selected_out)
{
  // Initialize tile status
  tile_state_1.InitializeStatus(num_tiles);
  tile_state_2.InitializeStatus(num_tiles);

  // Initialize d_num_selected_out
  if (blockIdx.x == 0)
  {
    if (threadIdx.x < 2)
    {
      d_num_selected_out[threadIdx.x] = 0;
    }
  }
}

/******************************************************************************
 * Dispatch
 ******************************************************************************/

template <
  typename    InputIteratorT,
  typename    FlagsInputIteratorT,
  typename    SelectedOutputIteratorT,
  typename    NumSelectedIteratorT,
  typename    SelectOp1T,
  typename    SelectOp2T,
  typename    OffsetT>                        
struct DispatchThreeWayPartitionIf
{
  /******************************************************************************
   * Types and constants
   ******************************************************************************/

  // The output value type
  typedef typename cub::If<(cub::Equals<typename std::iterator_traits<SelectedOutputIteratorT>::value_type, void>::VALUE),  // OutputT =  (if output iterator's value type is void) ?
    typename std::iterator_traits<InputIteratorT>::value_type,                                                  // ... then the input iterator's value type,
    typename std::iterator_traits<SelectedOutputIteratorT>::value_type>::Type OutputT;                          // ... else the output iterator's value type

  // The flag value type
  typedef typename std::iterator_traits<FlagsInputIteratorT>::value_type FlagT;

  enum
  {
    INIT_KERNEL_THREADS = 128,
  };

  // Tile status descriptor interface type
  typedef cub::ScanTileState<OffsetT> ScanTileStateT;


  /******************************************************************************
   * Tuning policies
   ******************************************************************************/

  /// SM35
  struct Policy350
  {
    enum {
      NOMINAL_4B_ITEMS_PER_THREAD = 10,
      ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(OutputT)))),
    };

    typedef cub::AgentThreeWayPartitionPolicy<128,
                                              ITEMS_PER_THREAD,
                                              cub::BLOCK_LOAD_DIRECT,
                                              cub::LOAD_LDG,
                                              cub::BLOCK_SCAN_WARP_SCANS>
      ThreeWayPartitionPolicy;
  };

  /******************************************************************************
   * Tuning policies of current PTX compiler pass
   ******************************************************************************/

  typedef Policy350 PtxPolicy;

  // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
  struct PtxThreeWayPartitionPolicyT : PtxPolicy::ThreeWayPartitionPolicy {};


  /******************************************************************************
   * Utilities
   ******************************************************************************/

  /**
   * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
   */
  template <typename KernelConfig>
  CUB_RUNTIME_FUNCTION __forceinline__
  static void InitConfigs(
    int             ptx_version,
    KernelConfig    &select_if_config)
  {
    if (CUB_IS_DEVICE_CODE)
    {
#if CUB_INCLUDE_DEVICE_CODE
      (void)ptx_version;
      // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
      select_if_config.template Init<PtxThreeWayPartitionPolicyT>();
#endif
    }
    else
    {
#if CUB_INCLUDE_HOST_CODE
      // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version

      // (There's only one policy right now)
      (void)ptx_version;
      select_if_config.template Init<typename Policy350::ThreeWayPartitionPolicy>();
#endif
    }
  }


  /**
   * Kernel kernel dispatch configuration.
   */
  struct KernelConfig
  {
    int block_threads;
    int items_per_thread;
    int tile_items;

    template <typename PolicyT>
    CUB_RUNTIME_FUNCTION __forceinline__
    void Init()
    {
      block_threads       = PolicyT::BLOCK_THREADS;
      items_per_thread    = PolicyT::ITEMS_PER_THREAD;
      tile_items          = block_threads * items_per_thread;
    }
  };


  /******************************************************************************
   * Dispatch entrypoints
   ******************************************************************************/

  /**
   * Internal dispatch routine for computing a device-wide selection using the
   * specified kernel functions.
   */
  template <
    typename                    ScanInitKernelPtrT,             ///< Function type of cub::DeviceScanInitKernel
    typename                    SelectIfKernelPtrT>             ///< Function type of cub::SelectIfKernelPtrT
  CUB_RUNTIME_FUNCTION __forceinline__
  static cudaError_t Dispatch(
    void*                       d_temp_storage,                 ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t&                     temp_storage_bytes,             ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
    InputIteratorT              d_in,                           ///< [in] Pointer to the input sequence of data items
    FlagsInputIteratorT         d_flags,                        ///< [in] Pointer to the input sequence of selection flags (if applicable)
    SelectedOutputIteratorT     d_selected_out_1,               ///< [in] Pointer to the output sequence of selected data items
    SelectedOutputIteratorT     d_selected_out_2,               ///< [in] Pointer to the output sequence of selected data items
    NumSelectedIteratorT        d_num_selected_out,             ///< [in] Pointer to the total number of items selected (i.e., length of \p d_selected_out)
    SelectOp1T                  select_op_1,                    ///< [in] Selection operator
    SelectOp2T                  select_op_2,                    ///< [in] Selection operator
    OffsetT                     num_items,                      ///< [in] Total number of input items (i.e., length of \p d_in)
    cudaStream_t                stream,                         ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    bool                        debug_synchronous,              ///< [in] Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    int                         /*ptx_version*/,                ///< [in] PTX version of dispatch kernels
    ScanInitKernelPtrT          scan_init_kernel,               ///< [in] Kernel function pointer to parameterization of cub::DeviceScanInitKernel
    SelectIfKernelPtrT          select_if_kernel,               ///< [in] Kernel function pointer to parameterization of cub::DeviceSelectSweepKernel
    KernelConfig                select_if_config)               ///< [in] Dispatch parameters that match the policy that \p select_if_kernel was compiled for
  {
    cudaError error = cudaSuccess;

    do
    {
      // Get device ordinal
      int device_ordinal;
      if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

      // Number of input tiles
      int tile_size = select_if_config.block_threads * select_if_config.items_per_thread;
      int num_tiles = static_cast<int>(DivideAndRoundUp(num_items, tile_size));

      // Specify temporary storage allocation requirements
      size_t  allocation_sizes[2];
      if (CubDebug(error = ScanTileStateT::AllocationSize(num_tiles, allocation_sizes[0]))) break;    // bytes needed for tile status descriptors

      allocation_sizes[1] = allocation_sizes[0];

      // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
      void* allocations[2] = {};
      if (CubDebug(error = cub::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage allocation
        break;
      }

      // Construct the tile status interface
      ScanTileStateT tile_status_1;
      ScanTileStateT tile_status_2;

      if (CubDebug(error = tile_status_1.Init(num_tiles, allocations[0], allocation_sizes[0]))) break;
      if (CubDebug(error = tile_status_2.Init(num_tiles, allocations[1], allocation_sizes[1]))) break;

      // Log scan_init_kernel configuration
      int init_grid_size = CUB_MAX(1, DivideAndRoundUp(num_tiles, INIT_KERNEL_THREADS));
      if (debug_synchronous)
      {
        _CubLog("Invoking scan_init_kernel<<<%d, %d, 0, %lld>>>()\n",
                init_grid_size,
                INIT_KERNEL_THREADS,
                (long long)stream);
      }

      // Invoke scan_init_kernel to initialize tile descriptors
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        init_grid_size, INIT_KERNEL_THREADS, 0, stream
      ).doit(scan_init_kernel,
             tile_status_1,
             tile_status_2,
             num_tiles,
             d_num_selected_out);

      // Check for failure to launch
      if (CubDebug(error = cudaPeekAtLastError()))
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      if (debug_synchronous)
      {
        if (CubDebug(error = cub::SyncStream(stream)))
        {
          break;
        }
      }

      // Return if empty problem
      if (num_items == 0)
      {
        break;
      }

      // Get SM occupancy for select_if_kernel
      int range_select_sm_occupancy;
      if (CubDebug(error = MaxSmOccupancy(
        range_select_sm_occupancy,            // out
        select_if_kernel,
        select_if_config.block_threads)))
      {
        break;
      }

      // Get max x-dimension of grid
      int max_dim_x;
      if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal)))
      {
        break;
      }

      // Get grid size for scanning tiles
      dim3 scan_grid_size;
      scan_grid_size.z = 1;
      scan_grid_size.y = DivideAndRoundUp(num_tiles, max_dim_x);
      scan_grid_size.x = CUB_MIN(num_tiles, max_dim_x);

      // Log select_if_kernel configuration
      if (debug_synchronous)
      {
        _CubLog("Invoking select_if_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d "
                "items per thread, %d SM occupancy\n",
                scan_grid_size.x,
                scan_grid_size.y,
                scan_grid_size.z,
                select_if_config.block_threads,
                (long long)stream,
                select_if_config.items_per_thread,
                range_select_sm_occupancy);
      }

      // Invoke select_if_kernel
      thrust::cuda_cub::launcher::triple_chevron(
        scan_grid_size, select_if_config.block_threads, 0, stream
      ).doit(select_if_kernel,
             d_in,
             d_flags,
             d_selected_out_1,
             d_selected_out_2,
             d_num_selected_out,
             tile_status_1,
             tile_status_2,
             select_op_1,
             select_op_2,
             num_items,
             num_tiles);

      // Check for failure to launch
      if (CubDebug(error = cudaPeekAtLastError())) break;

      // Sync the stream if specified to flush runtime errors
      if (debug_synchronous && (CubDebug(error = cub::SyncStream(stream)))) break;
    }
    while (0);

    return error;
  }


  /**
   * Internal dispatch routine
   */
  CUB_RUNTIME_FUNCTION __forceinline__
  static cudaError_t Dispatch(
    void*                       d_temp_storage,                 ///< [in] %Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t&                     temp_storage_bytes,             ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
    InputIteratorT              d_in,                           ///< [in] Pointer to the input sequence of data items
    FlagsInputIteratorT         d_flags,                        ///< [in] Pointer to the input sequence of selection flags (if applicable)
    SelectedOutputIteratorT     d_selected_out_1,               ///< [in] Pointer to the output sequence of selected data items
    SelectedOutputIteratorT     d_selected_out_2,               ///< [in] Pointer to the output sequence of selected data items
    NumSelectedIteratorT        d_num_selected_out,             ///< [in] Pointer to the total number of items selected (i.e., length of \p d_selected_out)
    SelectOp1T                  select_op_1,                    ///< [in] Selection operator
    SelectOp2T                  select_op_2,                    ///< [in] Selection operator
    OffsetT                     num_items,                      ///< [in] Total number of input items (i.e., length of \p d_in)
    cudaStream_t                stream,                         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    bool                        debug_synchronous)              ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
  {
    cudaError error = cudaSuccess;
    do
    {
      // Get PTX version
      int ptx_version = 0;
      if (CubDebug(error = cub::PtxVersion(ptx_version)))
      {
        break;
      }

      // Get kernel kernel dispatch configurations
      KernelConfig select_if_config;
      InitConfigs(ptx_version, select_if_config);

      // Dispatch
      if (CubDebug(
            error = Dispatch(
              d_temp_storage,
              temp_storage_bytes,
              d_in,
              d_flags,
              d_selected_out_1,
              d_selected_out_2,
              d_num_selected_out,
              select_op_1,
              select_op_2,
              num_items,
              stream,
              debug_synchronous,
              ptx_version,
              DeviceCompactInit3Kernel<ScanTileStateT, NumSelectedIteratorT>,
              DeviceSelectSweep3Kernel<PtxThreeWayPartitionPolicyT,
                                       InputIteratorT,
                                       FlagsInputIteratorT,
                                       SelectedOutputIteratorT,
                                       NumSelectedIteratorT,
                                       ScanTileStateT,
                                       SelectOp1T,
                                       SelectOp2T,
                                       OffsetT>,
              select_if_config)))
      {
        break;
      }
    } while (0);

    return error;
  }
};


CUB_NAMESPACE_END
