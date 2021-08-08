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

#include <iterator>
#include <type_traits>

#include "../iterator/cache_modified_input_iterator.cuh"
#include "../warp/warp_exchange.cuh"
#include "../block/block_load.cuh"
#include "../config.cuh"
#include "../util_ptx.cuh"
#include "../util_type.cuh"

CUB_NAMESPACE_BEGIN

enum WarpLoadAlgorithm
{
  WARP_LOAD_DIRECT,
  WARP_LOAD_STRIPED,
  WARP_LOAD_VECTORIZE,
  WARP_LOAD_TRANSPOSE
};

template <typename          InputT,
          int               ITEMS_PER_THREAD,
          WarpLoadAlgorithm ALGORITHM            = WARP_LOAD_DIRECT,
          int               LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS,
          int               PTX_ARCH             = CUB_PTX_ARCH>
class WarpLoad
{
  constexpr static bool IS_ARCH_WARP = LOGICAL_WARP_THREADS ==
                                       CUB_WARP_THREADS(PTX_ARCH);

private:

  /******************************************************************************
   * Algorithmic variants
   ******************************************************************************/

  /// Load helper
  template <WarpLoadAlgorithm _POLICY, int DUMMY>
  struct LoadInternal;


  /**
   * BLOCK_LOAD_DIRECT specialization of load helper
   */
  template <int DUMMY>
  struct LoadInternal<WARP_LOAD_DIRECT, DUMMY>
  {
    /// Shared memory storage layout type
    typedef NullType TempStorage;

    /// Linear thread-id
    int linear_tid;

    /// Constructor
    __device__ __forceinline__ LoadInternal(
      TempStorage &/*temp_storage*/,
      int linear_tid)
      :
      linear_tid(linear_tid)
    {}

    /// Load a linear segment of items from memory
    template <typename InputIteratorT>
    __device__ __forceinline__ void Load(
      InputIteratorT  block_itr,                      ///< [in] The thread block's base input iterator for loading from
      InputT          (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
    {
      LoadDirectBlocked(linear_tid, block_itr, items);
    }

    /// Load a linear segment of items from memory, guarded by range
    template <typename InputIteratorT>
    __device__ __forceinline__ void Load(
      InputIteratorT  block_itr,                      ///< [in] The thread block's base input iterator for loading from
      InputT          (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
      int             valid_items)                    ///< [in] Number of valid items to load
    {
      LoadDirectBlocked(linear_tid, block_itr, items, valid_items);
    }

    /// Load a linear segment of items from memory, guarded by range, with a fall-back assignment of out-of-bound elements
    template <typename InputIteratorT, typename DefaultT>
    __device__ __forceinline__ void Load(
      InputIteratorT  block_itr,                      ///< [in] The thread block's base input iterator for loading from
      InputT          (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
      int             valid_items,                    ///< [in] Number of valid items to load
      DefaultT        oob_default)                    ///< [in] Default value to assign out-of-bound items
    {
      LoadDirectBlocked(linear_tid, block_itr, items, valid_items, oob_default);
    }

  };


  /**
  * BLOCK_LOAD_STRIPED specialization of load helper
  */
  template <int DUMMY>
  struct LoadInternal<WARP_LOAD_STRIPED, DUMMY>
  {
    /// Shared memory storage layout type
    typedef NullType TempStorage;

    /// Linear thread-id
    int linear_tid;

    /// Constructor
    __device__ __forceinline__
    LoadInternal(TempStorage & /*temp_storage*/,
                 int linear_tid)
        : linear_tid(linear_tid)
    {}

    /// Load a linear segment of items from memory
    template <typename InputIteratorT>
    __device__ __forceinline__ void Load(
      InputIteratorT  block_itr,                      ///< [in] The thread block's base input iterator for loading from
      InputT          (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load{
    {
      LoadDirectStriped<LOGICAL_WARP_THREADS>(linear_tid, block_itr, items);
    }

    /// Load a linear segment of items from memory, guarded by range
    template <typename InputIteratorT>
    __device__ __forceinline__ void Load(
      InputIteratorT  block_itr,                      ///< [in] The thread block's base input iterator for loading from
      InputT          (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
      int             valid_items)                    ///< [in] Number of valid items to load
    {
      LoadDirectStriped<LOGICAL_WARP_THREADS>(linear_tid,
                                              block_itr,
                                              items,
                                              valid_items);
    }

    /// Load a linear segment of items from memory, guarded by range, with a fall-back assignment of out-of-bound elements
    template <typename InputIteratorT, typename DefaultT>
    __device__ __forceinline__ void Load(
      InputIteratorT  block_itr,                      ///< [in] The thread block's base input iterator for loading from
      InputT          (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
      int             valid_items,                    ///< [in] Number of valid items to load
      DefaultT        oob_default)                    ///< [in] Default value to assign out-of-bound items
    {
      LoadDirectStriped<LOGICAL_WARP_THREADS>(linear_tid,
                                              block_itr,
                                              items,
                                              valid_items,
                                              oob_default);
    }

  };


  /**
   * BLOCK_LOAD_VECTORIZE specialization of load helper
   */
  template <int DUMMY>
  struct LoadInternal<WARP_LOAD_VECTORIZE, DUMMY>
  {
    /// Shared memory storage layout type
    typedef NullType TempStorage;

    /// Linear thread-id
    int linear_tid;

    /// Constructor
    __device__ __forceinline__
    LoadInternal(TempStorage &/*temp_storage*/,
                 int linear_tid)
      : linear_tid(linear_tid)
    {}

    template <typename InputIteratorT>
    __device__ __forceinline__ void Load(
      InputT               *block_ptr,                     ///< [in] The thread block's base input iterator for loading from
      InputT               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
    {
      InternalLoadDirectBlockedVectorized<LOAD_DEFAULT>(linear_tid,
                                                        block_ptr,
                                                        items);
    }

    template <typename InputIteratorT>
    __device__ __forceinline__ void Load(
      const InputT         *block_ptr,                     ///< [in] The thread block's base input iterator for loading from
      InputT               (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load
    {
      InternalLoadDirectBlockedVectorized<LOAD_DEFAULT>(linear_tid,
                                                        block_ptr,
                                                        items);
    }

    /// Load a linear segment of items from memory, specialized for native pointer types (attempts vectorization)
    template <
      CacheLoadModifier   MODIFIER,
      typename            ValueType,
      typename            OffsetT>
    __device__ __forceinline__ void Load(
      CacheModifiedInputIterator<MODIFIER, ValueType, OffsetT>    block_itr,
      InputT                                                     (&items)[ITEMS_PER_THREAD])
    {
      InternalLoadDirectBlockedVectorized<MODIFIER>(linear_tid,
                                                    block_itr.ptr,
                                                    items);
    }

    template <typename _InputIteratorT>
    __device__ __forceinline__ void Load(
      _InputIteratorT   block_itr,
      InputT           (&items)[ITEMS_PER_THREAD])
    {
      LoadDirectBlocked(linear_tid, block_itr, items);
    }

    template <typename InputIteratorT>
    __device__ __forceinline__ void Load(
      InputIteratorT  block_itr,
      InputT          (&items)[ITEMS_PER_THREAD],
      int             valid_items)
    {
      LoadDirectBlocked(linear_tid, block_itr, items, valid_items);
    }

    template <typename InputIteratorT, typename DefaultT>
    __device__ __forceinline__ void Load(
      InputIteratorT  block_itr,
      InputT          (&items)[ITEMS_PER_THREAD],
      int             valid_items,
      DefaultT        oob_default)
    {
      LoadDirectBlocked(linear_tid, block_itr, items, valid_items, oob_default);
    }

  };


  /**
   * BLOCK_LOAD_TRANSPOSE specialization of load helper
   */
  template <int DUMMY>
  struct LoadInternal<WARP_LOAD_TRANSPOSE, DUMMY>
  {
    // BlockExchange utility type for keys
    using WarpExchangeT =
      WarpExchange<InputT, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, PTX_ARCH>;

    /// Shared memory storage layout type
    struct _TempStorage : WarpExchangeT::TempStorage
    {};

    /// Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};

    /// Thread reference to shared storage
    _TempStorage &temp_storage;

    /// Linear thread-id
    int linear_tid;

    /// Constructor
    __device__ __forceinline__ LoadInternal(
      TempStorage &temp_storage,
      int linear_tid)
      :
      temp_storage(temp_storage.Alias()),
      linear_tid(linear_tid)
    {}

    template <typename InputIteratorT>
    __device__ __forceinline__ void Load(
      InputIteratorT  block_itr,
      InputT          (&items)[ITEMS_PER_THREAD])
    {
      LoadDirectStriped<LOGICAL_WARP_THREADS>(linear_tid, block_itr, items);
      WarpExchangeT(temp_storage).StripedToBlocked(items, items);
    }

    template <typename InputIteratorT>
    __device__ __forceinline__ void Load(
      InputIteratorT  block_itr,
      InputT          (&items)[ITEMS_PER_THREAD],
      int             valid_items)
    {
      LoadDirectStriped<LOGICAL_WARP_THREADS>(linear_tid, block_itr, items, valid_items);
      WarpExchangeT(temp_storage).StripedToBlocked(items, items);
    }

    template <typename InputIteratorT, typename DefaultT>
    __device__ __forceinline__ void Load(
      InputIteratorT  block_itr,                      ///< [in] The thread block's base input iterator for loading from
      InputT          (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
      int             valid_items,                    ///< [in] Number of valid items to load
      DefaultT        oob_default)                    ///< [in] Default value to assign out-of-bound items
    {
      LoadDirectStriped<LOGICAL_WARP_THREADS>(linear_tid, block_itr, items, valid_items, oob_default);
      WarpExchangeT(temp_storage).StripedToBlocked(items, items);
    }

  };

  /******************************************************************************
   * Type definitions
   ******************************************************************************/

  /// Internal load implementation to use
  using InternalLoad = LoadInternal<ALGORITHM, 0>;

  /// Shared memory storage layout type
  using _TempStorage = typename InternalLoad::TempStorage;


  /******************************************************************************
   * Utility methods
   ******************************************************************************/

  /// Internal storage allocator
  __device__ __forceinline__ _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }


  /******************************************************************************
   * Thread fields
   ******************************************************************************/

  /// Thread reference to shared storage
  _TempStorage &temp_storage;

  /// Linear thread-id
  int linear_tid;

public:

  /// \smemstorage{WarpLoad}
  struct TempStorage : Uninitialized<_TempStorage> {};

  __device__ __forceinline__ WarpLoad()
      : temp_storage(PrivateStorage())
      , linear_tid(IS_ARCH_WARP ? LaneId() : (LaneId() % LOGICAL_WARP_THREADS))
  {}

  __device__ __forceinline__ WarpLoad(TempStorage &temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(IS_ARCH_WARP ? LaneId() : (LaneId() % LOGICAL_WARP_THREADS))
  {}

  template <typename InputIteratorT>
  __device__ __forceinline__ void Load(
    InputIteratorT  block_itr,                  ///< [in] The thread block's base input iterator for loading from
    InputT          (&items)[ITEMS_PER_THREAD]) ///< [out] Data to load
  {
    InternalLoad(temp_storage, linear_tid).Load(block_itr, items);
  }

  template <typename InputIteratorT>
  __device__ __forceinline__ void Load(
    InputIteratorT  block_itr,                  ///< [in] The thread block's base input iterator for loading from
    InputT          (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
    int             valid_items)                ///< [in] Number of valid items to load
  {
    InternalLoad(temp_storage, linear_tid).Load(block_itr, items, valid_items);
  }


  template <typename InputIteratorT,
            typename DefaultT>
  __device__ __forceinline__ void Load(
    InputIteratorT  block_itr,                  ///< [in] The thread block's base input iterator for loading from
    InputT          (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
    int             valid_items,                ///< [in] Number of valid items to load
    DefaultT        oob_default)                ///< [in] Default value to assign out-of-bound items
  {
    InternalLoad(temp_storage, linear_tid).Load(block_itr, items, valid_items, oob_default);
  }

};

CUB_NAMESPACE_END