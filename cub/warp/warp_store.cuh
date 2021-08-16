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

#include "../warp/warp_exchange.cuh"
#include "../block/block_store.cuh"
#include "../config.cuh"
#include "../util_ptx.cuh"
#include "../util_type.cuh"


CUB_NAMESPACE_BEGIN


enum WarpStoreAlgorithm
{
  WARP_STORE_DIRECT,
  WARP_STORE_STRIPED,
  WARP_STORE_VECTORIZE,
  WARP_STORE_TRANSPOSE
};


template <typename           T,
          int                ITEMS_PER_THREAD,
          WarpStoreAlgorithm ALGORITHM            = WARP_STORE_DIRECT,
          int                LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS,
          int                PTX_ARCH             = CUB_PTX_ARCH>
class WarpStore
{
  static_assert(PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE,
                "LOGICAL_WARP_THREADS must be a power of two");

  constexpr static bool IS_ARCH_WARP = LOGICAL_WARP_THREADS ==
                                       CUB_WARP_THREADS(PTX_ARCH);

private:

  /// Store helper
  template <WarpStoreAlgorithm _POLICY, int DUMMY>
  struct StoreInternal;

  template <int DUMMY>
  struct StoreInternal<WARP_STORE_DIRECT, DUMMY>
  {
    typedef NullType TempStorage;

    int linear_tid;

    __device__ __forceinline__ StoreInternal(TempStorage &/*temp_storage*/,
                                             int linear_tid)
      : linear_tid(linear_tid)
    {}

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                          T (&items)[ITEMS_PER_THREAD])
    {
      StoreDirectBlocked(linear_tid, block_itr, items);
    }

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                          T (&items)[ITEMS_PER_THREAD],
                                          int valid_items)
    {
      StoreDirectBlocked(linear_tid, block_itr, items, valid_items);
    }
  };


  template <int DUMMY>
  struct StoreInternal<WARP_STORE_STRIPED, DUMMY>
  {
    typedef NullType TempStorage;

    int linear_tid;

    __device__ __forceinline__ StoreInternal(TempStorage & /*temp_storage*/,
                                             int linear_tid)
        : linear_tid(linear_tid)
    {}

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                          T (&items)[ITEMS_PER_THREAD])
    {
      StoreDirectStriped<LOGICAL_WARP_THREADS>(linear_tid, block_itr, items);
    }

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                          T (&items)[ITEMS_PER_THREAD],
                                          int valid_items)
    {
      StoreDirectStriped<LOGICAL_WARP_THREADS>(linear_tid,
                                               block_itr,
                                               items,
                                               valid_items);
    }
  };


  template <int DUMMY>
  struct StoreInternal<WARP_STORE_VECTORIZE, DUMMY>
  {
    typedef NullType TempStorage;

    int linear_tid;

    __device__ __forceinline__ StoreInternal(TempStorage & /*temp_storage*/,
                                             int linear_tid)
        : linear_tid(linear_tid)
    {}

    __device__ __forceinline__ void Store(T *block_ptr,
                                          T (&items)[ITEMS_PER_THREAD])
    {
      StoreDirectBlockedVectorized(linear_tid, block_ptr, items);
    }

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                          T (&items)[ITEMS_PER_THREAD])
    {
      StoreDirectBlocked(linear_tid, block_itr, items);
    }

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                          T (&items)[ITEMS_PER_THREAD],
                                          int valid_items)
    {
      StoreDirectBlocked(linear_tid, block_itr, items, valid_items);
    }
  };


  template <int DUMMY>
  struct StoreInternal<WARP_STORE_TRANSPOSE, DUMMY>
  {
    using WarpExchangeT =
      WarpExchange<T, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, PTX_ARCH>;

    struct _TempStorage : WarpExchangeT::TempStorage
    {};

    struct TempStorage : Uninitialized<_TempStorage> {};

    _TempStorage &temp_storage;

    int linear_tid;

    __device__ __forceinline__ StoreInternal(TempStorage &temp_storage,
                                             int linear_tid)
        : temp_storage(temp_storage.Alias())
        , linear_tid(linear_tid)
    {}

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                          T (&items)[ITEMS_PER_THREAD])
    {
      WarpExchangeT(temp_storage).BlockedToStriped(items, items);
      StoreDirectStriped<LOGICAL_WARP_THREADS>(linear_tid, block_itr, items);
    }

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                          T (&items)[ITEMS_PER_THREAD],
                                          int valid_items)
    {
      WarpExchangeT(temp_storage).BlockedToStriped(items, items);
      StoreDirectStriped<LOGICAL_WARP_THREADS>(linear_tid,
                                               block_itr,
                                               items,
                                               valid_items);
    }
  };


  /// Internal load implementation to use
  using InternalStore = StoreInternal<ALGORITHM, 0>;

  /// Shared memory storage layout type
  using _TempStorage = typename InternalStore::TempStorage;


  __device__ __forceinline__ _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }


  _TempStorage &temp_storage;

  int linear_tid;

public:

  struct TempStorage : Uninitialized<_TempStorage> {};

  __device__ __forceinline__ WarpStore()
      : temp_storage(PrivateStorage())
      , linear_tid(IS_ARCH_WARP ? LaneId() : (LaneId() % LOGICAL_WARP_THREADS))
  {}

  __device__ __forceinline__ WarpStore(TempStorage &temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(IS_ARCH_WARP ? LaneId() : (LaneId() % LOGICAL_WARP_THREADS))
  {}

  template <typename OutputIteratorT>
  __device__ __forceinline__ void Store(
    OutputIteratorT     block_itr,
    T                   (&items)[ITEMS_PER_THREAD])
  {
    InternalStore(temp_storage, linear_tid).Store(block_itr, items);
  }

  template <typename OutputIteratorT>
  __device__ __forceinline__ void Store(
    OutputIteratorT     block_itr,
    T                   (&items)[ITEMS_PER_THREAD],
    int                 valid_items)
  {
    InternalStore(temp_storage, linear_tid).Store(block_itr, items, valid_items);
  }
};


CUB_NAMESPACE_END

