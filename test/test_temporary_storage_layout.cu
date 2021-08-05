#include "cub/util_device.cuh"
#include "test_util.h"

#include <memory>

template <int Items>
std::size_t GetTemporaryStorageSize(std::size_t (&sizes)[Items])
{
  void *pointers[Items]{};
  std::size_t temp_storage_bytes{};
  cub::AliasTemporaries(nullptr, temp_storage_bytes, pointers, sizes);
  return temp_storage_bytes;
}

std::size_t GetActualZero()
{
  std::size_t sizes[1] {};

  return GetTemporaryStorageSize(sizes);
}

template <int StorageSlots>
void TestEmptyStorage()
{
  cub::TemporaryStorageLayout<StorageSlots> temporary_storage;

  std::size_t temp_storage_bytes {};
  temporary_storage.MapRequirements(nullptr, temp_storage_bytes);

  AssertEquals(temp_storage_bytes, GetActualZero());
}

template <int StorageSlots>
void TestPartiallyFilledStorage()
{
  using target_type = std::uint64_t;
  constexpr std::size_t target_elements = 42;
  constexpr std::size_t full_slot_elements = target_elements * sizeof(target_type);
  constexpr std::size_t empty_slot_elements {};

  cub::TemporaryStorageLayout<StorageSlots> temporary_storage;

  std::unique_ptr<cub::TemporaryStorageArray<target_type>> arrays[StorageSlots];
  std::size_t sizes[StorageSlots] {};

  for (int slot_id = 0; slot_id < StorageSlots; slot_id++)
  {
    auto slot = temporary_storage.NextSlot();

    const std::size_t elements = slot_id % 2 == 0
                               ? full_slot_elements
                               : empty_slot_elements;
    sizes[slot_id] = elements * sizeof(target_type);
    arrays[slot_id].reset(new cub::TemporaryStorageArray<target_type>(
      slot->template GetAlias<target_type>(elements)));
  }

  std::size_t temp_storage_bytes {};
  temporary_storage.MapRequirements(nullptr, temp_storage_bytes);

  std::unique_ptr<std::uint8_t[]> temp_storage(
    new std::uint8_t[temp_storage_bytes]);

  temporary_storage.MapRequirements(temp_storage.get(),
                                    temp_storage_bytes);

  AssertEquals(temp_storage_bytes, GetTemporaryStorageSize(sizes));

  for (int slot_id = 0; slot_id < StorageSlots; slot_id++)
  {
    if (slot_id % 2 == 0)
    {
      AssertTrue(arrays[slot_id]->Get() != nullptr);
    }
    else
    {
      AssertTrue(arrays[slot_id]->Get() == nullptr);
    }
  }
}

template <int StorageSlots>
void Test()
{
  TestEmptyStorage<StorageSlots>();
  TestPartiallyFilledStorage<StorageSlots>();
}

int main()
{
  Test<1>();
  Test<4>();
  Test<42>();
}