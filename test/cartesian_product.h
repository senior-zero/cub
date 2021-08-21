//
// Created by evtus on 8/22/2021.
//

#ifndef CUB_CARTESIAN_PRODUCT_H
#define CUB_CARTESIAN_PRODUCT_H

#include <cstdint>
#include <tuple>

namespace nvbench
{

template <typename... Ts>
struct type_list
{};

template <typename T>
struct wrapped_type
{
  using type = T;
};

namespace detail
{

template <typename... Ts>
auto size(nvbench::type_list<Ts...>)
-> std::integral_constant<std::size_t, sizeof...(Ts)>;

template <std::size_t I, typename... Ts>
auto get(nvbench::type_list<Ts...>)
-> typename std::tuple_element<I, std::tuple<Ts...>>::type;

template <typename... Ts, typename... Us>
auto concat(nvbench::type_list<Ts...>, nvbench::type_list<Us...>)
-> nvbench::type_list<Ts..., Us...>;

//------------------------------------------------------------------------------
template <typename T, typename TLs>
struct prepend_each;

template <typename T>
struct prepend_each<T, nvbench::type_list<>>
{
  using type = nvbench::type_list<>;
};

template <typename T, typename TL, typename... TLTail>
struct prepend_each<T, nvbench::type_list<TL, TLTail...>>
{
  using cur = decltype(detail::concat(nvbench::type_list<T>{}, TL{}));
  using next =
  typename detail::prepend_each<T, nvbench::type_list<TLTail...>>::type;
  using type = decltype(detail::concat(nvbench::type_list<cur>{}, next{}));
};

//------------------------------------------------------------------------------
template <typename TLs>
struct cartesian_product;

template <>
struct cartesian_product<nvbench::type_list<>>
{ // If no input type_lists are provided, there's just one output --
  // a null type_list:
  using type = nvbench::type_list<nvbench::type_list<>>;
};

template <typename... TLTail>
struct cartesian_product<nvbench::type_list<nvbench::type_list<>, TLTail...>>
{ // This is a recursion base case -- in practice empty type_lists should
  // not be passed into cartesian_product.
  using type = nvbench::type_list<>;
};

template <typename T, typename... Ts>
struct cartesian_product<nvbench::type_list<nvbench::type_list<T, Ts...>>>
{
  using cur = nvbench::type_list<nvbench::type_list<T>>;
  using next = typename
  std::conditional<sizeof...(Ts) != 0,
    typename detail::cartesian_product<
      nvbench::type_list<nvbench::type_list<Ts...>>>::type,
    nvbench::type_list<>>::type;
  using type = decltype(detail::concat(cur{}, next{}));
};

template <typename T, typename... Tail, typename TL, typename... TLTail>
struct cartesian_product<
  nvbench::type_list<nvbench::type_list<T, Tail...>, TL, TLTail...>>
{
  using tail_prod =
  typename detail::cartesian_product<nvbench::type_list<TL, TLTail...>>::type;
  using cur  = typename detail::prepend_each<T, tail_prod>::type;
  using next = typename detail::cartesian_product<
    nvbench::type_list<nvbench::type_list<Tail...>, TL, TLTail...>>::type;
  using type = decltype(detail::concat(cur{}, next{}));
};


} // namespace detail
} // namespace nvbench

#include <tuple>
#include <type_traits>

namespace nvbench
{

template <typename... Ts>
struct type_list;

// Wraps a type for use with nvbench::foreach.
template <typename T>
struct wrapped_type;

/**
 * Get the size of a type_list as a `std::integral_constant<size_t, N>`.
 *
 * ```c++
 * using TL = nvbench::type_list<T0, T1, T2, T3, T4>;
 * static_assert(nvbench::tl::size<TL>::value == 5);
 * ```
 */
template <typename TypeList>
using size = decltype(detail::size(TypeList{}));

/**
 * Get the type at the specified index of a type_list.
 *
 * ```c++
 * using TL = nvbench::type_list<T0, T1, T2, T3, T4>;
 * static_assert(std::is_same_v<nvbench::tl::get<0, TL>, T0>);
 * static_assert(std::is_same_v<nvbench::tl::get<1, TL>, T1>);
 * static_assert(std::is_same_v<nvbench::tl::get<2, TL>, T2>);
 * static_assert(std::is_same_v<nvbench::tl::get<3, TL>, T3>);
 * static_assert(std::is_same_v<nvbench::tl::get<4, TL>, T4>);
 * ```
 */
template <std::size_t Index, typename TypeList>
using get = decltype(detail::get<Index>(TypeList{}));

/**
 * Concatenate two type_lists.
 *
 * ```c++
 * using TL01 = nvbench::type_list<T0, T1>;
 * using TL23 = nvbench::type_list<T2, T3>;
 * using TL0123 = nvbench::type_list<T0, T1, T2, T3>;
 * static_assert(std::is_same_v<nvbench::tl::concat<TL01, TL23>, T0123>);
 * ```
 */
template <typename TypeList1, typename TypeList2>
using concat = decltype(detail::concat(TypeList1{}, TypeList2{}));

/**
 * Given a type `T` and a type_list of type_lists `TypeLists`, create
 * a new type_list containing each entry from TypeLists prepended with T.
 *
 * ```c++
 *  using TypeLists = type_list<type_list<T0, T1>,
 *                              type_list<T2, T3>>;
 *  using Result = nvbench::tl::prepend_each<T, TypeLists>;
 *  using Reference = type_list<type_list<T, T0, T1>,
 *                              type_list<T, T2, T3>>;
 *  static_assert(std::is_same_v<Result, Reference>);
 * ```
 */
template <typename T, typename TypeLists>
using prepend_each = typename detail::prepend_each<T, TypeLists>::type;

/**
 * Given a type_list of type_lists, compute the cartesian product across all
 * nested type_lists. Supports arbitrary numbers and sizes of nested type_lists.
 *
 * Beware that the result grows very quickly in size.
 *
 * ```c++
 * using T01 = type_list<T0, T1>;
 * using U012 = type_list<U0, U1, U2>;
 * using V01 = type_list<V0, V1>;
 * using TLs = type_list<T01, U012, V01>;
 * using CartProd = type_list<type_list<T0, U0, V0>,
 *                            type_list<T0, U0, V1>,
 *                            type_list<T0, U1, V0>,
 *                            type_list<T0, U1, V1>,
 *                            type_list<T0, U2, V0>,
 *                            type_list<T0, U2, V1>,
 *                            type_list<T1, U0, V0>,
 *                            type_list<T1, U0, V1>,
 *                            type_list<T1, U1, V0>,
 *                            type_list<T1, U1, V1>,
 *                            type_list<T1, U2, V0>,
 *                            type_list<T1, U2, V1>>;
 *  static_assert(std::is_same_v<bench::tl::cartesian_product<TLs>, CartProd>);
 * ```
 */
template <typename TypeLists>
using cartesian_product = typename detail::cartesian_product<TypeLists>::type;

template <typename TypeList, std::size_t>
struct Fold;

template <typename TypeList>
struct Fold<TypeList, 0>
{
  template <typename ActionType>
  void operator()(ActionType action)
  {
    action(wrapped_type<decltype(detail::get<0>(TypeList{}))>{});
  }
};

template <typename TypeList, std::size_t Index>
struct Fold
{
  template <typename ActionType>
  void operator()(ActionType action)
  {
    action(wrapped_type<decltype(detail::get<Index>(TypeList{}))>{});
    Fold<TypeList, Index - 1>{}(action);
  }
};

template <typename TypeList, typename Functor>
void foreach (Functor &&f)
{
  constexpr std::size_t list_size = decltype(detail::size(TypeList{}))::value;

  Fold<TypeList, list_size - 1>{}(f);
}

} // namespace nvbench


#endif // CUB_CARTESIAN_PRODUCT_H
