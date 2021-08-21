#include <gtest/gtest.h>

// Value parameter
enum class MyType
{
  MY_FOO = 0,
  MY_BAR = 1
};

class MyTestSuite
    : public testing::TestWithParam<std::tuple<MyType, std::string>>
{};

INSTANTIATE_TEST_SUITE_P(
  MyGroup,

  MyTestSuite,

  testing::Combine(testing::Values(MyType::MY_FOO, MyType::MY_BAR),
                   testing::Values("A", "B")),

  [](const testing::TestParamInfo<MyTestSuite::ParamType> &info) {
    std::string name = std::get<0>(info.param) == MyType::MY_FOO ? "Foo" : "Bar";
    return name;
  });


// Type parameter
struct A1 {
  char ch = 'A';
};

struct A2 {
  char ch = 'a';
};

struct B1 {
  char ch = 'B';
};

struct B2 {
  char ch = 'b';
};


template <typename T>
class pair_test : public ::testing::Test {};

using test_types = ::testing::Types<std::pair<A1,A2>, std::pair<B1,B2>>;
TYPED_TEST_SUITE(pair_test, test_types);

TYPED_TEST(pair_test, compare_no_case)
{
  typename TypeParam::first_type param1;
  typename TypeParam::second_type param2;
  ASSERT_TRUE(param1.ch == std::toupper(param2.ch));
}
