#include <gtest/gtest.h>

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