#include <iostream>
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

unsigned int Factorial( unsigned int number )
{
  return number <= 1 ? number : Factorial(number-1)*number;
}

TEST_CASE( "Factorials are computed", "[factorial]" )
{
  REQUIRE( Factorial(1) == 1 );
  REQUIRE( Factorial(2) == 2 );
  REQUIRE( Factorial(3) == 6 );
  REQUIRE( Factorial(10) == 3628800 );
}

int main(int argc, char *argv[])
{
  // global setup...

  int result = Catch::Session().run(argc, argv);

  // global clean-up...

  return result;
}