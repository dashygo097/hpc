#include <gtest/gtest.h>

TEST(SanityCheck, TrueIsTrue) {
  EXPECT_TRUE(true);
  std::cout << "Sanity check passed: true is true." << std::endl;
}
