#include <gtest/gtest.h>

TEST(SanityCheck, TrueIsTrue) { EXPECT_TRUE(true); }

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
