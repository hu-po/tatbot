#include <gtest/gtest.h>

#include "tatbot_hardware/names.hpp"

TEST(Names, SafetyInterfacesAreUnique)
{
  using tatbot_hardware::names::kSafetyState;
  for (size_t i = 0; i < kSafetyState.size(); ++i) {
    for (size_t j = i + 1; j < kSafetyState.size(); ++j) {
      EXPECT_NE(kSafetyState[i], kSafetyState[j]);
    }
  }
  EXPECT_EQ(tatbot_hardware::names::kLatchStepRefused, 10);
}
