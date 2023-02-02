#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <particle_utility/position_distribution.hpp>
#include <random>
#include <vector>

using namespace NESO;

TEST(ParticleUtility, PositionDistributionSobol) {

  double correct[10] = {0.5,  0.5,   0.75,  0.25,  0.25,
                        0.75, 0.375, 0.375, 0.875, 0.875};
  double extents[2] = {1.0, 1.0};

  auto to_test = sobol_within_extents(5, 2, extents);

  for (int px = 0; px < 5; px++) {
    for (int dx = 0; dx < 2; dx++) {
      const double c = correct[px * 2 + dx];
      const double t = to_test[dx][px];
      ASSERT_NEAR(c, t, 1.0e-14);
    }
  }

  const int offset = 2;
  auto to_test_2 = sobol_within_extents(5, 2, extents, offset);
  for (int px = 0; px < (5 - offset); px++) {
    for (int dx = 0; dx < 2; dx++) {
      const double c = correct[(px + offset) * 2 + dx];
      const double t = to_test_2[dx][px];
      ASSERT_NEAR(c, t, 1.0e-14);
    }
  }

  double extents_2[2] = {3.0, 5.0};
  auto to_test_3 = sobol_within_extents(5, 2, extents_2);

  for (int px = 0; px < 5; px++) {
    for (int dx = 0; dx < 2; dx++) {
      const double c = correct[px * 2 + dx] * extents_2[dx];
      const double t = to_test_3[dx][px];
      ASSERT_NEAR(c, t, 1.0e-14);
    }
  }
}
