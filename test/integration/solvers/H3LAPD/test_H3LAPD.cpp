#include <gtest/gtest.h>

#include "H3LAPD.hpp"
#include "test_H3LAPD.h"

/**
 * Tests for H3LAPD solver. Note that HWTest::get_solver_name(), together with
 * the test name is used to determine the locations of the config file, mesh and
 * initial conditions.
 */

TEST_F(HWTest, 2Din3DHWGrowthRates) { check_growth_rates(); }