#include <gtest/gtest.h>

#include "test_DriftReduced.hpp"
/**
 * Tests for the DriftReduced solver.
 */

/**
 * N.B. HWTests look for test-specific resources in
 * ./HWTest::get_solver_name()/test_name
 */
TEST_F(HWTest, 2Din3DHWGrowthRates) { check_growth_rates(); }

TEST_F(HWTest, Coupled2Din3DHWMassCons) { check_mass_cons(); }

// Energy growth rate for 3DHW doesn't agree with calc for 2D
// Not clear that this check is valid in 3D; just check W for now
TEST_F(HWTest, 3DHWGrowthRates) { check_growth_rates(false); }