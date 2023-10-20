#include <gtest/gtest.h>

#include "H3LAPD.hpp"
#include "test_H3LAPD.h"
/**
 * Tests for the H3LAPD solver.
 */

/**
 * N.B. HWTests look for test-specific resources in
 * ./HWTest::get_solver_name()/test_name
 */
TEST_F(HWTest, 2Din3DHWGrowthRates) { check_growth_rates(); }

TEST_F(HWTest, Coupled2Din3DHWMassCons) { check_mass_cons(); }