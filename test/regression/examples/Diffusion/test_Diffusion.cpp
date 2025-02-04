#include "Diffusion.hpp"
#include <gtest/gtest.h>

#include "../SolverRegTest.hpp"

namespace NESO::Solvers {

class DiffusionRegTest : public SolverRegTest {};

/**
 * Regression tests for the Diffusion solver examples. Note that the test name
 * itself is used to determine the location of the corresponding example
 * directory, config file and mesh file.
 */
TEST_F(DiffusionRegTest, unsteady_aniso) { run_and_regress(); }

} // namespace NESO::Solvers