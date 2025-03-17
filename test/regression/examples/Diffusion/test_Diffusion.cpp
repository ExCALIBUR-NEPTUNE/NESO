#include "Diffusion.hpp"
#include <gtest/gtest.h>

#include "../SolverRegTest.hpp"

namespace NESO::Solvers {

/**
 * @brief Regression tests for the Diffusion solver examples. Note that test
 * names are used to determine the location of the corresponding example
 * directories and hence the config and mesh files.
 */
class DiffusionRegTest : public SolverRegTest {};

TEST_F(DiffusionRegTest, unsteady_aniso) { run_and_regress(); }

} // namespace NESO::Solvers