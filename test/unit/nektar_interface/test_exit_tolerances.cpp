#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Driver.h>
#include <SpatialDomains/MeshGraphIO.h>
#include <gtest/gtest.h>

#include "nektar_interface/particle_cell_mapping/newton_relative_exit_tolerances.hpp"
#include "nektar_interface/utility_mesh.hpp"

using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;
using namespace NESO;

TEST(NewtonMethod, ExitTolerances) {

  auto xmapx = [&](auto eta) { return eta[0] * 2.0; };
  auto xmapy = [&](auto eta) { return eta[1] * 3.0; };
  auto xmapz = [&](auto eta) { return eta[2] * 5.0; };

  const int num_modes = 3;
  auto h = make_hex_geom(num_modes, xmapx, xmapy, xmapz);

  NewtonRelativeExitTolerances newton_relative_exit_tolerances;
  create_newton_relative_exit_tolerances(h, &newton_relative_exit_tolerances);

  auto m_xmap = h->GetXmap();
  auto m_geomFactors = h->GetGeomFactors();
  Array<OneD, const NekDouble> Jac =
      m_geomFactors->GetJac(m_xmap->GetPointsKeys());
  NekDouble tol_scaling =
      Vmath::Vsum(Jac.size(), Jac, 1) / ((NekDouble)Jac.size());
  const auto scaling_jacobian = static_cast<REAL>(ABS(1.0 / tol_scaling));

  ASSERT_NEAR(scaling_jacobian,
              newton_relative_exit_tolerances.scaling_jacobian, 1.0e-14);

  ASSERT_TRUE(1.0 / 2.0 >= newton_relative_exit_tolerances.scaling_dir[0]);
  ASSERT_TRUE(1.0 / 3.0 >= newton_relative_exit_tolerances.scaling_dir[1]);
  ASSERT_TRUE(1.0 / 5.0 >= newton_relative_exit_tolerances.scaling_dir[2]);
  ASSERT_TRUE(0.0 < newton_relative_exit_tolerances.scaling_dir[0]);
  ASSERT_TRUE(0.0 < newton_relative_exit_tolerances.scaling_dir[1]);
  ASSERT_TRUE(0.0 < newton_relative_exit_tolerances.scaling_dir[2]);
  ASSERT_TRUE(newton_relative_exit_tolerances.scaling_dir[0] >
              newton_relative_exit_tolerances.scaling_dir[1]);
  ASSERT_TRUE(newton_relative_exit_tolerances.scaling_dir[1] >
              newton_relative_exit_tolerances.scaling_dir[2]);
  ASSERT_TRUE(newton_relative_exit_tolerances.scaling_dir[0] >
              newton_relative_exit_tolerances.scaling_dir[2]);
}
