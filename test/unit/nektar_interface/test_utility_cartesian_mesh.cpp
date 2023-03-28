#include "nektar_interface/utility_mesh_cartesian.hpp"
#include <gtest/gtest.h>
#include <neso_particles.hpp>

#include <array>
#include <vector>

using namespace NESO;
using namespace NESO::Particles;

TEST(DeviceCartesianMesh, Base) {

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  const int ndim = 2;
  std::vector<double> origin = {-1.0, -2.0};
  std::vector<double> extents = {4.0, 6.0};
  std::vector<int> cell_counts = {4, 3};

  DeviceCartesianMesh mesh{sycl_target, ndim, origin, extents, cell_counts};

  for (int dimx = 0; dimx < ndim; dimx++) {
    ASSERT_EQ(mesh.dh_origin->h_buffer.ptr[dimx], origin[dimx]);
    ASSERT_EQ(mesh.dh_extents->h_buffer.ptr[dimx], extents[dimx]);
    ASSERT_EQ(mesh.dh_cell_counts->h_buffer.ptr[dimx], cell_counts[dimx]);

    const double extent = extents[dimx];
    const double cell_countf = cell_counts[dimx];
    const double cell_width = extent / cell_countf;
    const double inverse_cell_width = 1.0 / cell_width;

    ASSERT_NEAR(cell_width, mesh.dh_cell_widths->h_buffer.ptr[dimx], 1.0e-10);
    ASSERT_NEAR(inverse_cell_width,
                mesh.dh_inverse_cell_widths->h_buffer.ptr[dimx], 1.0e-10);
  }

  std::array<double, 6> bb;
  std::array<int, 2> cell = {1, 1};
  mesh.get_bounding_box(cell, bb);

  ASSERT_NEAR(bb[0], 0.0, 1.0e-10);
  ASSERT_NEAR(bb[1], 0.0, 1.0e-10);
  ASSERT_NEAR(bb[0 + 3], 1.0, 1.0e-10);
  ASSERT_NEAR(bb[1 + 3], 2.0, 1.0e-10);

  const int linear_cell = mesh.get_linear_cell_index(cell);
  ASSERT_EQ(linear_cell, 1 + 4 * 1);

  EXPECT_EQ(0, mesh.get_cell_in_dimension(0, -0.5));
  EXPECT_EQ(0, mesh.get_cell_in_dimension(1, -0.5));
  EXPECT_EQ(1, mesh.get_cell_in_dimension(0, 0.0001));
  EXPECT_EQ(3, mesh.get_cell_in_dimension(0, 2.99999));
  EXPECT_EQ(2, mesh.get_cell_in_dimension(1, 3.99999));
  EXPECT_EQ(12, mesh.get_cell_count());
}
