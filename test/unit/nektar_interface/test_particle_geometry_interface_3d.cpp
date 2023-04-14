#include "nektar_interface/particle_interface.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Driver.h>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace std;
using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::LibUtilities;
using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

TEST(ParticleGeometryInterface, ShapeEnums) {

  const int int_tmp0 = shape_type_to_int(ShapeType::eHexahedron);
  ASSERT_EQ(ShapeType::eHexahedron, int_to_shape_type(int_tmp0));
  const int int_tmp1 = shape_type_to_int(ShapeType::ePrism);
  ASSERT_EQ(ShapeType::ePrism, int_to_shape_type(int_tmp1));
  const int int_tmp2 = shape_type_to_int(ShapeType::ePyramid);
  ASSERT_EQ(ShapeType::ePyramid, int_to_shape_type(int_tmp2));
  const int int_tmp3 = shape_type_to_int(ShapeType::eTetrahedron);
  ASSERT_EQ(ShapeType::eTetrahedron, int_to_shape_type(int_tmp3));
  const int int_tmp4 = shape_type_to_int(ShapeType::eQuadrilateral);
  ASSERT_EQ(ShapeType::eQuadrilateral, int_to_shape_type(int_tmp4));
  const int int_tmp5 = shape_type_to_int(ShapeType::eTriangle);
  ASSERT_EQ(ShapeType::eTriangle, int_to_shape_type(int_tmp5));
  std::array<int, 6> int_values = {int_tmp0, int_tmp1, int_tmp2,
                                   int_tmp3, int_tmp4, int_tmp5};
  for (int vx = 0; vx < 6; vx++) {
    for (int ux = vx + 1; ux < 6; ux++) {
      ASSERT_NE(int_values[vx], int_values[ux]);
    }
  }
}

TEST(ParticleGeometryInterface, Init3D) {

  int size, rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;

  int argc = 3;
  char *argv[3];
  copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);

  // std::filesystem::path source_file = __FILE__;
  // std::filesystem::path source_dir = source_file.parent_path();
  // std::filesystem::path test_resources_dir =
  //     source_dir / "../../test_resources";
  std::filesystem::path conditions_file =
      "/home/js0259/git-ukaea/NESO-workspace/3D/conditions.xml";
  copy_to_cstring(std::string(conditions_file), &argv[1]);
  std::filesystem::path mesh_file =
      "/home/js0259/git-ukaea/NESO-workspace/3D/reference_cube.xml";
  copy_to_cstring(std::string(mesh_file), &argv[2]);

  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);

  // Create MeshGraph.
  graph = SpatialDomains::MeshGraph::Read(session);

  ParticleMeshInterface particle_mesh_interface(graph);

  particle_mesh_interface.free();
  delete[] argv[0];
  delete[] argv[1];
}
