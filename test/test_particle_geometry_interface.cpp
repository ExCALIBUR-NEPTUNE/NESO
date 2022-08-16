#include "nektar_interface/particle_interface.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Driver.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

using namespace std;
using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::SpatialDomains;

TEST(ParticleGeometryInterface, Init2D) {

  int size, rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;

  int argc = 2;
  char *argv[2] = {"test_particle_geometry_interface",
                   "test/test_resources/unit_square_0_5.xml"};

  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);

  // Create MeshGraph.
  graph = SpatialDomains::MeshGraph::Read(session);

  ParticleMeshInterface particle_mesh_interface(graph);

  ASSERT_EQ(particle_mesh_interface.ndim, 2);

  for (int dx = 0; dx < 6; dx++) {
    std::cout << dx << " " << particle_mesh_interface.bounding_box[dx] << " "
              << particle_mesh_interface.global_bounding_box[dx] << std::endl;
  }
}
