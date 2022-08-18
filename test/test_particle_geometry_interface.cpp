#include "nektar_interface/particle_interface.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Driver.h>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;
using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;

// Test overlap area computation
TEST(ParticleGeometryInterface, Overlap1D) {

  ASSERT_EQ(0.0, overlap_1d(0.0, 1.0, 1, 1.0));
  ASSERT_EQ(0.0, overlap_1d(1.0, 2.0, 0, 1.0));
  ASSERT_EQ(2.0, overlap_1d(1.00, 8.25, 2, 2.0));
  ASSERT_EQ(0.25, overlap_1d(1.75, 2.0, 1, 1.0));
  ASSERT_EQ(0.25, overlap_1d(1.00, 3.25, 3, 1.0));
}

// dummy element that returns a bounding box like Nektar++
class DummyElement {
private:
public:
  std::array<double, 6> b;

  ~DummyElement(){};
  DummyElement(const double b0 = 0.0, const double b1 = 0.0,
               const double b2 = 0.0, const double b3 = 0.0,
               const double b4 = 0.0, const double b5 = 0.0) {
    this->b[0] = b0;
    this->b[1] = b1;
    this->b[2] = b2;
    this->b[3] = b3;
    this->b[4] = b4;
    this->b[5] = b5;
  };

  inline std::array<double, 6> GetBoundingBox() { return this->b; }
};

// check the code that combines bounding boxes to get a global bounding box for
// several elemetns
TEST(ParticleGeometryInterface, BoundingBox) {

  auto e0 = std::make_shared<DummyElement>(-1.0, 1.0, -10.0, 0.0, 4.0, 0.0);
  auto e1 = std::make_shared<DummyElement>(1.0, 1.0, 1.0, 5.0, 2.0, 3.0);

  std::array<double, 6> bounding_box;
  for (int dimx = 0; dimx < 3; dimx++) {
    bounding_box[dimx] = std::numeric_limits<double>::max();
    bounding_box[dimx + 3] = std::numeric_limits<double>::min();
  }

  expand_bounding_box(e0, bounding_box);
  expand_bounding_box(e1, bounding_box);

  ASSERT_EQ(-1.0, bounding_box[0]);
  ASSERT_EQ(1.0, bounding_box[1]);
  ASSERT_EQ(-10.0, bounding_box[2]);
  ASSERT_EQ(5.0, bounding_box[3]);
  ASSERT_EQ(4.0, bounding_box[4]);
  ASSERT_EQ(3.0, bounding_box[5]);
}

// test the computation of overlap area between bounding boxes and cells.
TEST(ParticleGeometryInterface, BoundingBoxClaim) {

  const int ndim = 2;

  std::vector<int> dims(ndim);
  std::vector<double> origin(ndim);

  dims[0] = 1;
  dims[1] = 1;
  origin[0] = 0.0;
  origin[1] = 0.0;

  const int subdivision_order = 3;
  const double cell_extent = 1.0;

  MeshHierarchy mesh_hierarchy(MPI_COMM_WORLD, ndim, dims, origin, cell_extent,
                               subdivision_order);

  const auto cell_width_fine = mesh_hierarchy.cell_width_fine;

  const double cell_area = std::pow(cell_width_fine, ndim);

  // element only covering cell 0 in the lower left
  auto e0 = std::make_shared<DummyElement>(0.0, 0.0, 0.0, cell_width_fine,
                                           cell_width_fine, 0.0);

  LocalClaim local_claim0;
  MHGeomMap mh_geom_map_0;
  bounding_box_claim(42, e0, mesh_hierarchy, local_claim0, mh_geom_map_0);

  ASSERT_EQ(1, local_claim0.claim_cells.size());
  ASSERT_EQ(1, local_claim0.claim_cells.count(0));
  ASSERT_TRUE(std::abs(1000000 - local_claim0.claim_weights[0].weight) <= 1);
  ASSERT_TRUE(std::abs(1.0 - local_claim0.claim_weights[0].weightf) <= 1.0e-14);

  ASSERT_EQ(mh_geom_map_0[0].size(), 1);
  ASSERT_EQ(mh_geom_map_0[0][0], 42);

  // element covering the top right cell but overlapping into the adjacent cells
  auto e1 = std::make_shared<DummyElement>(
      (7 - 0.25) * cell_width_fine, (7 - 0.25) * cell_width_fine, 0.0,
      8 * cell_width_fine, 8 * cell_width_fine, 0.0);

  LocalClaim local_claim1;
  MHGeomMap mh_geom_map_1;
  bounding_box_claim(43, e1, mesh_hierarchy, local_claim1, mh_geom_map_1);

  ASSERT_EQ(4, local_claim1.claim_cells.size());

  ASSERT_TRUE(std::abs(1.0 - local_claim1.claim_weights[63].weightf) <=
              1.0e-14);
  ASSERT_TRUE(std::abs(0.25 - local_claim1.claim_weights[63 - 1].weightf) <=
              1.0e-14);
  ASSERT_TRUE(std::abs(0.25 - local_claim1.claim_weights[63 - 8].weightf) <=
              1.0e-14);
  ASSERT_TRUE(
      std::abs(0.25 * 0.25 - local_claim1.claim_weights[63 - 8 - 1].weightf) <=
      1.0e-14);

  ASSERT_TRUE(std::abs(1000000 - local_claim1.claim_weights[63].weight) <= 1);
  ASSERT_TRUE(std::abs(250000 - local_claim1.claim_weights[63 - 1].weight) <=
              1);
  ASSERT_TRUE(std::abs(250000 - local_claim1.claim_weights[63 - 8].weight) <=
              1);
  ASSERT_TRUE(std::abs(62500 - local_claim1.claim_weights[63 - 8 - 1].weight) <=
              1);

  ASSERT_EQ(mh_geom_map_1[63].size(), 1);
  ASSERT_EQ(mh_geom_map_1[63 - 1].size(), 1);
  ASSERT_EQ(mh_geom_map_1[63 - 8].size(), 1);
  ASSERT_EQ(mh_geom_map_1[63 - 8 - 1].size(), 1);
  ASSERT_EQ(mh_geom_map_1[63][0], 43);
  ASSERT_EQ(mh_geom_map_1[63 - 1][0], 43);
  ASSERT_EQ(mh_geom_map_1[63 - 8][0], 43);
  ASSERT_EQ(mh_geom_map_1[63 - 8 - 1][0], 43);

  // element that completely overlaps a cell adjacent to the top right corner
  // cell and should override the weight previously computed
  auto e2 = std::make_shared<DummyElement>(
      (6) * cell_width_fine, (6) * cell_width_fine, 0.0,
      (7 + 0.25) * cell_width_fine, (7 + 0.25) * cell_width_fine, 0.0);
  bounding_box_claim(44, e2, mesh_hierarchy, local_claim1, mh_geom_map_1);

  ASSERT_TRUE(std::abs(1.0 - local_claim1.claim_weights[63].weightf) <=
              1.0e-14);
  ASSERT_TRUE(std::abs(1000000 - local_claim1.claim_weights[63].weight) <= 1);
  ASSERT_TRUE(std::abs(1.0 - local_claim1.claim_weights[63 - 8 - 1].weightf) <=
              1.0e-14);
  ASSERT_TRUE(
      std::abs(1000000 - local_claim1.claim_weights[63 - 8 - 1].weight) <= 1);

  ASSERT_EQ(mh_geom_map_1[63].size(), 2);
  ASSERT_EQ(mh_geom_map_1[63 - 1].size(), 2);
  ASSERT_EQ(mh_geom_map_1[63 - 8].size(), 2);
  ASSERT_EQ(mh_geom_map_1[63 - 8 - 1].size(), 2);
  ASSERT_EQ(mh_geom_map_1[63][0], 43);
  ASSERT_EQ(mh_geom_map_1[63 - 1][0], 43);
  ASSERT_EQ(mh_geom_map_1[63 - 8][0], 43);
  ASSERT_EQ(mh_geom_map_1[63 - 8 - 1][0], 43);
  ASSERT_EQ(mh_geom_map_1[63][1], 44);
  ASSERT_EQ(mh_geom_map_1[63 - 1][1], 44);
  ASSERT_EQ(mh_geom_map_1[63 - 8][1], 44);
  ASSERT_EQ(mh_geom_map_1[63 - 8 - 1][1], 44);

  mesh_hierarchy.free();
}

// test bounding box intersection
TEST(ParticleGeometryInterface, BoundingBoxIntersection) {

  const int ndim = 2;

  std::vector<int> dims(ndim);
  std::vector<double> origin(ndim);

  dims[0] = 1;
  dims[1] = 1;
  origin[0] = 1.0;
  origin[1] = 1.0;

  const int subdivision_order = 3;
  const double cell_extent = 8.0;

  MeshHierarchy mesh_hierarchy(MPI_COMM_WORLD, ndim, dims, origin, cell_extent,
                               subdivision_order);

  const auto cell_width_fine = mesh_hierarchy.cell_width_fine;

  std::vector<INT> owned_cells(2);
  owned_cells[0] = 0;
  owned_cells[1] = 63 - 7;

  MeshHierarchyBoundingBoxIntersection mhbbi(mesh_hierarchy, owned_cells);
  std::array<double, 6> bb;

  // test the cell bounding boxes
  get_bounding_box(mesh_hierarchy, 0, bb);
  ASSERT_TRUE(std::abs(bb[0] - 1.0) < 1.0e-14);
  ASSERT_TRUE(std::abs(bb[1] - 1.0) < 1.0e-14);
  ASSERT_TRUE(std::abs(bb[3] - 2.0) < 1.0e-14);
  ASSERT_TRUE(std::abs(bb[4] - 2.0) < 1.0e-14);

  get_bounding_box(mesh_hierarchy, 63 - 7, bb);
  ASSERT_TRUE(std::abs(bb[0] - 1.0) < 1.0e-14);
  ASSERT_TRUE(std::abs(bb[1] - 8.0) < 1.0e-14);
  ASSERT_TRUE(std::abs(bb[3] - 2.0) < 1.0e-14);
  ASSERT_TRUE(std::abs(bb[4] - 9.0) < 1.0e-14);

  // make a far away bounding box
  bb[0] = 8.0;
  bb[1] = 8.0;
  bb[3] = 9.0;
  bb[4] = 9.0;
  ASSERT_TRUE(!mhbbi.intersects(bb));

  // close but not touching
  bb[0] = 0.0;
  bb[1] = 0.0;
  bb[3] = 0.9;
  bb[4] = 0.9;
  ASSERT_TRUE(!mhbbi.intersects(bb));

  // slight overlap
  bb[0] = 0.0;
  bb[1] = 0.0;
  bb[3] = 1.1;
  bb[4] = 1.1;
  ASSERT_TRUE(mhbbi.intersects(bb));
  bb[0] = 1.9;
  bb[1] = 8.0;
  bb[3] = 3.0;
  bb[4] = 9.0;
  ASSERT_TRUE(mhbbi.intersects(bb));

  mesh_hierarchy.free();
}

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

  for (auto &cellx : particle_mesh_interface.owned_mh_cells) {
    std::cout << cellx << std::endl;
  }

  particle_mesh_interface.free();
}
