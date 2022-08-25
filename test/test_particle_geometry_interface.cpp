#include "nektar_interface/particle_interface.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Driver.h>
#include <array>
#include <cmath>
#include <deque>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <random>
#include <set>
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
                   //"test/test_resources/unit_square_0_5.xml"};
                   "test/test_resources/unit_square_0_05.xml"};

  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);

  // Create MeshGraph.
  graph = SpatialDomains::MeshGraph::Read(session);

  ParticleMeshInterface particle_mesh_interface(graph);

  ASSERT_EQ(particle_mesh_interface.ndim, 2);

  // get bounding boxes of owned cells
  MeshHierarchyBoundingBoxIntersection mhbbi(
      particle_mesh_interface.mesh_hierarchy,
      particle_mesh_interface.owned_mh_cells);

  // get all remote geometry objects on this rank
  auto remote_triangles =
      get_all_remote_geoms_2d(MPI_COMM_WORLD, graph->GetAllTriGeoms());
  auto remote_quads =
      get_all_remote_geoms_2d(MPI_COMM_WORLD, graph->GetAllQuadGeoms());

  // filter to keep the geoms which intersect owned MeshHierarchy cells
  std::deque<std::shared_ptr<RemoteGeom2D<TriGeom>>> ring_passed_tris;
  std::deque<std::shared_ptr<RemoteGeom2D<QuadGeom>>> ring_passed_quads;

  std::set<int> ids_tris;
  std::set<int> ids_quads;

  for (auto &geom : remote_triangles) {
    std::array bounding_box = geom->geom->GetBoundingBox();
    if (mhbbi.intersects(bounding_box)) {
      ids_tris.insert(geom->id);
      ring_passed_tris.push_back(geom);
    }
  }
  for (auto &geom : remote_quads) {
    std::array bounding_box = geom->geom->GetBoundingBox();
    if (mhbbi.intersects(bounding_box)) {
      ids_quads.insert(geom->id);
      ring_passed_quads.push_back(geom);
    }
  }

  // nprint("tris:", ids_tris.size(),
  //        particle_mesh_interface.remote_triangles.size());
  // nprint("quads:", ids_quads.size(),
  //        particle_mesh_interface.remote_quads.size());

  // check the same geoms where ring passed as communicated in the interface
  // class
  for (auto &geom : particle_mesh_interface.remote_triangles) {
    const int id = geom->id;
    ASSERT_EQ(ids_tris.count(id), 1);
    ids_tris.erase(id);
  }
  ASSERT_EQ(ids_tris.size(), 0);

  for (auto &geom : particle_mesh_interface.remote_quads) {
    const int id = geom->id;
    ASSERT_EQ(ids_quads.count(id), 1);
    ids_quads.erase(id);
  }
  ASSERT_EQ(ids_quads.size(), 0);

  particle_mesh_interface.free();
}

TEST(ParticleGeometryInterface, PBC) {

  int size, rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;

  int argc = 2;
  char *argv[2] = {"test_particle_geometry_interface",
                   //"test/test_resources/unit_square_0_5.xml"};
                   "test/test_resources/unit_square_0_05.xml"};

  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);

  // Create MeshGraph.
  graph = SpatialDomains::MeshGraph::Read(session);

  // create particle mesh interface
  ParticleMeshInterface particle_mesh_interface(graph);

  SYCLTarget sycl_target{0, MPI_COMM_WORLD};

  Domain domain(particle_mesh_interface);
  const int ndim = particle_mesh_interface.ndim;

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("P_ORIG"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  ParticleGroup A(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A.position_dat);

  ASSERT_TRUE(std::abs(pbc.global_origin[0]) < 1.0e-14);
  ASSERT_TRUE(std::abs(pbc.global_origin[1]) < 1.0e-14);
  ASSERT_TRUE(std::abs(pbc.global_extent[0] - 1.0) < 1.0e-14);
  ASSERT_TRUE(std::abs(pbc.global_extent[1] - 1.0) < 1.0e-14);

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_rank(18241 + rank);
  std::mt19937 rng_cell(3258241 + rank);

  const int N = 1024;

  std::uniform_real_distribution<double> uniform_rng(-100.0, 100.0);

  const int cell_count = particle_mesh_interface.get_cell_count();
  std::uniform_int_distribution<int> cell_dist(0, cell_count - 1);

  ParticleSet initial_distribution(N, A.get_particle_spec());

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      const REAL pos = uniform_rng(rng_pos);
      initial_distribution[Sym<REAL>("P")][px][dimx] = pos;
      initial_distribution[Sym<REAL>("P_ORIG")][px][dimx] = pos;
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cell_dist(rng_cell);
    initial_distribution[Sym<INT>("ID")][px][0] = px;
  }
  A.add_particles_local(initial_distribution);

  pbc.execute();

  // ParticleDat P should contain the perodically mapped positions in P_ORIG

  // for each local cell
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto pos = A[Sym<REAL>("P")]->cell_dat.get_cell(cellx);
    auto pos_orig = A[Sym<REAL>("P_ORIG")]->cell_dat.get_cell(cellx);

    ASSERT_EQ(pos->nrow, pos_orig->nrow);
    const int nrow = pos->nrow;

    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {

      // for each dimension
      for (int dimx = 0; dimx < ndim; dimx++) {
        const REAL correct_pos =
            std::fmod((*pos_orig)[dimx][rowx] + 1000.0, 1.0);
        const REAL to_test_pos = (*pos)[dimx][rowx];
        ASSERT_TRUE(ABS(to_test_pos - correct_pos) < 1.0e-10);
        ASSERT_TRUE(correct_pos >= 0.0);
        ASSERT_TRUE(correct_pos <= 1.0);
      }
    }
  }

  particle_mesh_interface.free();
}