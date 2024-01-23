#include "nektar_interface/composite_interaction/composite_interaction.hpp"
#include "nektar_interface/particle_interface.hpp"
#include "nektar_interface/utility_mesh_plotting.hpp"
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
#include <neso_particles.hpp>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace std;
using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::LibUtilities;
using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;
using namespace NESO::CompositeInteraction;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

namespace {

class CompositeIntersectionTester
    : public CompositeInteraction::CompositeIntersection {

public:
  inline void test_find_cells(ParticleGroupSharedPtr particle_group,
                              std::set<INT> &cells) {
    return this->find_cells(particle_group, cells);
  }

  inline std::unique_ptr<CompositeTransport> &get_composite_transport() {
    return this->composite_collections->composite_transport;
  }

  inline std::shared_ptr<CompositeCollections> &get_composite_collections() {
    return this->composite_collections;
  }

  CompositeIntersectionTester(
      SYCLTargetSharedPtr sycl_target,
      ParticleMeshInterfaceSharedPtr particle_mesh_interface,
      std::vector<int> &composite_indices)
      : CompositeIntersection(sycl_target, particle_mesh_interface,
                              composite_indices) {}
};

class CompositeTransportTester
    : public CompositeInteraction::CompositeTransport {

public:
  CompositeTransportTester(
      ParticleMeshInterfaceSharedPtr particle_mesh_interface,
      std::vector<int> &composite_indices)
      : CompositeTransport(particle_mesh_interface, composite_indices) {}

  inline auto &get_packed_geoms() { return this->packed_geoms; }
};

} // namespace

TEST(CompositeInteraction, GeometryTransport) {
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;

  int argc = 3;
  char *argv[3];
  copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";
  std::filesystem::path conditions_file =
      test_resources_dir / "reference_all_types_cube/conditions.xml";
  copy_to_cstring(std::string(conditions_file), &argv[1]);
  std::filesystem::path mesh_file =
      test_resources_dir / "reference_all_types_cube/mixed_ref_cube_0.2.xml";
  copy_to_cstring(std::string(mesh_file), &argv[2]);

  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);

  // Create MeshGraph.
  graph = SpatialDomains::MeshGraph::Read(session);
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto comm = mesh->get_comm();
  auto sycl_target = std::make_shared<SYCLTarget>(0, comm);

  std::vector<int> composite_indices = {100, 200, 300, 400, 500, 600};

  auto composite_transport =
      std::make_shared<CompositeTransportTester>(mesh, composite_indices);

  auto &packed_geoms = composite_transport->get_packed_geoms();

  int cell = -1;
  int num_bytes;
  for (auto &itemx : packed_geoms) {
    if (itemx.second.size() > 0) {
      cell = itemx.first;
      num_bytes = itemx.second.size();
    }
  }

  int rank = sycl_target->comm_pair.rank_parent;
  int possible_rank = (cell == -1) ? -1 : rank;
  int chosen_rank;
  MPICHK(
      MPI_Allreduce(&possible_rank, &chosen_rank, 1, MPI_INT, MPI_MAX, comm));
  ASSERT_TRUE(chosen_rank >= 0);
  MPICHK(MPI_Bcast(&cell, 1, MPI_INT, chosen_rank, comm));
  MPICHK(MPI_Bcast(&num_bytes, 1, MPI_INT, chosen_rank, comm));

  // distribute the packed version of the geoms and check the distributed
  // packed versions are correct
  std::vector<unsigned char> recv_geoms(num_bytes);
  if (rank == chosen_rank) {
    std::copy(packed_geoms.at(cell).begin(), packed_geoms.at(cell).end(),
              recv_geoms.begin());
  }
  MPICHK(MPI_Bcast(recv_geoms.data(), num_bytes, MPI_UNSIGNED_CHAR, chosen_rank,
                   comm));

  std::set<INT> cell_arg;
  cell_arg.insert(cell);
  composite_transport->collect_geometry(cell_arg);

  auto to_test_geoms = packed_geoms.at(cell);
  ASSERT_EQ(to_test_geoms, recv_geoms);

  // Check the unpacked versions are correct
  std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::QuadGeom>>>
      remote_quads;
  std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::TriGeom>>>
      remote_tris;
  composite_transport->get_geometry(cell, remote_quads, remote_tris);

  auto lambda_check = [&](auto geom) {
    int correct_rank;
    int correct_id;
    if (rank == chosen_rank) {
      correct_rank = geom->rank;
      correct_id = geom->id;
    }
    MPICHK(MPI_Bcast(&correct_rank, 1, MPI_INT, chosen_rank, comm));
    MPICHK(MPI_Bcast(&correct_id, 1, MPI_INT, chosen_rank, comm));
    ASSERT_EQ(geom->rank, correct_rank);
    ASSERT_EQ(geom->id, correct_id);
  };

  for (auto &gx : remote_quads) {
    lambda_check(gx);
  }
  for (auto &gx : remote_tris) {
    lambda_check(gx);
  }

  mesh->free();
  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

TEST(CompositeInteraction, Collections) {
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;

  int argc = 3;
  char *argv[3];
  copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";
  std::filesystem::path conditions_file =
      test_resources_dir / "reference_all_types_cube/conditions.xml";
  copy_to_cstring(std::string(conditions_file), &argv[1]);
  std::filesystem::path mesh_file =
      test_resources_dir / "reference_all_types_cube/mixed_ref_cube_0.2.xml";
  copy_to_cstring(std::string(mesh_file), &argv[2]);

  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);

  // Create MeshGraph.
  graph = SpatialDomains::MeshGraph::Read(session);
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto comm = mesh->get_comm();
  auto sycl_target = std::make_shared<SYCLTarget>(0, comm);

  std::vector<int> composite_indices = {100, 200, 300, 400, 500, 600};

  auto composite_transport =
      std::make_shared<CompositeTransportTester>(mesh, composite_indices);

  auto &packed_geoms = composite_transport->get_packed_geoms();

  int cell = -1;
  int num_bytes;
  for (auto &itemx : packed_geoms) {
    if (itemx.second.size() > 0) {
      cell = itemx.first;
      num_bytes = itemx.second.size();
    }
  }

  int rank = sycl_target->comm_pair.rank_parent;
  int possible_rank = (cell == -1) ? -1 : rank;
  int chosen_rank;
  MPICHK(
      MPI_Allreduce(&possible_rank, &chosen_rank, 1, MPI_INT, MPI_MAX, comm));
  ASSERT_TRUE(chosen_rank >= 0);
  MPICHK(MPI_Bcast(&cell, 1, MPI_INT, chosen_rank, comm));
  MPICHK(MPI_Bcast(&num_bytes, 1, MPI_INT, chosen_rank, comm));

  // distribute the packed version of the geoms and check the distributed
  // packed versions are correct
  std::vector<unsigned char> recv_geoms(num_bytes);
  if (rank == chosen_rank) {
    std::copy(packed_geoms.at(cell).begin(), packed_geoms.at(cell).end(),
              recv_geoms.begin());
  }
  MPICHK(MPI_Bcast(recv_geoms.data(), num_bytes, MPI_UNSIGNED_CHAR, chosen_rank,
                   comm));

  std::set<INT> cell_arg;
  cell_arg.insert(cell);

  auto composite_collections = std::make_shared<CompositeCollections>(
      sycl_target, mesh, composite_indices);
  composite_collections->collect_geometry(cell_arg);

  auto map_cells_collections = composite_collections->map_cells_collections;

  CompositeCollection *d_cc;
  CompositeCollection h_cc;
  auto exists = map_cells_collections->host_get(cell, &d_cc);
  ASSERT_TRUE(exists);
  sycl_target->queue.memcpy(&h_cc, d_cc, sizeof(CompositeCollection))
      .wait_and_throw();

  int correct_num_quads;
  int correct_num_tris;
  int correct_stride_quads;
  int correct_stride_tris;

  if (rank == chosen_rank) {
    correct_num_quads = h_cc.num_quads;
    correct_num_tris = h_cc.num_tris;
    correct_stride_quads = h_cc.stride_quads;
    correct_stride_tris = h_cc.stride_tris;
  }

  MPICHK(MPI_Bcast(&correct_num_quads, 1, MPI_INT, chosen_rank, comm));
  MPICHK(MPI_Bcast(&correct_num_tris, 1, MPI_INT, chosen_rank, comm));
  MPICHK(MPI_Bcast(&correct_stride_quads, 1, MPI_INT, chosen_rank, comm));
  MPICHK(MPI_Bcast(&correct_stride_tris, 1, MPI_INT, chosen_rank, comm));

  EXPECT_EQ(h_cc.num_quads, correct_num_quads);
  EXPECT_EQ(h_cc.num_tris, correct_num_tris);
  EXPECT_EQ(h_cc.stride_quads, correct_stride_quads);
  EXPECT_EQ(h_cc.stride_tris, correct_stride_tris);

  std::vector<int> correct_composite_ids_quads(correct_num_quads);
  std::vector<int> correct_composite_ids_tris(correct_num_tris);
  std::vector<int> correct_geom_ids_quads(correct_num_quads);
  std::vector<int> correct_geom_ids_tris(correct_num_tris);
  std::vector<int> test_composite_ids_quads(correct_num_quads);
  std::vector<int> test_composite_ids_tris(correct_num_tris);
  std::vector<int> test_geom_ids_quads(correct_num_quads);
  std::vector<int> test_geom_ids_tris(correct_num_tris);

  auto q = sycl_target->queue;
  if (rank == chosen_rank) {
    q.memcpy(correct_composite_ids_quads.data(), h_cc.composite_ids_quads,
             correct_num_quads * sizeof(int))
        .wait_and_throw();
    q.memcpy(correct_composite_ids_tris.data(), h_cc.composite_ids_tris,
             correct_num_tris * sizeof(int))
        .wait_and_throw();
    q.memcpy(correct_geom_ids_quads.data(), h_cc.geom_ids_quads,
             correct_num_quads * sizeof(int))
        .wait_and_throw();
    q.memcpy(correct_geom_ids_tris.data(), h_cc.geom_ids_tris,
             correct_num_tris * sizeof(int))
        .wait_and_throw();
  }
  q.memcpy(test_composite_ids_quads.data(), h_cc.composite_ids_quads,
           correct_num_quads * sizeof(int))
      .wait_and_throw();
  q.memcpy(test_composite_ids_tris.data(), h_cc.composite_ids_tris,
           correct_num_tris * sizeof(int))
      .wait_and_throw();
  q.memcpy(test_geom_ids_quads.data(), h_cc.geom_ids_quads,
           correct_num_quads * sizeof(int))
      .wait_and_throw();
  q.memcpy(test_geom_ids_tris.data(), h_cc.geom_ids_tris,
           correct_num_tris * sizeof(int))
      .wait_and_throw();

  MPICHK(MPI_Bcast(correct_composite_ids_quads.data(), correct_num_quads,
                   MPI_INT, chosen_rank, comm));
  MPICHK(MPI_Bcast(correct_composite_ids_tris.data(), correct_num_tris, MPI_INT,
                   chosen_rank, comm));
  MPICHK(MPI_Bcast(correct_geom_ids_quads.data(), correct_num_quads, MPI_INT,
                   chosen_rank, comm));
  MPICHK(MPI_Bcast(correct_geom_ids_tris.data(), correct_num_tris, MPI_INT,
                   chosen_rank, comm));

  EXPECT_EQ(correct_composite_ids_quads, test_composite_ids_quads);
  EXPECT_EQ(correct_composite_ids_tris, test_composite_ids_tris);
  EXPECT_EQ(correct_geom_ids_quads, test_geom_ids_quads);
  EXPECT_EQ(correct_geom_ids_tris, test_geom_ids_tris);

  mesh->free();
  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

TEST(CompositeInteraction, AtomicFetchMaxMin) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  typedef int TEST_INT;

  std::vector<TEST_INT> h_buffer = {0, 0};
  auto dh_buffer = BufferDeviceHost<TEST_INT>(sycl_target, h_buffer);
  auto k_buffer = dh_buffer.d_buffer.ptr;

  sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(1), [=](sycl::id<1> idx) {
          sycl::atomic_ref<TEST_INT, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
              amax(k_buffer[0]);
          amax.fetch_max((TEST_INT)8);
          sycl::atomic_ref<TEST_INT, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
              amin(k_buffer[1]);
          amin.fetch_min((TEST_INT)-8);
        });
      })
      .wait_and_throw();

  dh_buffer.device_to_host();

  EXPECT_EQ(dh_buffer.h_buffer.ptr[0], 8);
  EXPECT_EQ(dh_buffer.h_buffer.ptr[1], -8);

  sycl_target->free();
}

TEST(CompositeInteraction, Intersection) {
  const int N_total = 5000;

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;

  int argc = 3;
  char *argv[3];
  copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";
  std::filesystem::path conditions_file =
      test_resources_dir / "reference_all_types_cube/conditions.xml";
  copy_to_cstring(std::string(conditions_file), &argv[1]);
  std::filesystem::path mesh_file =
      test_resources_dir / "reference_all_types_cube/mixed_ref_cube_0.2.xml";
  copy_to_cstring(std::string(mesh_file), &argv[2]);

  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);

  // Create MeshGraph.
  graph = SpatialDomains::MeshGraph::Read(session);
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());
  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = 3;
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;
  std::mt19937 rng_pos(52234234 + rank);
  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  int N = rend - rstart;
  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");
  const int cell_count = domain->mesh->get_cell_count();

  if (N > 0) {
    auto positions =
        uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);

    std::uniform_int_distribution<int> uniform_dist(
        0, sycl_target->comm_pair.size_parent - 1);
    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
      }
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % cell_count;
    }
    A->add_particles_local(initial_distribution);
  }

  pbc.execute();
  A->hybrid_move();
  cell_id_translation->execute();
  A->cell_move();

  std::vector<int> composite_indices = {100, 200, 300, 400, 500, 600};
  std::set<int> composite_indices_set = {100, 200, 300, 400, 500, 600};

  auto composite_intersection = std::make_shared<CompositeIntersectionTester>(
      sycl_target, mesh, composite_indices);

  // Test pre integration actually copied the current positions
  composite_intersection->pre_integration(A);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto P = A->position_dat->cell_dat.get_cell(cellx);
    auto PP = A->get_dat(composite_intersection->previous_position_sym)
                  ->cell_dat.get_cell(cellx);

    for (int rowx = 0; rowx < P->nrow; rowx++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        ASSERT_EQ((*P)[dimx][rowx], (*PP)[dimx][rowx]);
      }
    }
  }

  std::unique_ptr<CompositeTransport> &composite_transport =
      composite_intersection->get_composite_transport();

  // find cells on the unmoved particles should return the mesh hierarchy cells
  // the particles are currently in
  std::set<INT> cells;
  composite_intersection->test_find_cells(A, cells);

  auto mesh_hierarchy_mapper = std::make_unique<MeshHierarchyMapper>(
      sycl_target, mesh->get_mesh_hierarchy());
  const auto mesh_hierarchy_device_mapper =
      mesh_hierarchy_mapper->get_host_mapper();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto P = A->position_dat->cell_dat.get_cell(cellx);
    for (int rowx = 0; rowx < P->nrow; rowx++) {
      REAL position[3];
      INT mh_cell[6];
      for (int dimx = 0; dimx < ndim; dimx++) {
        position[dimx] = (*P)[dimx][rowx];
      }

      mesh_hierarchy_device_mapper.map_to_tuple(position, mh_cell);
      const INT linear_cell =
          mesh_hierarchy_device_mapper.tuple_to_linear_global(mh_cell);
      ASSERT_TRUE(cells.count(linear_cell));
    }
  }

  composite_transport->collect_geometry(cells);
  const int second_size = composite_transport->collect_geometry(cells);
  // two calls to collect geometry with the same set of cells should return 0
  // new cells collected on the second call.
  ASSERT_EQ(second_size, 0);

  REAL offset_x;
  REAL offset_y;
  REAL offset_z;

  auto lambda_apply_offset = [&]() {
    particle_loop(
        A,
        [=](auto P) {
          P.at(0) += offset_x;
          P.at(1) += offset_y;
          P.at(2) += offset_z;
        },
        Access::write(Sym<REAL>("P")))
        ->execute();
  };

  auto reset_positions = particle_loop(
      A,
      [&](auto P, auto PP) {
        for (int dx = 0; dx < ndim; dx++) {
          P.at(dx) = PP.at(dx);
        }
      },
      Access::write(Sym<REAL>("P")),
      Access::read(Sym<REAL>("NESO_COMP_INT_PREV_POS")));

  auto lambda_test = [&](const int expected_composite) {
    composite_intersection->pre_integration(A);
    ASSERT_TRUE(A->contains_dat(Sym<REAL>("NESO_COMP_INT_PREV_POS")));
    lambda_apply_offset();
    auto sub_groups = composite_intersection->get_intersections(A);
    ASSERT_TRUE(A->contains_dat(Sym<REAL>("NESO_COMP_INT_OUTPUT_POS")));
    ASSERT_TRUE(A->contains_dat(Sym<INT>("NESO_COMP_INT_OUTPUT_COMP")));

    int local_count = 0;
    for (int cellx = 0; cellx < cell_count; cellx++) {
      auto P = A->get_cell(Sym<REAL>("P"), cellx);
      auto IP = A->get_cell(Sym<REAL>("NESO_COMP_INT_OUTPUT_POS"), cellx);
      auto IC = A->get_cell(Sym<INT>("NESO_COMP_INT_OUTPUT_COMP"), cellx);
      for (int rowx = 0; rowx < P->nrow; rowx++) {

        auto hit_composite = IC->at(rowx, 0);
        auto composite_id = IC->at(rowx, 1);
        auto geom_id = IC->at(rowx, 2);
        ASSERT_EQ(hit_composite, 1);
        ASSERT_TRUE(composite_indices_set.count(composite_id) == 1);

        auto geom = composite_intersection->composite_collections
                        ->map_composites_to_geoms.at(composite_id)
                        .at(geom_id);

        Array<OneD, NekDouble> point(3);
        point[0] = IP->at(rowx, 0);
        point[1] = IP->at(rowx, 1);
        point[2] = IP->at(rowx, 2);
        ASSERT_TRUE(geom->ContainsPoint(point));
        local_count++;
      }
    }

    for (const auto cx : composite_indices) {
      if (cx == expected_composite) {
        ASSERT_EQ(sub_groups.at(cx)->get_npart_local(), local_count);
      } else {
        ASSERT_EQ(sub_groups.at(cx)->get_npart_local(), 0);
      }
    }

    reset_positions->execute();
  };

  offset_x = 2.0;
  offset_y = 0.0;
  offset_z = 0.0;
  lambda_test(300);
  offset_x = -2.0;
  offset_y = 0.0;
  offset_z = 0.0;
  lambda_test(400);
  offset_x = 0.0;
  offset_y = 2.0;
  offset_z = 0.0;
  lambda_test(200);
  offset_x = 0.0;
  offset_y = -2.0;
  offset_z = 0.0;
  lambda_test(100);
  offset_x = 0.0;
  offset_y = 0.0;
  offset_z = 2.0;
  lambda_test(600);
  offset_x = 0.0;
  offset_y = 0.0;
  offset_z = -2.0;
  lambda_test(500);

  A->free();
  mesh->free();
  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

TEST(CompositeInteraction, Reflection) {
  const int N_total = 1000;
  const REAL dt = 0.1;
  const int N_steps = 50;

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;

  int argc = 3;
  char *argv[3];
  copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";
  std::filesystem::path conditions_file =
      test_resources_dir / "reference_all_types_cube/conditions.xml";
  copy_to_cstring(std::string(conditions_file), &argv[1]);
  std::filesystem::path mesh_file =
      test_resources_dir / "reference_all_types_cube/mixed_ref_cube_0.2.xml";
  copy_to_cstring(std::string(mesh_file), &argv[2]);

  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);

  // Create MeshGraph.
  graph = SpatialDomains::MeshGraph::Read(session);
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto mapping_config = std::make_shared<ParameterStore>();
  mapping_config->set<REAL>("MapParticles3DRegular/tol", 1.0e-10);
  auto nektar_graph_local_mapper = std::make_shared<NektarGraphLocalMapper>(
      sycl_target, mesh, mapping_config);

  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = 3;
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;
  std::mt19937 rng_pos(52234234 + rank);
  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  int N = rend - rstart;
  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");
  const int cell_count = domain->mesh->get_cell_count();

  if (N > 0) {
    auto positions =
        uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);

    auto velocities =
        NESO::Particles::normal_distribution(N, 3, 0.0, 0.5, rng_pos);

    std::uniform_int_distribution<int> uniform_dist(
        0, sycl_target->comm_pair.size_parent - 1);
    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
        initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
      }
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % cell_count;
      initial_distribution[Sym<INT>("ID")][px][0] = px;
    }
    A->add_particles_local(initial_distribution);
  }

  pbc.execute();
  A->hybrid_move();
  cell_id_translation->execute();
  A->cell_move();

  std::vector<int> composite_indices = {100, 200, 300, 400, 500, 600};

  auto composite_intersection = std::make_shared<CompositeIntersectionTester>(
      sycl_target, mesh, composite_indices);

  auto reflection = std::make_shared<NektarCompositeTruncatedReflection>(
      Sym<REAL>("V"), sycl_target,
      composite_intersection->composite_collections, composite_indices);

  auto loop_advect = particle_loop(
      A,
      [=](auto P, auto V) {
        P.at(0) += dt * V.at(0);
        P.at(1) += dt * V.at(1);
        P.at(2) += dt * V.at(2);
      },
      Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")));

  // H5Part h5part("traj.h5part", A, Sym<REAL>("P"), Sym<REAL>("V"),
  //               Sym<INT>("ID"), Sym<INT>("CELL_ID"),
  //               Sym<INT>("NESO_COMP_INT_OUTPUT_COMP"),
  //               Sym<REAL>("NESO_COMP_INT_OUTPUT_POS"));
  for (int stepx = 0; stepx < N_steps; stepx++) {
    nprint(stepx);
    composite_intersection->pre_integration(A);
    loop_advect->execute();
    auto composite_intersections = composite_intersection->get_intersections(A);
    reflection->execute(composite_intersections);
    A->hybrid_move();
    cell_id_translation->execute();
    A->cell_move();
    // h5part.write();
  }
  // h5part.close();

  A->free();
  mesh->free();
  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

TEST(CompositeInteraction, SubGroupReflection) {
  const int N_total = 200;
  const REAL dt = 0.1;
  const int N_steps = 50;

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;

  int argc = 3;
  char *argv[3];
  copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";
  std::filesystem::path conditions_file =
      test_resources_dir / "reference_all_types_cube/conditions.xml";
  copy_to_cstring(std::string(conditions_file), &argv[1]);
  std::filesystem::path mesh_file =
      test_resources_dir / "reference_all_types_cube/mixed_ref_cube_0.2.xml";
  copy_to_cstring(std::string(mesh_file), &argv[2]);

  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);

  // Create MeshGraph.
  graph = SpatialDomains::MeshGraph::Read(session);
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto mapping_config = std::make_shared<ParameterStore>();
  mapping_config->set<REAL>("MapParticles3DRegular/tol", 1.0e-10);
  auto nektar_graph_local_mapper = std::make_shared<NektarGraphLocalMapper>(
      sycl_target, mesh, mapping_config);

  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = 3;
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;
  std::mt19937 rng_pos(52234234 + rank);
  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  int N = rend - rstart;
  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");
  const int cell_count = domain->mesh->get_cell_count();

  if (N > 0) {
    auto positions =
        uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);

    auto velocities =
        NESO::Particles::normal_distribution(N, 3, 0.0, 0.5, rng_pos);

    std::uniform_int_distribution<int> uniform_dist(
        0, sycl_target->comm_pair.size_parent - 1);
    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
        initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
      }
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % cell_count;
      initial_distribution[Sym<INT>("ID")][px][0] = px;
    }
    A->add_particles_local(initial_distribution);
  }

  pbc.execute();
  A->hybrid_move();
  cell_id_translation->execute();
  A->cell_move();

  std::vector<int> composite_indices = {100, 200, 300, 400, 500, 600};

  auto composite_intersection = std::make_shared<CompositeIntersectionTester>(
      sycl_target, mesh, composite_indices);

  auto reflection = std::make_shared<NektarCompositeTruncatedReflection>(
      Sym<REAL>("V"), sycl_target,
      composite_intersection->composite_collections, composite_indices);

  auto aa = particle_sub_group(
      A, [=](auto ID) { return (ID.at(0) % 2) == 0; }, Access::read(Sym<INT>("ID")));

  auto loop_advect = particle_loop(
      aa,
      [=](auto P, auto V) {
        P.at(0) += dt * V.at(0);
        P.at(1) += dt * V.at(1);
        P.at(2) += dt * V.at(2);
      },
      Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")));

  H5Part h5part("traj_single.h5part", A, Sym<REAL>("P"), Sym<REAL>("V"),
                Sym<INT>("ID"), Sym<INT>("CELL_ID"),
                Sym<INT>("NESO_COMP_INT_OUTPUT_COMP"),
                Sym<REAL>("NESO_COMP_INT_OUTPUT_POS"));

  for (int stepx = 0; stepx < N_steps; stepx++) {
    nprint(stepx);
    composite_intersection->pre_integration(aa);
    loop_advect->execute();
    auto composite_intersections =
        composite_intersection->get_intersections(aa);
    reflection->execute(composite_intersections);
    A->hybrid_move();
    cell_id_translation->execute();
    A->cell_move();
    h5part.write();
  }
  h5part.close();

  A->free();
  mesh->free();
  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}
