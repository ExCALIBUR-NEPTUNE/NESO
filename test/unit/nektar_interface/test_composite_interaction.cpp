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

  CompositeIntersectionTester(
      SYCLTargetSharedPtr sycl_target,
      ParticleMeshInterfaceSharedPtr particle_mesh_interface,
      std::vector<int> &composite_indices)
      : CompositeIntersection(sycl_target, particle_mesh_interface,
                              composite_indices) {}
};

} // namespace

TEST(CompositeInteraction, Intersection) {
  const int N_total = 2000;

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
  const int N = rend - rstart;
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

  std::vector<int> composite_indices = {100, 200};
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

  write_vtk_mesh_hierarchy_cells_owned("mesh_hierarchy_cells", mesh);
  write_vtk_cells_owned("mesh_owned_cells", mesh);
  H5Part h5part("trajectory.h5part", A, Sym<REAL>("P"));
  h5part.write();
  h5part.close();

  A->free();
  mesh->free();
  delete[] argv[0];
  delete[] argv[1];
}
