#include "nektar_interface/function_projection.hpp"
#include "nektar_interface/particle_interface.hpp"
#include "nektar_interface/utilities.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ContField.h>
#include <MultiRegions/DisContField.h>
#include <SolverUtils/Driver.h>
#include <SpatialDomains/MeshGraphIO.h>
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
using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

template <typename FIELD_TYPE>
static inline void projection_wrapper_3d(std::string condtions_file_s,
                                         std::string mesh_file_s,
                                         const double tol) {

  const int N_total = 32000;

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";

  std::filesystem::path condtions_file_basename{condtions_file_s};
  std::filesystem::path mesh_file_basename{mesh_file_s};
  std::filesystem::path conditions_file =
      test_resources_dir / condtions_file_basename;
  std::filesystem::path mesh_file = test_resources_dir / mesh_file_basename;

  int argc = 3;
  char *argv[3];
  copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);
  copy_to_cstring(std::string(conditions_file), &argv[1]);
  copy_to_cstring(std::string(mesh_file), &argv[2]);

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);
  graph = SpatialDomains::MeshGraphIO::Read(session);

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = 3;
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<REAL>("E"), 1),
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
  const int N = rend - rstart;
  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");
  const int cell_count = domain->mesh->get_cell_count();
  if (N > 0) {
    auto positions =
        uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);

    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
      }

      initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
      initial_distribution[Sym<INT>("ID")][px][0] = px;
      initial_distribution[Sym<REAL>("E")][px][0] = 2.12412;
    }
    A->add_particles_local(initial_distribution);
  }
  reset_mpi_ranks((*A)[Sym<INT>("NESO_MPI_RANK")]);

  pbc.execute();
  A->hybrid_move();
  cell_id_translation->execute();
  A->cell_move();

  auto field = std::make_shared<FIELD_TYPE>(session, graph, "u");

  std::vector<std::shared_ptr<FIELD_TYPE>> fields = {field};
  auto field_project = std::make_shared<FieldProject<FIELD_TYPE>>(
      fields, A, cell_id_translation);

  field_project->testing_enable();

  std::vector<Sym<REAL>> project_syms = {Sym<REAL>("E")};
  std::vector<int> project_components = {0};

  field_project->project(project_syms, project_components);

  // Checks that the SYCL version matches the original version computed
  // using nektar
  field_project->project_host(project_syms, project_components);

  double *rhs_host, *rhs_device;
  field_project->testing_get_rhs(&rhs_host, &rhs_device);

  const int ncoeffs = field->GetNcoeffs();
  for (int cx = 0; cx < ncoeffs; cx++) {
    const double err_rel = relative_error(rhs_host[cx], rhs_device[cx]);
    const double err_abs = std::abs(rhs_host[cx] - rhs_device[cx]);
    ASSERT_TRUE(err_rel < tol || err_abs < tol);
  }

  A->free();
  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

TEST(ParticleFunctionProjection3DBasisEval, ContField) {
  projection_wrapper_3d<MultiRegions::ContField>(
      "reference_all_types_cube/conditions_cg.xml",
      "reference_all_types_cube/linear_non_regular_0.5.xml", 1.0e-7);
}
TEST(ParticleFunctionProjection3D, DisContFieldHex) {
  projection_wrapper_3d<MultiRegions::DisContField>(
      "reference_hex_cube/conditions.xml",
      "reference_hex_cube/hex_cube_0.5.xml", 1.0e-7);
}
TEST(ParticleFunctionProjection3D, DisContFieldPrismTet) {
  projection_wrapper_3d<MultiRegions::DisContField>(
      "reference_prism_tet_cube/conditions.xml",
      "reference_prism_tet_cube/prism_tet_cube_0.5.xml", 1.0e-7);
}
