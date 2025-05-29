#include "nektar_interface/function_projection.hpp"
#include "nektar_interface/particle_interface.hpp"
#include "nektar_interface/utilities.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
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
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace std;
using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::SpatialDomains;
using namespace Nektar::MultiRegions;
using namespace NESO::Particles;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

TEST(ParticleFunctionProjection, BasisEvalCorrectnessCG) {

  const int N_total = 1000;

  int argc = 3;
  char *argv[3];

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";
  std::filesystem::path mesh_file =
      test_resources_dir / "square_triangles_quads.xml";
  std::filesystem::path conditions_file =
      test_resources_dir / "conditions_cg.xml";

  copy_to_cstring(std::string("test_particle_function_projection"), &argv[0]);
  copy_to_cstring(std::string(mesh_file), &argv[1]);
  copy_to_cstring(std::string(conditions_file), &argv[2]);

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

  const int ndim = 2;
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<REAL>("Q"), 2),
                             ParticleProp(Sym<REAL>("Q2"), 3)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(5223423);

  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  int N = rend - rstart;

  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");

  const int cell_count = domain->mesh->get_cell_count();

  if (N > 0) {
    auto positions =
        uniform_within_extents(N_total, ndim, pbc.global_extent, rng_pos);

    std::uniform_int_distribution<int> uniform_dist(
        0, sycl_target->comm_pair.size_parent - 1);
    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig =
            positions[dimx][rstart + px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
      }
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % cell_count;
    }
    A->add_particles_local(initial_distribution);
  }
  reset_mpi_ranks((*A)[Sym<INT>("NESO_MPI_RANK")]);

  MeshHierarchyGlobalMap mesh_hierarchy_global_map(
      sycl_target, domain->mesh, A->position_dat, A->cell_id_dat,
      A->mpi_rank_dat);

  pbc.execute();
  mesh_hierarchy_global_map.execute();
  A->hybrid_move();
  cell_id_translation->execute();
  A->cell_move();

  const REAL two_over_sqrt_pi = 1.1283791670955126;
  const REAL reweight =
      pbc.global_extent[0] * pbc.global_extent[1] / ((REAL)N_total);

  auto ga_sum = std::make_shared<GlobalArray<REAL>>(sycl_target, 1, 0.0);
  particle_loop(
      A,
      [=](auto k_ga_sum, auto k_P, auto k_Q, auto k_Q2) {
        const REAL x = k_P.at(0);
        const REAL y = k_P.at(1);
        const REAL exp_eval =
            two_over_sqrt_pi * sycl::exp(-(4.0 * ((x) * (x) + (y) * (y))));
        k_Q.at(0) = exp_eval * reweight;
        k_Q.at(1) = -1.0 * exp_eval * reweight;
        k_Q2.at(1) = reweight;
        k_ga_sum.add(0, exp_eval * reweight);
      },
      Access::add(ga_sum), Access::read(Sym<REAL>("P")),
      Access::write(Sym<REAL>("Q")), Access::write(Sym<REAL>("Q2")))
      ->execute();

  auto cont_field_u = std::make_shared<ContField>(session, graph, "u");
  auto cont_field_v = std::make_shared<ContField>(session, graph, "v");
  auto cont_field_n = std::make_shared<ContField>(session, graph, "n");

  std::vector<std::shared_ptr<ContField>> cont_fields = {
      cont_field_u, cont_field_v, cont_field_n};
  // create projection object
  auto field_project = std::make_shared<FieldProject<ContField>>(
      cont_fields, A, cell_id_translation);

  // project field at particle locations
  std::vector<Sym<REAL>> project_syms = {Sym<REAL>("Q"), Sym<REAL>("Q"),
                                         Sym<REAL>("Q2")};
  std::vector<int> project_components = {0, 1, 1};

  field_project->testing_enable();
  field_project->project(project_syms, project_components);

  // Checks that the SYCL version matches the original version computed
  // using nektar
  field_project->project_host(project_syms, project_components);
  double *rhs_host, *rhs_device;
  field_project->testing_get_rhs(&rhs_host, &rhs_device);
  const int ncoeffs = cont_field_u->GetNcoeffs();
  for (int cx = 0; cx < ncoeffs; cx++) {
    EXPECT_NEAR(rhs_host[cx], rhs_device[cx], 1.0e-5);
    EXPECT_NEAR(rhs_host[cx + ncoeffs], rhs_device[cx + ncoeffs], 1.0e-5);
    EXPECT_NEAR(rhs_host[cx + 2 * ncoeffs], rhs_device[cx + 2 * ncoeffs],
                1.0e-5);
  }

  const double integral = cont_field_u->Integral(cont_field_u->GetPhys());
  auto global_sum = ga_sum->get().at(0);
  EXPECT_NEAR(global_sum, integral, 0.005);

  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

TEST(ParticleFunctionProjection, BasisEvalCorrectnessDG) {

  const int N_total = 1000;

  int argc = 3;
  char *argv[3];

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";
  std::filesystem::path mesh_file =
      test_resources_dir / "square_triangles_quads.xml";
  std::filesystem::path conditions_file = test_resources_dir / "conditions.xml";

  copy_to_cstring(std::string("test_particle_function_projection"), &argv[0]);
  copy_to_cstring(std::string(mesh_file), &argv[1]);
  copy_to_cstring(std::string(conditions_file), &argv[2]);

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

  const int ndim = 2;
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<REAL>("Q"), 2),
                             ParticleProp(Sym<REAL>("Q2"), 3)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(5223423);

  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  int N = rend - rstart;

  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");

  const int cell_count = domain->mesh->get_cell_count();

  if (N > 0) {
    auto positions =
        uniform_within_extents(N_total, ndim, pbc.global_extent, rng_pos);

    std::uniform_int_distribution<int> uniform_dist(
        0, sycl_target->comm_pair.size_parent - 1);
    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig =
            positions[dimx][rstart + px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
      }
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % cell_count;
    }
    A->add_particles_local(initial_distribution);
  }
  reset_mpi_ranks((*A)[Sym<INT>("NESO_MPI_RANK")]);

  MeshHierarchyGlobalMap mesh_hierarchy_global_map(
      sycl_target, domain->mesh, A->position_dat, A->cell_id_dat,
      A->mpi_rank_dat);

  pbc.execute();
  mesh_hierarchy_global_map.execute();
  A->hybrid_move();
  cell_id_translation->execute();
  A->cell_move();

  const REAL two_over_sqrt_pi = 1.1283791670955126;
  const REAL reweight =
      pbc.global_extent[0] * pbc.global_extent[1] / ((REAL)N_total);

  auto ga_sum = std::make_shared<GlobalArray<REAL>>(sycl_target, 1, 0.0);
  particle_loop(
      A,
      [=](auto k_ga_sum, auto k_P, auto k_Q, auto k_Q2) {
        const REAL x = k_P.at(0);
        const REAL y = k_P.at(1);
        const REAL exp_eval =
            two_over_sqrt_pi * sycl::exp(-(4.0 * ((x) * (x) + (y) * (y))));
        k_Q.at(0) = exp_eval * reweight;
        k_Q.at(1) = -1.0 * exp_eval * reweight;
        k_Q2.at(1) = reweight;
        k_ga_sum.add(0, exp_eval * reweight);
      },
      Access::add(ga_sum), Access::read(Sym<REAL>("P")),
      Access::write(Sym<REAL>("Q")), Access::write(Sym<REAL>("Q2")))
      ->execute();

  auto dis_cont_field_u = std::make_shared<DisContField>(session, graph, "u");
  auto dis_cont_field_v = std::make_shared<DisContField>(session, graph, "v");
  auto dis_cont_field_n = std::make_shared<DisContField>(session, graph, "n");

  std::vector<std::shared_ptr<DisContField>> dis_cont_fields = {
      dis_cont_field_u, dis_cont_field_v, dis_cont_field_n};
  // create projection object
  auto field_project = std::make_shared<FieldProject<DisContField>>(
      dis_cont_fields, A, cell_id_translation);

  // project field at particle locations
  std::vector<Sym<REAL>> project_syms = {Sym<REAL>("Q"), Sym<REAL>("Q"),
                                         Sym<REAL>("Q2")};
  std::vector<int> project_components = {0, 1, 1};

  field_project->testing_enable();
  field_project->project(project_syms, project_components);

  // Checks that the SYCL version matches the original version computed
  // using nektar
  field_project->project_host(project_syms, project_components);
  double *rhs_host, *rhs_device;
  field_project->testing_get_rhs(&rhs_host, &rhs_device);
  const int ncoeffs = dis_cont_field_u->GetNcoeffs();
  for (int cx = 0; cx < ncoeffs; cx++) {
    EXPECT_NEAR(rhs_host[cx], rhs_device[cx], 1.0e-5);
    EXPECT_NEAR(rhs_host[cx + ncoeffs], rhs_device[cx + ncoeffs], 1.0e-5);
    EXPECT_NEAR(rhs_host[cx + 2 * ncoeffs], rhs_device[cx + 2 * ncoeffs],
                1.0e-5);
  }

  const double integral =
      dis_cont_field_u->Integral(dis_cont_field_u->GetPhys());

  auto global_sum = ga_sum->get().at(0);
  EXPECT_NEAR(global_sum, integral, 0.005);

  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}
