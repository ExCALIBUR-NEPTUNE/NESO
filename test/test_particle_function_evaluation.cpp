#include "nektar_interface/function_evaluation.hpp"
#include "nektar_interface/particle_interface.hpp"
#include "nektar_interface/utilities.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/DisContField.h>
#include <SolverUtils/Driver.h>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
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
using namespace Nektar::MultiRegions;
using namespace NESO::Particles;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

TEST(ParticleFunctionEvaluation, Scalar) {

  const int N_total = 1000000;
  const double tol = 1.0e-10;
  int argc = 3;
  char *argv[3];
  copy_to_cstring(std::string("test_particle_function_evaluation"), &argv[0]);
  copy_to_cstring(std::string("test/test_resources/square_triangles_quads.xml"),
                  &argv[1]);
  copy_to_cstring(std::string("test/test_resources/conditions.xml"), &argv[2]);

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);
  graph = SpatialDomains::MeshGraph::Read(session);

  auto dis_cont_field = std::make_shared<DisContField>(session, graph, "u");

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapperT>(sycl_target, mesh, tol);

  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = 2;
  const double extent[2] = {1.0, 1.0};
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<REAL>("FUNC_EVALS"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);
  std::mt19937 rng_rank(18241);

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
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
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

  auto lambda_f = [&](const NekDouble x, const NekDouble y) {
    return 2.0 * (x + 0.5) * (x - 0.5) * (y + 0.8) * (y - 0.8);
  };

  interpolate_onto_nektar_field_2d(lambda_f, dis_cont_field);

  write_vtu(dis_cont_field, "func.vtu");

  // create evaluation object
  auto field_evaluate = std::make_shared<FieldEvaluate<DisContField>>(
      dis_cont_field, A, cell_id_translation);

  // evaluate field at particle locations
  field_evaluate->evaluate(Sym<REAL>("FUNC_EVALS"));

  H5Part h5part("exp.h5part", A, Sym<REAL>("P"), Sym<INT>("NESO_MPI_RANK"),
                Sym<REAL>("NESO_REFERENCE_POSITIONS"),
                Sym<REAL>("FUNC_EVALS"));
  h5part.write();
  h5part.close();

  // check evaluations
  for (int cellx = 0; cellx < cell_count; cellx++) {

    auto positions = A->position_dat->cell_dat.get_cell(cellx);
    auto func_evals = (*A)[Sym<REAL>("FUNC_EVALS")]->cell_dat.get_cell(cellx);

    for (int rowx = 0; rowx < positions->nrow; rowx++) {

      const double x = (*positions)[0][rowx];
      const double y = (*positions)[1][rowx];

      const double eval_dat = (*func_evals)[0][rowx];
      const double eval_correct = lambda_f(x, y);

      const double err = ABS(eval_correct - eval_dat);

      ASSERT_TRUE(err <= 1.0e-5);
    }
  }

  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

TEST(ParticleFunctionEvaluation, Derivative) {

  const int N_total = 1000000;
  const double tol = 1.0e-10;
  int argc = 3;
  char *argv[3];
  copy_to_cstring(std::string("test_particle_function_evaluation"), &argv[0]);
  copy_to_cstring(std::string("test/test_resources/square_triangles_quads.xml"),
                  &argv[1]);
  copy_to_cstring(std::string("test/test_resources/conditions.xml"), &argv[2]);

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);
  graph = SpatialDomains::MeshGraph::Read(session);

  auto dis_cont_field = std::make_shared<DisContField>(session, graph, "u");

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapperT>(sycl_target, mesh, tol);

  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = 2;
  const double extent[2] = {1.0, 1.0};
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<REAL>("FUNC_EVALS_VECTOR"), 2)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);
  std::mt19937 rng_rank(18241);

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
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
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

  auto lambda_f = [&](const NekDouble x, const NekDouble y) {
    return 2.0 * (x + 0.5) * (x - 0.5) * (y + 0.8) * (y - 0.8);
  };

  interpolate_onto_nektar_field_2d(lambda_f, dis_cont_field);

  // write_vtu(dis_cont_field, "func.vtu");

  // create evaluation object
  auto field_evaluate = std::make_shared<FieldEvaluate<DisContField>>(
      dis_cont_field, A, cell_id_translation, true);

  // evaluate field at particle locations
  field_evaluate->evaluate(Sym<REAL>("FUNC_EVALS_VECTOR"));

  /*
    PhysTensorDeriv(
       const Array<OneD, const NekDouble> &inarray,
       Array<OneD, NekDouble> &outarray_d0, Array<OneD, NekDouble> &outarray_d1)
  */
  H5Part h5part(
      "exp_vector.h5part", A, Sym<REAL>("P"), Sym<INT>("NESO_MPI_RANK"),
      Sym<REAL>("NESO_REFERENCE_POSITIONS"), Sym<REAL>("FUNC_EVALS_VECTOR"));
  h5part.write();
  h5part.close();

  // check evaluations
  // for (int cellx = 0; cellx < cell_count; cellx++) {

  //  auto positions = A->position_dat->cell_dat.get_cell(cellx);
  //  auto func_evals =
  //  (*A)[Sym<REAL>("FUNC_EVALS_VECTOR")]->cell_dat.get_cell(cellx);

  //  for (int rowx = 0; rowx < positions->nrow; rowx++) {

  //    const double x = (*positions)[0][rowx];
  //    const double y = (*positions)[1][rowx];

  //    const double eval_dat = (*func_evals)[0][rowx];
  //    const double eval_correct = lambda_f(x, y);

  //    const double err = ABS(eval_correct - eval_dat);

  //    ASSERT_TRUE(err <= 1.0e-5);
  //  }
  //}

  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}
