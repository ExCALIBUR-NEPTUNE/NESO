#include "nektar_interface/function_evaluation.hpp"
#include "nektar_interface/particle_interface.hpp"
#include "nektar_interface/utilities.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ContField.h>
#include <MultiRegions/DisContField.h>
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
using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

template <typename FIELD_TYPE>
static inline void evaluation_wrapper_3d(std::string condtions_file_s,
                                         std::string mesh_file_s,
                                         const double tol) {

  const int N_total = 16000;

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
  graph = SpatialDomains::MeshGraph::Read(session);

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  std::mt19937 rng{182348};

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
    }
    A->add_particles_local(initial_distribution);
  }
  reset_mpi_ranks((*A)[Sym<INT>("NESO_MPI_RANK")]);

  pbc.execute();
  A->hybrid_move();
  cell_id_translation->execute();
  A->cell_move();

  auto field = std::make_shared<FIELD_TYPE>(session, graph, "u");

  auto lambda_f = [&](const NekDouble x, const NekDouble y, const NekDouble z) {
    return std::pow((x + 1.0) * (x - 1.0) * (y + 1.0) * (y - 1.0) * (z + 1.0) *
                        (z - 1.0),
                    4);
  };
  interpolate_onto_nektar_field_3d(lambda_f, field);
  NESOCellsToNektarExp map_cells_to_exp(field, cell_id_translation);

  auto field_evaluate = std::make_shared<FieldEvaluate<FIELD_TYPE>>(
      field, A, cell_id_translation);
  field_evaluate->evaluate(Sym<REAL>("E"));

  Array<OneD, NekDouble> local_coord(3);

  for (int cellx = 0; cellx < cell_count; cellx++) {

    auto cell_ids = A->cell_id_dat->cell_dat.get_cell(cellx);
    auto reference_positions =
        (*A)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]->cell_dat.get_cell(cellx);
    auto E = (*A)[Sym<REAL>("E")]->cell_dat.get_cell(cellx);

    const int exp_id = map_cells_to_exp.get_exp_id(cellx);
    const auto exp = map_cells_to_exp.get_exp(cellx);
    const auto exp_phys = field->GetPhys() + field->GetPhys_Offset(exp_id);

    for (int rowx = 0; rowx < cell_ids->nrow; rowx++) {

      local_coord[0] = (*reference_positions)[0][rowx];
      local_coord[1] = (*reference_positions)[1][rowx];
      local_coord[2] = (*reference_positions)[2][rowx];

      const REAL to_test = (*E)[0][rowx];
      const REAL correct = exp->StdPhysEvaluate(local_coord, exp_phys);

      const double err = relative_error(correct, to_test);
      const double err_abs = std::abs(correct - to_test);
      EXPECT_TRUE(err < tol || err_abs < tol);
    }
  }

  A->free();
  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

TEST(ParticleFunctionEvaluation3D, ContField) {
  evaluation_wrapper_3d<MultiRegions::ContField>(
      "reference_all_types_cube/conditions_cg.xml",
      "reference_all_types_cube/mixed_ref_cube_0.5_perturbed.xml", 1.0e-7);
}
TEST(ParticleFunctionEvaluation3D, DisContFieldHex) {
  evaluation_wrapper_3d<MultiRegions::DisContField>(
      "reference_hex_cube/conditions.xml",
      "reference_hex_cube/hex_cube_0.3_perturbed.xml", 1.0e-7);
}
TEST(ParticleFunctionEvaluation3D, DisContFieldPrismTet) {
  evaluation_wrapper_3d<MultiRegions::DisContField>(
      "reference_prism_tet_cube/conditions.xml",
      "reference_prism_tet_cube/prism_tet_cube_0.5_perturbed.xml", 1.0e-7);
}
