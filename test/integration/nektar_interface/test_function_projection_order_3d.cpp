#include "nektar_interface/function_projection.hpp"
#include "nektar_interface/particle_interface.hpp"
#include "nektar_interface/utilities.hpp"
#include "particle_utility/position_distribution.hpp"
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

static inline NekDouble func(const NekDouble x, const NekDouble y,
                             const NekDouble z) {
  return ((x + 1.0) * (x - 1.0) * (y + 1.0) * (y - 1.0) * (z + 1.0) *
          (z - 1.0));
};

template <typename FIELD_TYPE>
static inline void projection_wrapper_order_3d(std::string condtions_file_s,
                                               std::string mesh_file_s) {

  auto project_run = [&](const int N_total, const int samplex) {
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
                               ParticleProp(Sym<REAL>("Q"), 1),
                               ParticleProp(Sym<INT>("ID"), 1)};

    auto A =
        std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

    NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
    auto cell_id_translation =
        std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);
    const int rank = sycl_target->comm_pair.rank_parent;
    const int size = sycl_target->comm_pair.size_parent;

    std::mt19937 rng_pos(52234234 + samplex + rank);
    int rstart, rend;
    get_decomp_1d(size, N_total, rank, &rstart, &rend);
    const int N = rend - rstart;
    int N_check = -1;
    MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
    NESOASSERT(N_check == N_total, "Error creating particles");
    const int cell_count = domain->mesh->get_cell_count();
    if (N > 0) {
      auto positions =
          uniform_within_extents(N_total, ndim, pbc.global_extent, rng_pos);
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

    const REAL reweight = pbc.global_extent[0] * pbc.global_extent[1] *
                          pbc.global_extent[2] / ((REAL)N_total);

    particle_loop(
        A,
        [=](auto k_P, auto k_Q) {
          const REAL eval0 = func(k_P.at(0), k_P.at(1), k_P.at(2));
          const REAL eval = reweight * eval0;
          k_Q.at(0) = eval;
        },
        Access::read(Sym<REAL>("P")), Access::write(Sym<REAL>("Q")))
        ->execute();

    auto field = std::make_shared<FIELD_TYPE>(session, graph, "u");
    std::vector<std::shared_ptr<FIELD_TYPE>> fields = {field};
    auto field_project = std::make_shared<FieldProject<FIELD_TYPE>>(
        fields, A, cell_id_translation);

    field_project->testing_enable();
    std::vector<Sym<REAL>> project_syms = {Sym<REAL>("Q")};
    std::vector<int> project_components = {0};

    field_project->project(project_syms, project_components);

    const int tot_quad_points = field->GetTotPoints();
    Array<OneD, NekDouble> phys_correct(tot_quad_points);
    Array<OneD, NekDouble> phys_projected_u(tot_quad_points);
    phys_projected_u = field->GetPhys();
    interpolate_onto_nektar_field_3d(func, field);
    phys_correct = field->GetPhys();

    const double err = field->L2(phys_projected_u, phys_correct);

    A->free();
    mesh->free();

    delete[] argv[0];
    delete[] argv[1];
    delete[] argv[2];
    return err;
  };

  const int Nsample = 4;

  std::vector<int> Nparticles;
  Nparticles.push_back(200000);
  Nparticles.push_back(800000);

  std::map<int, std::vector<double>> R_errors;

  for (auto N_total : Nparticles) {
    for (int samplex = 0; samplex < Nsample; samplex++) {
      const double err = project_run(N_total, samplex);
      R_errors[N_total].push_back(err);
    }
  }

  auto lambda_average = [=](std::vector<double> errors) {
    double avg = 0.0;
    for (auto &err : errors) {
      avg += err;
    }
    avg /= errors.size();
    return avg;
  };

  const double err_0 = lambda_average(R_errors[Nparticles[0]]);
  const double err_1 = lambda_average(R_errors[Nparticles[1]]);

  const double order_abs = ABS((err_0 / err_1));
  ASSERT_NEAR(order_abs, 2.0, 0.075);
}

TEST(ParticleFunctionProjectionOrder3D, DisContFieldHex) {
  projection_wrapper_order_3d<MultiRegions::DisContField>(
      "reference_hex_cube/conditions_nummodes_4.xml",
      "reference_hex_cube/hex_cube_0.5.xml");
}
TEST(ParticleFunctionProjectionOrder3D, DisContFieldPrismTet) {
  projection_wrapper_order_3d<MultiRegions::DisContField>(
      "reference_prism_tet_cube/conditions_nummodes_4.xml",
      "reference_prism_tet_cube/prism_tet_cube_0.5.xml");
}
TEST(ParticleFunctionProjectionOrder3D, ContFieldHex) {
  projection_wrapper_order_3d<MultiRegions::DisContField>(
      "reference_hex_cube/conditions_cg_nummodes_4.xml",
      "reference_hex_cube/hex_cube_0.5.xml");
}
TEST(ParticleFunctionProjectionOrder3D, ContFieldPrismTet) {
  projection_wrapper_order_3d<MultiRegions::DisContField>(
      "reference_prism_tet_cube/conditions_cg_nummodes_4.xml",
      "reference_prism_tet_cube/prism_tet_cube_0.5.xml");
}
