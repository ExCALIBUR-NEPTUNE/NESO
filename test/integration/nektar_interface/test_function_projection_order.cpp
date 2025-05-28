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

using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::SpatialDomains;
using namespace Nektar::MultiRegions;
using namespace NESO::Particles;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

TEST(ParticleFunctionProjection, DisContScalarExpQuantity) {

  auto project_run = [&](const int N_total, const int samplex) {
    int argc = 3;
    char *argv[3];

    std::filesystem::path source_file = __FILE__;
    std::filesystem::path source_dir = source_file.parent_path();
    std::filesystem::path test_resources_dir =
        source_dir / "../../test_resources";
    std::filesystem::path mesh_file =
        test_resources_dir / "square_triangles_quads_nummodes_2.xml";
    std::filesystem::path conditions_file =
        test_resources_dir / "conditions.xml";

    copy_to_cstring(std::string("test_particle_function_projection"), &argv[0]);
    copy_to_cstring(std::string(mesh_file), &argv[1]);
    copy_to_cstring(std::string(conditions_file), &argv[2]);

    LibUtilities::SessionReaderSharedPtr session;
    SpatialDomains::MeshGraphSharedPtr graph;
    // Create session reader.
    session = LibUtilities::SessionReader::CreateInstance(argc, argv);
    graph = SpatialDomains::MeshGraphIO::Read(session);

    auto dis_cont_field = std::make_shared<DisContField>(session, graph, "u");

    auto mesh = std::make_shared<ParticleMeshInterface>(graph);
    auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

    auto nektar_graph_local_mapper =
        std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh);

    auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

    const int ndim = 2;
    ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                               ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                               ParticleProp(Sym<REAL>("Q"), 1)};

    auto A =
        std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

    NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
    auto cell_id_translation =
        std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

    const int rank = sycl_target->comm_pair.rank_parent;
    const int size = sycl_target->comm_pair.size_parent;

    std::mt19937 rng_pos(52234234 + samplex);

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

    particle_loop(
        A,
        [=](auto k_P, auto k_Q) {
          const REAL x = k_P.at(0);
          const REAL y = k_P.at(1);
          const REAL exp_eval =
              two_over_sqrt_pi * sycl::exp(-(4.0 * ((x) * (x) + (y) * (y))));
          k_Q.at(0) = exp_eval * reweight;
        },
        Access::read(Sym<REAL>("P")), Access::write(Sym<REAL>("Q")))
        ->execute();

    // create projection object
    auto field_project = std::make_shared<FieldProject<DisContField>>(
        dis_cont_field, A, cell_id_translation);

    // evaluate field at particle locations
    field_project->project(Sym<REAL>("Q"));

    const int tot_quad_points = dis_cont_field->GetTotPoints();
    Array<OneD, NekDouble> phys_projected(tot_quad_points);
    Array<OneD, NekDouble> phys_correct(tot_quad_points);

    phys_projected = dis_cont_field->GetPhys();

    // H5Part h5part("exp.h5part", A, Sym<REAL>("P"), Sym<INT>("NESO_MPI_RANK"),
    //               Sym<REAL>("NESO_REFERENCE_POSITIONS"), Sym<REAL>("Q"));
    // h5part.write();
    // h5part.close();

    // write_vtu(dis_cont_field, "func_projected.vtu", "u");

    auto lambda_f = [&](const NekDouble x, const NekDouble y) {
      const REAL two_over_sqrt_pi = 1.1283791670955126;
      const REAL exp_eval =
          two_over_sqrt_pi * exp(-(4.0 * ((x) * (x) + (y) * (y))));
      return exp_eval;
    };
    interpolate_onto_nektar_field_2d(lambda_f, dis_cont_field);
    phys_correct = dis_cont_field->GetPhys();

    // write_vtu(dis_cont_field, "func_correct.vtu", "u");

    const double err = dis_cont_field->L2(phys_projected, phys_correct);

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
      // nprint(N_total, err);
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

  // nprint(err_0, err_1);
  // nprint(err_0 / err_1);

  ASSERT_NEAR(ABS(err_0 / err_1), 2.0, 0.075);
}

TEST(ParticleFunctionProjection, ContScalarExpQuantity) {

  auto project_run = [&](const int N_total, const int samplex) {
    int argc = 3;
    char *argv[3];

    std::filesystem::path source_file = __FILE__;
    std::filesystem::path source_dir = source_file.parent_path();
    std::filesystem::path test_resources_dir =
        source_dir / "../../test_resources";
    std::filesystem::path mesh_file =
        test_resources_dir / "square_triangles_quads_nummodes_2.xml";
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

    auto cont_field = std::make_shared<ContField>(session, graph, "u");

    auto mesh = std::make_shared<ParticleMeshInterface>(graph);
    auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

    auto nektar_graph_local_mapper =
        std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh);

    auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

    const int ndim = 2;
    ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                               ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                               ParticleProp(Sym<REAL>("Q"), 1)};

    auto A =
        std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

    NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
    auto cell_id_translation =
        std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

    const int rank = sycl_target->comm_pair.rank_parent;
    const int size = sycl_target->comm_pair.size_parent;

    std::mt19937 rng_pos(52234234 + samplex);

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

    particle_loop(
        A,
        [=](auto k_P, auto k_Q) {
          const REAL x = k_P.at(0);
          const REAL y = k_P.at(1);
          const REAL exp_eval =
              two_over_sqrt_pi * sycl::exp(-(4.0 * ((x) * (x) + (y) * (y))));
          k_Q.at(0) = exp_eval * reweight;
        },
        Access::read(Sym<REAL>("P")), Access::write(Sym<REAL>("Q")))
        ->execute();

    // create projection object
    auto field_project = std::make_shared<FieldProject<ContField>>(
        cont_field, A, cell_id_translation);

    // evaluate field at particle locations
    field_project->project(Sym<REAL>("Q"));

    const int tot_quad_points = cont_field->GetTotPoints();
    Array<OneD, NekDouble> phys_projected(tot_quad_points);
    Array<OneD, NekDouble> phys_correct(tot_quad_points);

    phys_projected = cont_field->GetPhys();

    // H5Part h5part("exp.h5part", A, Sym<REAL>("P"), Sym<INT>("NESO_MPI_RANK"),
    //               Sym<REAL>("NESO_REFERENCE_POSITIONS"), Sym<REAL>("Q"));
    // h5part.write();
    // h5part.close();

    // write_vtu(cont_field, "func_projected_" + std::to_string(rank) +
    // "_0.vtu", "u"); nprint("project integral:", cont_field->Integral(),
    // 0.876184);

    auto lambda_f = [&](const NekDouble x, const NekDouble y) {
      const REAL two_over_sqrt_pi = 1.1283791670955126;
      const REAL exp_eval =
          two_over_sqrt_pi * exp(-(4.0 * ((x) * (x) + (y) * (y))));
      return exp_eval;
    };
    interpolate_onto_nektar_field_2d(lambda_f, cont_field);
    phys_correct = cont_field->GetPhys();

    // write_vtu(cont_field, "func_correct_" + std::to_string(rank) + "_0.vtu",
    // "u"); nprint("correct integral:", cont_field->Integral(), 0.876184);

    const double err = cont_field->L2(phys_projected, phys_correct);

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
      // nprint(N_total, err);
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

  // nprint(err_0, err_1);
  // nprint(err_0 / err_1);

  ASSERT_NEAR(ABS(err_0 / err_1), 2.0, 0.075);
}

TEST(ParticleFunctionProjection, ContScalarExpQuantityMultiple) {

  auto project_run = [&](const int N_total, const int samplex, double *err) {
    int argc = 3;
    char *argv[3];

    std::filesystem::path source_file = __FILE__;
    std::filesystem::path source_dir = source_file.parent_path();
    std::filesystem::path test_resources_dir =
        source_dir / "../../test_resources";
    std::filesystem::path mesh_file =
        test_resources_dir / "square_triangles_quads_nummodes_2.xml";
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

    auto A =
        std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

    NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
    auto cell_id_translation =
        std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

    const int rank = sycl_target->comm_pair.rank_parent;
    const int size = sycl_target->comm_pair.size_parent;

    std::mt19937 rng_pos(52234234 + samplex);

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

    particle_loop(
        A,
        [=](auto k_P, auto k_Q, auto k_Q2) {
          const REAL x = k_P.at(0);
          const REAL y = k_P.at(1);
          const REAL exp_eval =
              two_over_sqrt_pi * sycl::exp(-(4.0 * ((x) * (x) + (y) * (y))));
          k_Q.at(0) = exp_eval * reweight;
          k_Q.at(1) = -1.0 * exp_eval * reweight;
          k_Q2.at(1) = reweight;
        },
        Access::read(Sym<REAL>("P")), Access::write(Sym<REAL>("Q")),
        Access::write(Sym<REAL>("Q2")))
        ->execute();

    auto cont_field_u = std::make_shared<ContField>(session, graph, "u");
    auto cont_field_v = std::make_shared<ContField>(session, graph, "v");
    auto cont_field_n = std::make_shared<ContField>(session, graph, "n");

    std::vector<std::shared_ptr<ContField>> cont_fields = {
        cont_field_u, cont_field_v, cont_field_n};
    // create projection object
    auto field_project = std::make_shared<FieldProject<ContField>>(
        cont_fields, A, cell_id_translation);

    // const double err = 1.0;

    // project field at particle locations
    std::vector<Sym<REAL>> project_syms = {Sym<REAL>("Q"), Sym<REAL>("Q"),
                                           Sym<REAL>("Q2")};
    std::vector<int> project_components = {0, 1, 1};

    if (samplex == 0) {
      field_project->testing_enable();
    }
    field_project->project(project_syms, project_components);
    if (samplex == 0) {
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
    }

    const int tot_quad_points = cont_field_u->GetTotPoints();
    Array<OneD, NekDouble> phys_correct(tot_quad_points);
    Array<OneD, NekDouble> phys_projected_u(tot_quad_points);
    Array<OneD, NekDouble> phys_projected_v(tot_quad_points);
    Array<OneD, NekDouble> phys_projected_n(tot_quad_points);

    phys_projected_u = cont_field_u->GetPhys();
    phys_projected_v = cont_field_v->GetPhys();
    phys_projected_n = cont_field_n->GetPhys();

    // write_vtu(cont_field_u,
    //           "func_projected_u_" + std::to_string(rank) + "_0.vtu", "u");
    // write_vtu(cont_field_v,
    //           "func_projected_v_" + std::to_string(rank) + "_0.vtu", "v");
    // write_vtu(cont_field_n,
    //           "func_projected_n_" + std::to_string(rank) + "_0.vtu", "n");

    for (int cx = 0; cx < tot_quad_points; cx++) {
      ASSERT_NEAR(phys_projected_u[cx], phys_projected_v[cx] * -1.0, 1.0e-2);

      // This bound is huge so there is also an L2 error norm test below.
      ASSERT_NEAR(phys_projected_n[cx], 1.0, 0.4);
    }

    // H5Part h5part("exp.h5part", A, Sym<REAL>("P"), Sym<INT>("NESO_MPI_RANK"),
    //               Sym<REAL>("NESO_REFERENCE_POSITIONS"), Sym<REAL>("Q"));
    // h5part.write();
    // h5part.close();
    // nprint("project integral:", cont_field_u->Integral(),
    // 0.876184);

    auto lambda_f = [&](const NekDouble x, const NekDouble y) {
      const REAL two_over_sqrt_pi = 1.1283791670955126;
      const REAL exp_eval =
          two_over_sqrt_pi * exp(-(4.0 * ((x) * (x) + (y) * (y))));
      return exp_eval;
    };
    interpolate_onto_nektar_field_2d(lambda_f, cont_field_u);
    phys_correct = cont_field_u->GetPhys();

    // write_vtu(cont_field_u, "func_correct_" + std::to_string(rank) +
    // "_0.vtu", "u"); nprint("correct integral:", cont_field_u->Integral(),
    // 0.876184);

    *err = cont_field_u->L2(phys_projected_u, phys_correct);

    for (int cx = 0; cx < tot_quad_points; cx++) {
      phys_correct[cx] = 1.0;
    }

    const double err2 = cont_field_n->L2(phys_projected_n, phys_correct);
    ASSERT_TRUE(err2 < 0.15);

    mesh->free();

    delete[] argv[0];
    delete[] argv[1];
    delete[] argv[2];
  };

  const int Nsample = 4;

  std::vector<int> Nparticles;
  Nparticles.push_back(200000);
  Nparticles.push_back(800000);

  std::map<int, std::vector<double>> R_errors;

  for (auto N_total : Nparticles) {
    for (int samplex = 0; samplex < Nsample; samplex++) {
      double err = -1;
      project_run(N_total, samplex, &err);
      // nprint(N_total, err);
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

  // nprint(err_0, err_1);
  // nprint(err_0 / err_1);

  ASSERT_NEAR(ABS(err_0 / err_1), 2.0, 0.075);
}
