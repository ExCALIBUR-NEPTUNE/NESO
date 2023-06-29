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
using namespace Nektar::MultiRegions;
using namespace NESO::Particles;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

TEST(JacobiCoeffModBasis, Coeff) {

  ASSERT_NEAR(jacobi(5, 0.3, 3, 4), 1.5229637499999997, 1.0e-10);
  ASSERT_NEAR(jacobi(13, 0.3, 7, 11), -9.5066868221006, 1.0e-10);
  ASSERT_NEAR(jacobi(13, -0.5, 7, 11), -47.09590101242081, 1.0e-10);

  JacobiCoeffModBasis jacobi_coeff(13, 7);

  ASSERT_NEAR(jacobi(13, -0.5, 7, 1), jacobi_coeff.host_evaluate(13, 7, -0.5),
              1.0e-10);
  ASSERT_NEAR(jacobi(0, -0.5, 7, 1), 1.0, 1.0e-10);
  ASSERT_NEAR(jacobi(7, 0.5, 3, 1), jacobi_coeff.host_evaluate(7, 3, 0.5),
              1.0e-10);
}

TEST(ParticleFunctionBasisEvaluation, DisContFieldScalar) {

  const int N_total = 2000;
  const double tol = 1.0e-10;
  int argc = 3;
  char *argv[3];

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";
  std::filesystem::path mesh_file =
      test_resources_dir / "square_triangles_quads_nummodes_6.xml";
  std::filesystem::path conditions_file = test_resources_dir / "conditions.xml";

  copy_to_cstring(std::string("test_particle_function_evaluation"), &argv[0]);
  copy_to_cstring(std::string(mesh_file), &argv[1]);
  copy_to_cstring(std::string(conditions_file), &argv[2]);

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
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<REAL>("FUNC_EVALS"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(2234234 + rank);

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

  auto basis_evaluate_base =
      std::make_shared<FunctionEvaluateBasis<DisContField>>(
          dis_cont_field, mesh, cell_id_translation);

  auto coeffs = dis_cont_field->GetCoeffs();

  // auto t0 = profile_timestamp();
  basis_evaluate_base->evaluate(A, Sym<REAL>("FUNC_EVALS"), 0, coeffs);
  // nprint("time taken:", profile_elapsed(t0, profile_timestamp()));

  // check evaluations
  for (int cellx = 0; cellx < cell_count; cellx++) {

    auto positions = A->position_dat->cell_dat.get_cell(cellx);
    auto func_evals = (*A)[Sym<REAL>("FUNC_EVALS")]->cell_dat.get_cell(cellx);

    for (int rowx = 0; rowx < positions->nrow; rowx++) {

      const double x = (*positions)[0][rowx];
      const double y = (*positions)[1][rowx];

      const double eval_dat = (*func_evals)[0][rowx];
      const double eval_correct = evaluate_scalar_2d(dis_cont_field, x, y);
      const double err = ABS(eval_correct - eval_dat);

      EXPECT_NEAR(eval_correct, eval_dat, 1.0e-8);
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

TEST(ParticleFunctionBasisEvaluation, ContFieldScalar) {

  const int N_total = 2000;
  const double tol = 1.0e-10;
  int argc = 3;
  char *argv[3];

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";
  std::filesystem::path mesh_file =
      test_resources_dir / "square_triangles_quads_nummodes_6.xml";
  std::filesystem::path conditions_file =
      test_resources_dir / "conditions_cg.xml";

  copy_to_cstring(std::string("test_particle_function_evaluation"), &argv[0]);
  copy_to_cstring(std::string(mesh_file), &argv[1]);
  copy_to_cstring(std::string(conditions_file), &argv[2]);

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);
  graph = SpatialDomains::MeshGraph::Read(session);

  auto cont_field = std::make_shared<ContField>(session, graph, "u");

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapperT>(sycl_target, mesh, tol);

  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = 2;
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<REAL>("FUNC_EVALS"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(2234234 + rank);

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

  interpolate_onto_nektar_field_2d(lambda_f, cont_field);

  auto basis_evaluate_base = std::make_shared<FunctionEvaluateBasis<ContField>>(
      cont_field, mesh, cell_id_translation);

  auto coeffs = cont_field->GetCoeffs();

  // auto t0 = profile_timestamp();
  basis_evaluate_base->evaluate(A, Sym<REAL>("FUNC_EVALS"), 0, coeffs);
  // nprint("time taken:", profile_elapsed(t0, profile_timestamp()));

  // check evaluations
  for (int cellx = 0; cellx < cell_count; cellx++) {

    auto positions = A->position_dat->cell_dat.get_cell(cellx);
    auto func_evals = (*A)[Sym<REAL>("FUNC_EVALS")]->cell_dat.get_cell(cellx);

    for (int rowx = 0; rowx < positions->nrow; rowx++) {

      const double x = (*positions)[0][rowx];
      const double y = (*positions)[1][rowx];

      const double eval_dat = (*func_evals)[0][rowx];
      const double eval_correct = evaluate_scalar_2d(cont_field, x, y);
      const double err = ABS(eval_correct - eval_dat);

      ASSERT_NEAR(eval_correct, eval_dat, 1.0e-8);
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

TEST(ParticleFunctionBasisEvaluation, Basis3D) {

  std::tuple<std::string, std::string, double> param = {
      "reference_all_types_cube/conditions.xml",
      "reference_all_types_cube/mixed_ref_cube_0.5_perturbed.xml", 2.0e-4};

  const int N_total = 2000;
  const double tol = std::get<2>(param);

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";

  std::filesystem::path condtions_file_basename =
      static_cast<std::string>(std::get<0>(param));
  std::filesystem::path mesh_file_basename =
      static_cast<std::string>(std::get<1>(param));
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

  auto cont_field = std::make_shared<ContField>(session, graph, "u");

  auto lambda_func = [](const double x, const double y, const double z) {
    return sin(x) * cos(y) + exp(z);
  };

  interpolate_onto_nektar_field_3d(lambda_func, cont_field);

  auto lambda_get_name = [&](Nektar::LibUtilities::BasisType type) {
    if (type == eModified_A) {
      return "eModified_A";
    } else if (type == eModified_B) {
      return "eModified_B";
    } else if (type == eModified_C) {
      return "eModified_C";
    } else if (type == eModifiedPyr_C) {
      return "eModifiedPyr_C";
    } else {
      return "oh dear";
    }
  };

  auto lambda_eval_basis = [&](Nektar::LibUtilities::BasisType type,
                               const int num_modes, const double z,
                               std::vector<double> &o) {
    if (type == eModified_A) {
      for (int p = 0; p < num_modes; p++) {
        o.at(p) = eval_modA_i(p, z);
      }
      return 0;
    } else if (type == eModified_B) {
      int mode = 0;
      for (int p = 0; p < num_modes; p++) {
        for (int q = 0; q < (num_modes - p); q++) {
          o.at(mode) = eval_modB_ij(p, q, z);
          mode++;
        }
      }
      return 0;
    } else if (type == eModified_C) {
      int mode = 0;
      for (int p = 0; p < num_modes; p++) {
        for (int q = 0; q < (num_modes - p); q++) {
          for (int r = 0; r < (num_modes - p - q); r++) {
            o.at(mode) = eval_modC_ijk(p, q, r, z);
            mode++;
          }
        }
      }
      return 0;
    } else if (type == eModifiedPyr_C) {
      int mode = 0;
      for (int p = 0; p < num_modes; p++) {
        for (int q = 0; q < (num_modes); q++) {
          for (int r = 0; r < (num_modes - std::max(p, q)); r++) {
            o.at(mode) = eval_modPyrC_ijk(p, q, r, z);
            // nprint(mode, "p", p, "q", q, "r", r, ":", o.at(mode));
            mode++;
          }
        }
      }
      return 1;
    } else {
      o.at(0) = -999999.0;
      return 0;
    }
  };


  double max_err = 0.0;
  for (int ei = 0; ei < cont_field->GetNumElmts(); ei++) {
    auto ex = cont_field->GetExp(ei);

    // test the basis evaluation in each direction
    for (int dimx = 0; dimx < 3; dimx++) {
      auto b0 = ex->GetBasis(dimx);

      auto bb0 = b0->GetBdata();
      auto zb0 = b0->GetZ();
      auto nb0 = b0->GetNumModes();
      auto mb0 = b0->GetNumPoints();

      auto total_num_modes = b0->GetTotNumModes();

      std::vector<double> to_test(total_num_modes);

      for (int m = 0; m < mb0; m++) {
        const double zz = zb0[m];
        if (lambda_eval_basis(b0->GetBasisType(), nb0, zz, to_test)) {
          for (int mx = 0; mx < total_num_modes; mx++) {
            ASSERT_TRUE((mx * mb0 + m) < bb0.size());
            const double zc = bb0[mx * mb0 + m];
            const double tt = to_test.at(mx);
            // const std::string msg = (abs(tt - zc) < 1.0e-4) ? " " : "NAY";
            // nprint(mx, zz, zc, tt, msg);
            ASSERT_NEAR(zc, tt, 1.0e-10);
          }
        }
      }
    }
    // test the computation of each overall mode
    auto shape = ex->DetShapeType();
    auto geom = ex->GetGeom();
    auto bounding_box = geom->GetBoundingBox();
    std::uniform_real_distribution<double> uniform_dist0(bounding_box[0],
                                                         bounding_box[0] + 3);
    std::uniform_real_distribution<double> uniform_dist1(bounding_box[1],
                                                         bounding_box[1] + 3);
    std::uniform_real_distribution<double> uniform_dist2(bounding_box[2],
                                                         bounding_box[2] + 3);

    Array<OneD, NekDouble> local_collapsed(3);
    Array<OneD, NekDouble> local_coord(3);
    Array<OneD, NekDouble> global_coord(3);

    // if (shape != eTetrahedron) {
    //   continue;
    // }

    for (int testx = 0; testx < 1; testx++) {

      bool is_contained = false;
      while (!is_contained) {

        global_coord[0] = uniform_dist0(rng);
        global_coord[1] = uniform_dist1(rng);
        global_coord[2] = uniform_dist2(rng);
        is_contained =
            ex->GetGeom()->ContainsPoint(global_coord, local_coord, 1.0e-8);
      }
      ex->LocCoordToLocCollapsed(local_coord, local_collapsed);

      nprint("collapsed:", local_collapsed[0], local_collapsed[1],
             local_collapsed[2]);
      nprint("local:", local_coord[0], local_coord[1], local_coord[2]);

      int num_modes[3];

      const int num_coeffs = ex->GetNcoeffs();
      std::vector<std::vector<double>> evals = {
          std::vector<double>(num_coeffs), std::vector<double>(num_coeffs),
          std::vector<double>(num_coeffs)};

      std::vector<double> mode_evals(num_coeffs);
      std::vector<double> mode_evals_basis(num_coeffs);
      std::vector<double> mode_correct(num_coeffs);

      for (int dimx = 0; dimx < 3; dimx++) {
        auto basis = ex->GetBasis(dimx);
        nprint(lambda_get_name(basis->GetBasisType()));
        const int num_modes_basis = basis->GetNumModes();
        const double z = local_collapsed[dimx];
        lambda_eval_basis(basis->GetBasisType(), num_modes_basis, z,
                          evals[dimx]);
        num_modes[dimx] = num_modes_basis;
        nprint("modes:", basis->GetNumModes(), basis->GetTotNumModes());
      }

      const int P = num_modes[0];
      const int Q = num_modes[1];
      const int R = num_modes[2];

      nprint("P", P, "Q", Q, "R", R);

      if (shape == eHexahedron) {
        nprint("Hex");

        int mode = 0;
        for (int mz = 0; mz < num_modes[2]; mz++) {
          for (int my = 0; my < num_modes[1]; my++) {
            for (int mx = 0; mx < num_modes[0]; mx++) {
              mode_evals[mode] = evals[0][mx] * evals[1][my] * evals[2][mz];
              mode_evals_basis[mode] = evals[0][mx] * evals[1][my] * evals[2][mz];
              mode++;
            }
          }
        }

      } else if (shape == ePyramid) {
        nprint("Pyramid");

        int mode = 0;
        for (int p = 0; p < P; ++p) {
          for (int q = 0; q < Q; ++q) {
            int maxpq = max(p, q);
            for (int r = 0; r < R - maxpq; ++r, ++mode) {
              // const double contrib_0 = evals[0][p];
              // const double contrib_1 = evals[1][q];
              // const double contrib_2 = evals[2][r];

              const double contrib_0 = eval_modA_i(p, local_collapsed[0]);
              const double contrib_1 = eval_modA_i(q, local_collapsed[1]);
              const double contrib_2 = eval_modA_i(r, local_collapsed[2]);

              if (mode == 1) {
                mode_evals[mode] = contrib_2;
              } else {
                mode_evals[mode] = contrib_0 * contrib_1 * contrib_2;
              }
            }
          }
        }

        mode = 0;
        for (int p = 0; p < P; ++p) {
          for (int q = 0; q < Q; ++q) {
            int maxpq = max(p, q);
            for (int r = 0; r < R - maxpq; ++r, ++mode) {
              const double contrib_0 = eval_modA_i(p, local_collapsed[0]);
              const double contrib_1 = eval_modA_i(q, local_collapsed[1]);
              const double contrib_2 =
                  eval_modPyrC_ijk(p, q, r, local_collapsed[2]);
              if (mode == 1) {
                mode_evals_basis[mode] = contrib_2;
              } else {
                mode_evals_basis[mode] = contrib_0 * contrib_1 * contrib_2;
              }
            }
          }
        }

      } else if (shape == ePrism) {
        nprint("Prism");

        int mode = 0;
        int mode_pr = 0;
        for (int p = 0; p < P; p++) {
          for (int q = 0; q < P; q++) {
            for (int r = 0; r < (P - p); r++) {
              // const double contrib_0 = evals[0][p];
              // const double contrib_1 = evals[1][q];
              // const double contrib_2 = evals[2][r];

              const double contrib_0 = eval_modA_i(p, local_collapsed[0]);
              const double contrib_1 = eval_modA_i(q, local_collapsed[1]);
              const double contrib_2 = eval_modA_i(r, local_collapsed[2]);

              mode_evals[mode] = contrib_0 * contrib_1 * contrib_2;

              if ((p == 0) && (r == 1)) {
                mode_evals[mode] /= eval_modA_i(p, local_collapsed[0]);
              }
              mode++;
            }
          }
          mode_pr += P - p;
        }

        mode = 0;
        mode_pr = 0;
        for (int p = 0; p < P; p++) {
          for (int q = 0; q < P; q++) {
            for (int r = 0; r < (P - p); r++) {

              const double contrib_0 = eval_modA_i(p, local_collapsed[0]);
              const double contrib_1 = eval_modA_i(q, local_collapsed[1]);
              const double contrib_2 = eval_modB_ij(p, r, local_collapsed[2]);

              mode_evals_basis[mode] = contrib_0 * contrib_1 * contrib_2;

              if ((p == 0) && (r == 1)) {
                mode_evals_basis[mode] /= eval_modA_i(p, local_collapsed[0]);
              }
              mode++;
            }
          }
          mode_pr += P - p;
        }

      } else if (shape == eTetrahedron) {

        nprint("Tetrahedron", num_modes[0], num_modes[1], num_modes[2]);

        auto lambda_get_mode = [&](const int I, const int J, const int K) {
          const int Q = num_modes[1];
          const int R = num_modes[2];

          int i, j, q_hat, k_hat;
          int cnt = 0;

          // Traverse to q-r plane number I
          for (i = 0; i < I; ++i) {
            // Size of triangle part
            q_hat = min(Q, R - i);
            // Size of rectangle part
            k_hat = max(R - Q - i, 0);
            cnt += q_hat * (q_hat + 1) / 2 + k_hat * Q;
          }

          // Traverse to q column J
          q_hat = R - I;
          for (j = 0; j < J; ++j) {
            cnt += q_hat;
            q_hat--;
          }
          // Traverse up stacks to K
          cnt += K;
          return cnt;
        };

        auto lambda_mode_to_modes = [&](const int mode) {
          const int nm1 = Q;
          const int nm2 = R;

          const int b = 2 * nm2 + 1;
          const int mode0 = floor(0.5 * (b - sqrt(b * b - 8.0 * mode / nm1)));
          const int tmp = mode - nm1 * (mode0 * (nm2 - 1) + 1 -
                                        (mode0 - 2) * (mode0 - 1) / 2);

          const int mode1 = tmp / (nm2 - mode0);
          const int mode2 = tmp % (nm2 - mode0);

          // nprint("n->", mode, mode0, mode1, mode2, lambda_get_mode(mode0,
          // mode1, mode2));
          //  nprint("n->", mode, mode0, mode1, mode2);
        };

        for (int mode = 0; mode < num_coeffs; mode++) {
          lambda_mode_to_modes(mode);
        }

        int mode = 0;
        for (int p = 0; p < 2 && (mode < num_coeffs); p++) {
          for (int q = 0; q < (P - p) && (mode < num_coeffs); q++) {
            for (int r = 0; r < (P - p) && (mode < num_coeffs); r++) {
              // const double contrib_0 = evals[0][p];
              // const double contrib_1 = evals[1][q];
              // const double contrib_2 = evals[2][r];

              const double contrib_0 = eval_modA_i(p, local_collapsed[0]);
              const double contrib_1 = eval_modA_i(q, local_collapsed[1]);
              const double contrib_2 = eval_modA_i(r, local_collapsed[2]);

              double eval;

              if (mode == 1) {
                // StdExpansion::BaryEvaluateBasis<2>(coll[2], 1);
                eval = evals[2][1];
              } else if (p == 0 && r == 1) {
                // StdExpansion::BaryEvaluateBasis<1>(coll[1], 0) *
                // StdExpansion::BaryEvaluateBasis<2>(coll[2], 1);
                eval = evals[1][0] * evals[2][1];
              } else if (p == 1 && q == 1 && r == 0) {
                // StdExpansion::BaryEvaluateBasis<0>(coll[0], 0) *
                // StdExpansion::BaryEvaluateBasis<1>(coll[1], 1);
                eval = evals[0][0] * evals[1][1];
              } else {
                eval = contrib_0 * contrib_1 * contrib_2;
              }

              mode_evals[mode] = eval;
              // nprint("e->", mode, p, q, r, eval);
              // lambda_mode_to_modes(mode);

              mode++;
            }
          }
        }

        mode = 0;
        for (int p = 0; p < P && (mode < num_coeffs); p++) {
          for (int q = 0; q < (P - p) && (mode < num_coeffs); q++) {
            for (int r = 0; r < (P - p - q) && (mode < num_coeffs); r++) {
              // const double contrib_0 = evals[0][p];
              // const double contrib_1 = evals[1][q];
              // const double contrib_2 = evals[2][r];

              const double contrib_0 = eval_modA_i(p, local_collapsed[0]);
              const double contrib_1 = eval_modB_ij(p, q, local_collapsed[1]);
              const double contrib_2 =
                  eval_modC_ijk(p, q, r, local_collapsed[2]);

              double eval;

              if (mode == 1) { // This seems correct.
                eval = contrib_2;
              } else if (p == 0 && q == 1) {
                eval = contrib_1 * contrib_2;
              } else {
                eval = contrib_0 * contrib_1 * contrib_2;
              }

              mode_evals_basis[mode] = eval;
              // lambda_mode_to_modes(mode);

              mode++;
            }
          }
        }

      } else {
        nprint("other");
      }

      auto coeffs_global = cont_field->GetCoeffs();
      auto phys_global = cont_field->GetPhys();

      auto coeffs = coeffs_global + cont_field->GetCoeff_Offset(ei);
      auto phys = phys_global + cont_field->GetPhys_Offset(ei);

      // evaluation using modes
      double eval_modes = 0.0;
      for (int modex = 0; modex < num_coeffs; modex++) {
        const double mode_tmp = ex->PhysEvaluateBasis(local_coord, modex);
        const double dof_tmp = coeffs[modex];
        eval_modes += dof_tmp * mode_tmp;
      }

      // get the correct modes
      for (int modex = 0; modex < num_coeffs; modex++) {
        const double mode_tmp = ex->PhysEvaluateBasis(local_coord, modex);
        mode_correct[modex] = mode_tmp;
      }
      // check computed modes match Nektar++ modes
      for (int modex = 0; modex < num_coeffs; modex++) {
        const double err = abs(mode_correct[modex] - mode_evals[modex]);
        // nprint(modex, err, mode_correct[modex], mode_evals[modex]);
        ASSERT_TRUE(err < 1.0e-10);
      }

      // get field evaluation at point
      const double eval_bary = evaluate_scalar_3d(
          cont_field, global_coord[0], global_coord[1], global_coord[2]);

      // avoid the id == 0 assert bug
      if (ei > 0) {
        const double eval_bary_global =
            cont_field->PhysEvaluate(global_coord, phys_global);
        ASSERT_NEAR(eval_bary, eval_bary_global, 1.0e-10);
      }

      /**
       *  This fails i.e. \sum_{m in modes} \phi_{m}(x) * coeff_{m} !=
       * PhysEvaluate(x)
       *
       */
      const double eval_err = abs(eval_bary - eval_modes);
      nprint("EVAL ERR:", eval_err);
      // ASSERT_TRUE(eval_err < 1.0e-10);

      const int global_num_coeffs = coeffs_global.size();
      const int global_num_phys = phys_global.size();

      Array<OneD, NekDouble> basis_coeffs(global_num_coeffs);
      Array<OneD, NekDouble> basis_phys(global_num_phys);

      auto lambda_zero = [&]() {
        for (int cx = 0; cx < global_num_coeffs; cx++) {
          basis_coeffs[cx] = 0.0;
        }
        for (int cx = 0; cx < global_num_phys; cx++) {
          basis_phys[cx] = 0.0;
        }
      };

      const int offset_coeff = cont_field->GetCoeff_Offset(ei);
      const int offset_phys = cont_field->GetPhys_Offset(ei);
      Array<OneD, NekDouble> mode_via_coeffs(mode_evals.size());

      for (int modex = 0; modex < num_coeffs; modex++) {
        // set the dof for the mode to 1 and the rest to 0
        lambda_zero();
        basis_coeffs[offset_coeff + modex] = 1.0;
        // convert to quadrature point values
        cont_field->BwdTrans(basis_coeffs, basis_phys);
        mode_via_coeffs[modex] =
            ex->StdPhysEvaluate(local_coord, basis_phys + offset_phys);
      }

      double test_eval = 0.0;
      for (int modex = 0; modex < num_coeffs; modex++) {
        test_eval += mode_via_coeffs[modex] * coeffs[modex];
      }
      const double eval_basis_err = abs(eval_bary - test_eval);
      nprint("EVAL BASIS ERR:", eval_basis_err);
      // ASSERT_TRUE(eval_basis_err < 1.0e-10);
      //
      for (int modex = 0; modex < num_coeffs; modex++) {
        const double dof_err =
            abs(mode_via_coeffs[modex] - mode_evals_basis[modex]);
        nprint(modex, dof_err, " |\t", mode_via_coeffs[modex],
               mode_evals_basis[modex]);
        max_err = max(max_err, dof_err);
        ASSERT_TRUE(dof_err < 1.0e-5);
      }
    }
  }

  nprint("maxerr", max_err);

  mesh->free();
  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}
