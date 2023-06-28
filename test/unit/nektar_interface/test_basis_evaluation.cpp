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

    Array<OneD, NekDouble> local_coord(3);
    local_coord[0] = -0.2123;
    local_coord[1] = -0.3689;
    local_coord[2] = -0.436;

    Array<OneD, NekDouble> local_collapsed(3);
    ex->LocCoordToLocCollapsed(local_coord, local_collapsed);

    nprint("collapsed:", local_collapsed[0], local_collapsed[1],
           local_collapsed[2]);

    int num_modes[3];

    const int num_coeffs = ex->GetNcoeffs();
    std::vector<std::vector<double>> evals = {std::vector<double>(num_coeffs),
                                              std::vector<double>(num_coeffs),
                                              std::vector<double>(num_coeffs)};

    std::vector<double> mode_evals(num_coeffs);
    std::vector<double> mode_correct(num_coeffs);

    for (int dimx = 0; dimx < 3; dimx++) {
      auto basis = ex->GetBasis(dimx);
      nprint(lambda_get_name(basis->GetBasisType()));
      const int num_modes_basis = basis->GetNumModes();
      const double z = local_collapsed[dimx];
      lambda_eval_basis(basis->GetBasisType(), num_modes_basis, z, evals[dimx]);
      num_modes[dimx] = num_modes_basis;
      nprint("modes:", basis->GetNumModes(), basis->GetTotNumModes());
    }

    bool printo = true;

    const int P = num_modes[0];
    const int Q = num_modes[1];
    const int R = num_modes[2];

    nprint("P", P, "Q", Q, "R", R);

    if (shape == eHexahedron) {
      nprint("Hex");
      printo = false;

      int mode = 0;
      for (int mz = 0; mz < num_modes[2]; mz++) {
        for (int my = 0; my < num_modes[1]; my++) {
          for (int mx = 0; mx < num_modes[0]; mx++) {
            mode_evals[mode] = evals[0][mx] * evals[1][my] * evals[2][mz];
            mode++;
          }
        }
      }

    } else if (shape == ePyramid) {
      nprint("Pyramid");
      printo = false;

      int mode = 0;
      for (int p = 0; p < P; ++p) {
        for (int q = 0; q < Q; ++q) {
          int maxpq = max(p, q);
          for (int r = 0; r < R - maxpq; ++r, ++mode) {
            const double contrib_0 = evals[0][p];
            const double contrib_1 = evals[1][q];
            const double contrib_2 = evals[2][r];

            if (mode == 1) {
              mode_evals[mode] = contrib_2;
            } else {
              mode_evals[mode] = contrib_0 * contrib_1 * contrib_2;
            }
          }
        }
      }

    } else if (shape == ePrism) {
      nprint("Prism");
      printo = false;

      int mode = 0;
      int mode_pr = 0;
      for (int p = 0; p < P; p++) {
        for (int q = 0; q < P; q++) {
          for (int r = 0; r < (P - p); r++) {
            const double contrib_0 = evals[0][p];
            const double contrib_1 = evals[1][q];
            const double contrib_2 = evals[2][r];
            mode_evals[mode] = contrib_0 * contrib_1 * contrib_2;

            if ((p == 0) && (r == 1)) {
              mode_evals[mode] /= eval_modA_i(p, local_collapsed[0]);
            }
            mode++;
          }
        }
        mode_pr += P - p;
      }

    } else if (shape == eTetrahedron) {
      nprint("Tetrahedron", num_modes[0], num_modes[1], num_modes[2]);
      printo = false;

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

      nprint(0, 0, 0, lambda_get_mode(0, 0, 0));

      int modex = lambda_get_mode(2, 2, 2);
      nprint(2, 2, 2, modex);

      nprint(ex->PhysEvaluateBasis(local_coord, modex));
      nprint(evals[0][2] * evals[1][2] * evals[2][2]);

      // mode_pqr = 0
      // mode_pq = 0
      // for p in P:
      //   for q in P - p:
      //     for r in P - p - r:
      //       out[mode_pqr*nq + r] = basis0[p*nq]*basis1[mode_pq +
      //       q]*basis2[mode_pqr + r]
      //     mode_pqr += (P - p - r)
      //   mode_pq += (P - p)

      /*
            int mode_pqr = 0;
            int mode_pq = 0;
            int mode = 0;
            for (int p = 0; p < P; p++) {
              for (int q = 0; q < (P - p); q++) {
                for (int r = 0; r < (P - p - q); r++) {
                  const int nq = (P - p - q);
                  const double contrib_0 = evals[0][p];
                  const double contrib_1 = evals[1][mode_pq + q];
                  const double contrib_2 = evals[2][mode_pqr + r];
                  mode_evals[mode] = contrib_0 * contrib_1 * contrib_2;
                  mode++;
                }
                mode_pqr += (P - p - q);
              }
              mode_pq += (P - p);
            }
      */

      // TODO
      int mode = 0;
      int mode_q = 0;
      for (int p = 0; p < P; p++) {
        for (int q = 0; q < (P - p); q++) {
          for (int r = 0; r < (P - p - q); r++) {
            const double contrib_0 = evals[0][p];
            const double contrib_1 = evals[1][mode_q];
            const double contrib_2 = evals[2][mode];
            mode_evals[mode] = contrib_0 * contrib_1 * contrib_2;
            mode++;
          }
          mode_q++;
        }
      }

    } else {
      nprint("other");
    }
    for (int modex = 0; modex < num_coeffs; modex++) {
      mode_correct[modex] = ex->PhysEvaluateBasis(local_coord, modex);
    }

    if (printo) {
      for (int modex = 0; modex < num_coeffs; modex++) {
        const double err = abs(mode_correct[modex] - mode_evals[modex]);
        nprint(modex, err, mode_correct[modex], mode_evals[modex]);
        // ASSERT_TRUE(err < 1.0e-10);
      }
    }
  }

  mesh->free();
  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}
