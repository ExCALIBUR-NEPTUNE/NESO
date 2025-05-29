#include "nektar_interface/function_evaluation.hpp"
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
using namespace NESO::BasisReference;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

TEST(JacobiCoeffModBasis, Coeff) {

  ASSERT_NEAR(jacobi(5, 0.3, 3, 4), 1.5229637499999997, 1.0e-10);
  ASSERT_NEAR(jacobi(13, 0.3, 7, 11), -9.5066868221006, 1.0e-10);
  ASSERT_NEAR(jacobi(13, -0.5, 7, 11), -47.09590101242081, 1.0e-10);

  const int max_n = 13;
  const int max_alpha = 7;
  JacobiCoeffModBasis jacobi_coeff(max_n, max_alpha);

  ASSERT_NEAR(jacobi(13, -0.5, 7, 1), jacobi_coeff.host_evaluate(13, 7, -0.5),
              1.0e-10);
  ASSERT_NEAR(jacobi(0, -0.5, 7, 1), 1.0, 1.0e-10);
  ASSERT_NEAR(jacobi(7, 0.5, 3, 1), jacobi_coeff.host_evaluate(7, 3, 0.5),
              1.0e-10);

  const double z = -0.4234;
  for (int n = 0; n <= max_n; n++) {
    for (int alpha = 1; alpha <= max_alpha; alpha++) {
      const double correct = jacobi(n, z, alpha, 1);
      const double to_test = jacobi_coeff.host_evaluate(n, alpha, z);
      const double err_rel = relative_error(correct, to_test);
      const double err_abs = std::abs(correct - to_test);
      ASSERT_TRUE(err_rel < 1.0e-14 || err_abs < 1.0e-14);
    }
  }
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

TEST(ParticleFunctionBasisEvaluation, Basis2D) {

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
  graph = SpatialDomains::MeshGraphIO::Read(session);
  auto dis_cont_field = std::make_shared<DisContField>(session, graph, "u");

  std::mt19937 rng{182348};

  int64_t errs_count = 0;
  double errs_total = 0.0;

  for (int ei = 0; ei < dis_cont_field->GetNumElmts(); ei++) {
    auto ex = dis_cont_field->GetExp(ei);
    auto shape = ex->DetShapeType();
    auto geom = ex->GetGeom();
    auto bounding_box = geom->GetBoundingBox();

    std::uniform_real_distribution<double> uniform_dist0(bounding_box[0],
                                                         bounding_box[0] + 3);
    std::uniform_real_distribution<double> uniform_dist1(bounding_box[1],
                                                         bounding_box[1] + 3);

    Array<OneD, NekDouble> local_collapsed(3);
    Array<OneD, NekDouble> local_coord(3);
    Array<OneD, NekDouble> global_coord(3);

    global_coord[2] = 0.0;

    for (int testx = 0; testx < 2; testx++) {
      bool is_contained = false;
      while (!is_contained) {
        global_coord[0] = uniform_dist0(rng);
        global_coord[1] = uniform_dist1(rng);
        is_contained =
            ex->GetGeom()->ContainsPoint(global_coord, local_coord, 1.0e-8);
      }
      ex->LocCoordToLocCollapsed(local_coord, local_collapsed);

      const int P = ex->GetBasis(0)->GetNumModes();
      const int Q = ex->GetBasis(1)->GetNumModes();
      ASSERT_EQ(P, Q);

      const int num_coeffs = ex->GetNcoeffs();
      const int to_test_num_coeffs = get_total_num_modes(shape, P);
      EXPECT_EQ(num_coeffs, to_test_num_coeffs);

      std::vector<double> mode_evals(num_coeffs);
      std::vector<double> mode_evals_basis(num_coeffs);
      std::vector<double> mode_correct(num_coeffs);

      eval_modes(shape, P, local_collapsed[0], local_collapsed[1],
                 local_collapsed[2], mode_evals_basis);

      auto lambda_err = [](const double correct, const double to_test) {
        const double abs_err = std::abs(correct - to_test);
        const double scaling = std::abs(correct);
        const double rel_err = scaling > 0 ? abs_err / scaling : abs_err;
        return std::min(abs_err, rel_err);
      };

      for (int modex = 0; modex < num_coeffs; modex++) {
        const double correct = ex->PhysEvaluateBasis(local_coord, modex);
        const double to_test = mode_evals_basis[modex];
        const double err = minimum_absrel_error(correct, to_test);
        errs_total += err;
        errs_count++;
        // near the collapsed singularity?
        if (std::abs(local_collapsed[1]) > 0.85) {
          EXPECT_TRUE((err < 1.0e-8) || std::abs(correct) < 1.0e-7);
        } else {
          EXPECT_TRUE((err < 1.0e-10) || std::abs(correct) < 1.0e-7);
        }
      }
    }
  }

  int64_t errs_count_global;
  double errs_total_global;

  MPICHK(MPI_Allreduce(&errs_count, &errs_count_global, 1, MPI_INT64_T, MPI_SUM,
                       MPI_COMM_WORLD));
  MPICHK(MPI_Allreduce(&errs_total, &errs_total_global, 1, MPI_DOUBLE, MPI_SUM,
                       MPI_COMM_WORLD));

  REAL errs_avg = errs_total_global / errs_count_global;
  ASSERT_TRUE(errs_avg < 1.0e-9);

  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

TEST(ParticleFunctionBasisEvaluation, Basis3D) {

  std::tuple<std::string, std::string, double> param = {
      "reference_all_types_cube/conditions.xml",
      "reference_all_types_cube/linear_non_regular_0.5.xml", 2.0e-4};

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
  graph = SpatialDomains::MeshGraphIO::Read(session);

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  std::mt19937 rng{182348};

  auto cont_field = std::make_shared<ContField>(session, graph, "u");

  auto lambda_func = [](const double x, const double y, const double z) {
    // return sin(x) * cos(y) + exp(z);
    return (x + 1) * (x - 1) * (y + 1) * (y - 1) * (z + 1) * (z - 1);
  };

  interpolate_onto_nektar_field_3d(lambda_func, cont_field);

  auto lambda_err = [](const double correct, const double to_test) {
    const double abs_err = std::abs(correct - to_test);
    const double scaling = std::abs(correct);
    const double rel_err = scaling > 0 ? abs_err / scaling : abs_err;
    return std::min(abs_err, rel_err);
  };

  double avg_err_dof = 0.0;
  double avg_err_eval = 0.0;
  INT avg_count_dof = 0;
  INT avg_count_eval = 0;

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

      // test that the NESO implementation of the basis matches the Nektar++
      // implementation of the basis at the quadrature points.
      for (int m = 0; m < mb0; m++) {
        const double zz = zb0[m];
        eval_basis(b0->GetBasisType(), nb0, zz, to_test);
        for (int mx = 0; mx < total_num_modes; mx++) {
          ASSERT_TRUE((mx * mb0 + m) < bb0.size());
          const double zc = bb0[mx * mb0 + m];
          const double tt = to_test.at(mx);
          ASSERT_NEAR(zc, tt, 1.0e-10);
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

    for (int testx = 0; testx < 2; testx++) {

      bool is_contained = false;
      while (!is_contained) {

        global_coord[0] = uniform_dist0(rng);
        global_coord[1] = uniform_dist1(rng);
        global_coord[2] = uniform_dist2(rng);
        is_contained =
            ex->GetGeom()->ContainsPoint(global_coord, local_coord, 1.0e-8);
      }
      ex->LocCoordToLocCollapsed(local_coord, local_collapsed);

      const int num_coeffs = ex->GetNcoeffs();
      std::vector<double> mode_evals(num_coeffs);
      std::vector<double> mode_evals_basis(num_coeffs);
      std::vector<double> mode_correct(num_coeffs);

      const int P = ex->GetBasis(0)->GetNumModes();
      const int Q = ex->GetBasis(1)->GetNumModes();
      const int R = ex->GetBasis(2)->GetNumModes();
      ASSERT_EQ(P, Q);
      ASSERT_EQ(P, R);

      const int to_test_num_coeffs = get_total_num_modes(shape, P);
      EXPECT_EQ(to_test_num_coeffs, num_coeffs);

      eval_modes(shape, P, local_collapsed[0], local_collapsed[1],
                 local_collapsed[2], mode_evals_basis);

      auto coeffs_global = cont_field->GetCoeffs();
      auto phys_global = cont_field->GetPhys();

      auto coeffs = coeffs_global + cont_field->GetCoeff_Offset(ei);
      auto phys = phys_global + cont_field->GetPhys_Offset(ei);

      // get each basis function evaluated at the point
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

      // get field evaluation at point
      const double eval_bary =
          ex->StdPhysEvaluate(local_coord, phys_global + offset_phys);

      // test the basis functions computed through StdPhysEvaluate vs the modes
      // we computed
      for (int modex = 0; modex < num_coeffs; modex++) {
        const double dof_err =
            lambda_err(mode_via_coeffs[modex], mode_evals_basis[modex]);
        const double dof_abs_err =
            std::abs(mode_via_coeffs[modex] - mode_evals_basis[modex]);
        avg_count_dof++;
        avg_err_dof += dof_err;
        EXPECT_TRUE(dof_err < 2.0e-5 || dof_abs_err < 1.0e-4);
      }
      // evaluation using modes we computed
      double eval_modes = 0.0;
      double eval_modes_nektar = 0.0;
      for (int modex = 0; modex < num_coeffs; modex++) {
        const double mode_tmp = mode_evals_basis[modex];
        const double mode_tmp_nektar = mode_via_coeffs[modex];
        const double dof_tmp = coeffs[modex];
        eval_modes += dof_tmp * mode_tmp;
        eval_modes_nektar += dof_tmp * mode_tmp_nektar;
      }

      const double eval_err = lambda_err(eval_bary, eval_modes);
      const double eval_err_nektar = lambda_err(eval_bary, eval_modes_nektar);
      const double eval_abs_err = std::abs(eval_bary - eval_modes);

      avg_count_eval++;
      avg_err_eval += eval_err;
      EXPECT_TRUE(eval_err < 2.0e-5 || eval_abs_err < 1.0e-4);
    }
  }

  double avg_err_tmp[2] = {avg_err_eval, avg_err_dof};
  double avg_err[2] = {0, 0};

  int64_t avg_count_tmp[2] = {avg_count_eval, avg_count_dof};
  int64_t avg_count[2] = {0, 0};

  MPICHK(MPI_Allreduce(avg_err_tmp, avg_err, 2, MPI_DOUBLE, MPI_SUM,
                       sycl_target->comm_pair.comm_parent));
  MPICHK(MPI_Allreduce(avg_count_tmp, avg_count, 2, MPI_INT64_T, MPI_SUM,
                       sycl_target->comm_pair.comm_parent));

  avg_err[0] /= ((double)avg_count[0]);
  avg_err[1] /= ((double)avg_count[1]);

  EXPECT_TRUE(avg_err[0] < 1.0e-8);
  EXPECT_TRUE(avg_err[1] < 1.0e-8);

  mesh->free();
  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

TEST(KernelBasis, EvalA) {

  const int P = 9;
  int max_alpha, max_n, total_num_modes;

  total_num_modes = P;
  BasisReference::get_total_num_modes(eQuadrilateral, P, &max_n, &max_alpha);
  JacobiCoeffModBasis jacobi_coeff(max_n, max_alpha);

  std::vector<double> to_test(total_num_modes);
  std::vector<double> correct(total_num_modes);

  const double z = -0.124124;

  eval_modA(P, z, correct);

  BasisJacobi::mod_A(P, z, jacobi_coeff.stride_n,
                     jacobi_coeff.coeffs_pnm10.data(),
                     jacobi_coeff.coeffs_pnm11.data(),
                     jacobi_coeff.coeffs_pnm2.data(), to_test.data());

  for (int p = 0; p < P; p++) {
    const double err = std::abs(correct[p] - to_test[p]);
    EXPECT_TRUE(err < 1.0e-14);
  }
}

TEST(KernelBasis, EvalB) {

  const int P = 9;
  int max_alpha, max_n, total_num_modes;

  total_num_modes =
      BasisReference::get_total_num_modes(eTriangle, P, &max_n, &max_alpha);
  JacobiCoeffModBasis jacobi_coeff(max_n, max_alpha);

  std::vector<double> to_test(total_num_modes);
  std::vector<double> correct(total_num_modes);

  const double z = -0.124124;

  eval_modB(P, z, correct);

  BasisJacobi::mod_B(P, z, jacobi_coeff.stride_n,
                     jacobi_coeff.coeffs_pnm10.data(),
                     jacobi_coeff.coeffs_pnm11.data(),
                     jacobi_coeff.coeffs_pnm2.data(), to_test.data());

  int mode = 0;
  for (int p = 0; p < P; p++) {
    for (int q = 0; q < (P - p); q++) {
      const double err = std::abs(correct[mode] - to_test[mode]);
      EXPECT_TRUE(err < 1.0e-14);
      mode++;
    }
  }
}

TEST(KernelBasis, EvalC) {

  const int P = 9;
  int max_alpha, max_n, total_num_modes;

  total_num_modes =
      BasisReference::get_total_num_modes(eTetrahedron, P, &max_n, &max_alpha);
  JacobiCoeffModBasis jacobi_coeff(max_n, max_alpha);

  std::vector<double> to_test(total_num_modes);
  std::vector<double> correct(total_num_modes);

  const double z = -0.124124;

  eval_modC(P, z, correct);

  BasisJacobi::mod_C(P, z, jacobi_coeff.stride_n,
                     jacobi_coeff.coeffs_pnm10.data(),
                     jacobi_coeff.coeffs_pnm11.data(),
                     jacobi_coeff.coeffs_pnm2.data(), to_test.data());

  int mode = 0;
  for (int p = 0; p < P; p++) {
    for (int q = 0; q < (P - p); q++) {
      for (int r = 0; r < (P - p - q); r++) {
        const double err = std::abs(correct[mode] - to_test[mode]);
        EXPECT_TRUE(err < 1.0e-14);
        mode++;
      }
    }
  }
}

TEST(KernelBasis, EvalPyrC) {

  const int P = 9;
  int max_alpha, max_n, total_num_modes;

  total_num_modes = get_total_num_modes(eModifiedPyr_C, P);
  BasisReference::get_total_num_modes(ePyramid, P, &max_n, &max_alpha);
  JacobiCoeffModBasis jacobi_coeff(max_n, max_alpha);

  std::vector<double> to_test(total_num_modes);
  std::vector<double> correct(total_num_modes);

  const double z = -0.124124;

  eval_modPyrC(P, z, correct);

  BasisJacobi::mod_PyrC(P, z, jacobi_coeff.stride_n,
                        jacobi_coeff.coeffs_pnm10.data(),
                        jacobi_coeff.coeffs_pnm11.data(),
                        jacobi_coeff.coeffs_pnm2.data(), to_test.data());

  int mode = 0;
  for (int p = 0; p < P; p++) {
    for (int q = 0; q < P; q++) {
      for (int r = 0; r < P - std::max(p, q); r++) {
        const double err = std::abs(correct[mode] - to_test[mode]);
        EXPECT_TRUE(err < 1.0e-14);
        mode++;
      }
    }
  }
}
