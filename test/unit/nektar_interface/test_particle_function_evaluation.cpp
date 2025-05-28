#include "nektar_interface/function_evaluation.hpp"
#include "nektar_interface/utilities.hpp"
#include "test_helper_utilities.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/DisContField.h>

TEST(ParticleFunctionEvaluation, DisContFieldScalar) {

  const int N_total = 2000;
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
      dis_cont_field, A, cell_id_translation);

  // evaluate field at particle locations
  field_evaluate->evaluate(Sym<REAL>("FUNC_EVALS"));

  // H5Part h5part("exp.h5part", A, Sym<REAL>("P"), Sym<INT>("NESO_MPI_RANK"),
  //               Sym<REAL>("NESO_REFERENCE_POSITIONS"),
  //               Sym<REAL>("FUNC_EVALS"));
  // h5part.write();
  // h5part.close();

  // check evaluations
  for (int cellx = 0; cellx < cell_count; cellx++) {

    auto positions = A->position_dat->cell_dat.get_cell(cellx);
    auto func_evals = (*A)[Sym<REAL>("FUNC_EVALS")]->cell_dat.get_cell(cellx);

    for (int rowx = 0; rowx < positions->nrow; rowx++) {

      const double x = (*positions)[0][rowx];
      const double y = (*positions)[1][rowx];

      const double eval_dat = (*func_evals)[0][rowx];
      // not expected to match due to BCs
      const double eval_correct = evaluate_scalar_2d(dis_cont_field, x, y);
      const double err = ABS(eval_correct - eval_dat);

      EXPECT_NEAR(eval_correct, eval_dat, 1.0e-5);
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

TEST(ParticleFunctionEvaluation, DisContFieldDerivative) {

  const int N_total = 2000;
  int argc = 3;
  char *argv[3];

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";
  std::filesystem::path mesh_file =
      test_resources_dir / "square_triangles_quads.xml";
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
                             ParticleProp(Sym<REAL>("FUNC_EVALS_VECTOR"), 2)};

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

  // H5Part h5part(
  //     "exp_vector.h5part", A, Sym<REAL>("P"), Sym<INT>("NESO_MPI_RANK"),
  //     Sym<REAL>("NESO_REFERENCE_POSITIONS"), Sym<REAL>("FUNC_EVALS_VECTOR"));
  // h5part.write();
  // h5part.close();

  // check evaluations
  // double err = 0.0;
  for (int cellx = 0; cellx < cell_count; cellx++) {

    auto positions = A->position_dat->cell_dat.get_cell(cellx);
    auto func_evals =
        (*A)[Sym<REAL>("FUNC_EVALS_VECTOR")]->cell_dat.get_cell(cellx);

    for (int rowx = 0; rowx < positions->nrow; rowx++) {

      const double x = (*positions)[0][rowx];
      const double y = (*positions)[1][rowx];

      const double eval_dat0 = (*func_evals)[0][rowx];
      const double eval_correct0 =
          evaluate_scalar_derivative_2d(dis_cont_field, x, y, 0);
      const double err0 = ABS(eval_correct0 - eval_dat0);

      const double eval_dat1 = (*func_evals)[1][rowx];
      const double eval_correct1 =
          evaluate_scalar_derivative_2d(dis_cont_field, x, y, 1);
      const double err1 = ABS(eval_correct1 - eval_dat1);

      // nprint(err0, err1, eval_correct0, eval_correct1);
      ASSERT_TRUE(err0 <= 2.7e-8);
      ASSERT_TRUE(err1 <= 2.7e-8);
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

TEST(ParticleFunctionEvaluation, ContFieldScalar) {

  const int N_total = 2000;
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

  // write_vtu(cont_field, "func.vtu");

  // create evaluation object
  auto field_evaluate = std::make_shared<FieldEvaluate<ContField>>(
      cont_field, A, cell_id_translation);

  // evaluate field at particle locations
  field_evaluate->evaluate(Sym<REAL>("FUNC_EVALS"));

  // H5Part h5part("exp.h5part", A, Sym<REAL>("P"), Sym<INT>("NESO_MPI_RANK"),
  //               Sym<REAL>("NESO_REFERENCE_POSITIONS"),
  //               Sym<REAL>("FUNC_EVALS"));
  // h5part.write();
  // h5part.close();

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
      EXPECT_NEAR(eval_correct, eval_dat, 1.0e-5);
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

TEST(ParticleFunctionEvaluation, ContFieldDerivative) {

  const int N_total = 2000;
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
                             ParticleProp(Sym<REAL>("FUNC_EVALS_VECTOR"), 2)};

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

  // write_vtu(cont_field, "func.vtu");

  // create evaluation object
  auto field_evaluate = std::make_shared<FieldEvaluate<ContField>>(
      cont_field, A, cell_id_translation, true);

  // evaluate field at particle locations
  field_evaluate->evaluate(Sym<REAL>("FUNC_EVALS_VECTOR"));

  // H5Part h5part(
  //     "exp_vector.h5part", A, Sym<REAL>("P"), Sym<INT>("NESO_MPI_RANK"),
  //     Sym<REAL>("NESO_REFERENCE_POSITIONS"),
  //     Sym<REAL>("FUNC_EVALS_VECTOR"));
  // h5part.write();
  // h5part.close();

  // check evaluations
  // double err = 0.0;
  for (int cellx = 0; cellx < cell_count; cellx++) {

    auto positions = A->position_dat->cell_dat.get_cell(cellx);
    auto func_evals =
        (*A)[Sym<REAL>("FUNC_EVALS_VECTOR")]->cell_dat.get_cell(cellx);

    for (int rowx = 0; rowx < positions->nrow; rowx++) {

      const double x = (*positions)[0][rowx];
      const double y = (*positions)[1][rowx];

      const double eval_dat0 = (*func_evals)[0][rowx];
      const double eval_correct0 =
          evaluate_scalar_derivative_2d(cont_field, x, y, 0);
      const double err0 = ABS(eval_correct0 - eval_dat0);

      const double eval_dat1 = (*func_evals)[1][rowx];
      const double eval_correct1 =
          evaluate_scalar_derivative_2d(cont_field, x, y, 1);
      const double err1 = ABS(eval_correct1 - eval_dat1);

      // nprint(err0, err1, eval_correct0, eval_correct1);
      ASSERT_TRUE(err0 <= 2.5e-4);
      ASSERT_TRUE(err1 <= 2.5e-4);
      // err = MAX(err, err0);
      // err = MAX(err, err1);
    }
  }
  // nprint("err:", err);

  A->free();
  sycl_target->free();
  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

TEST(BaryInterpolation, Evaluation2D) {

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

  auto lambda_f = [&](const NekDouble x, const NekDouble y) {
    return 2.0 * (x + 0.5) * (x - 0.5) * (y + 0.8) * (y - 0.8);
  };
  interpolate_onto_nektar_field_2d(lambda_f, cont_field);

  const auto global_physvals = cont_field->GetPhys();

  int rank;
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  std::mt19937 rng(22123234 + rank);
  std::uniform_real_distribution<double> uniform_rng(-0.1, 0.1);

  const int num_elts = cont_field->GetNumElmts();
  Array<OneD, NekDouble> Lcoord(2);
  Array<OneD, NekDouble> coord(2);
  for (int ex = 0; ex < num_elts; ex++) {
    auto exp = cont_field->GetExp(ex);
    auto geom = exp->GetGeom();
    auto base = exp->GetBase();
    const auto &z0 = base[0]->GetZ();
    const auto &bw0 = base[0]->GetBaryWeights();
    const auto &z1 = base[1]->GetZ();
    const auto &bw1 = base[1]->GetBaryWeights();
    const int num_phys0 = z0.size();
    const int num_phys1 = z1.size();
    const int num_phys = std::max(num_phys0, num_phys1);
    std::vector<REAL> div_space(2 * num_phys);
    std::vector<REAL> z0v(num_phys);
    std::vector<REAL> z1v(num_phys);
    std::vector<REAL> bw0v(num_phys);
    std::vector<REAL> bw1v(num_phys);
    for (int ix = 0; ix < num_phys0; ix++) {
      z0v[ix] = z0[ix];
      bw0v[ix] = bw0[ix];
    }
    for (int ix = 0; ix < num_phys1; ix++) {
      z1v[ix] = z1[ix];
      bw1v[ix] = bw1[ix];
    }
    const auto physvals = global_physvals + cont_field->GetPhys_Offset(ex);
    std::vector<REAL> physvalsv(num_phys0 * num_phys1);
    for (int ix = 0; ix < (num_phys0 * num_phys1); ix++) {
      physvalsv[ix] = physvals[ix];
    }

    // check bary eval at all the quad points
    for (int p0 = 0; p0 < num_phys0; p0++) {
      for (int p1 = 0; p1 < num_phys1; p1++) {
        const REAL x0 = z0[p0];
        const REAL x1 = z1[p1];
        coord[0] = x0;
        coord[1] = x1;
        exp->LocCollapsedToLocCoord(coord, Lcoord);
        const REAL correct = exp->StdPhysEvaluate(Lcoord, physvals);
        const REAL to_test = Bary::evaluate_2d(
            x0, x1, num_phys0, num_phys1, physvalsv.data(), div_space.data(),
            z0v.data(), z1v.data(), bw0v.data(), bw1v.data());

        const REAL err_abs = std::abs(correct - to_test);
        const REAL abs_correct = std::abs(correct);
        const REAL err_rel =
            abs_correct > 0 ? err_abs / abs_correct : abs_correct;
        EXPECT_TRUE(err_rel < 1.0e-12 || err_abs < 1.0e-12);
      }
    }
    // check bary eval at away from the quad points
    for (int p0 = 0; p0 < num_phys0; p0++) {
      for (int p1 = 0; p1 < num_phys1; p1++) {
        const REAL x0 = z0[p0] + uniform_rng(rng);
        const REAL x1 = z1[p1] + uniform_rng(rng);
        coord[0] = x0;
        coord[1] = x1;
        exp->LocCollapsedToLocCoord(coord, Lcoord);
        const REAL correct = exp->StdPhysEvaluate(Lcoord, physvals);
        const REAL to_test = Bary::evaluate_2d(
            x0, x1, num_phys0, num_phys1, physvalsv.data(), div_space.data(),
            z0v.data(), z1v.data(), bw0v.data(), bw1v.data());

        const REAL err_abs = std::abs(correct - to_test);
        const REAL abs_correct = std::abs(correct);
        const REAL err_rel =
            abs_correct > 0 ? err_abs / abs_correct : abs_correct;
        EXPECT_TRUE(err_rel < 1.0e-12 || err_abs < 1.0e-12);
      }
    }
  }

  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

TEST(BaryInterpolation, Generic) {

  const int stride = 3;
  const int num_phys0 = 7;
  const int num_phys1 = 5;
  const int num_phys2 = 9;
  const int num_phys = num_phys0 * num_phys1 * num_phys2;
  std::mt19937 rng(22123257);
  std::uniform_real_distribution<double> uniform_rng(-2.0, 2.0);

  auto lambda_rng = [&]() -> REAL { return uniform_rng(rng); };

  auto lambda_make_phys_vals = [&]() -> std::vector<REAL> {
    std::vector<REAL> data(num_phys);
    std::generate(data.begin(), data.end(), lambda_rng);
    return data;
  };

  auto lambda_interlace_2 = [](auto p0, auto p1) -> std::vector<REAL> {
    std::vector<REAL> data(p0.size() + p1.size());
    EXPECT_EQ(p0.size(), p1.size());
    const auto N = p0.size();
    std::size_t index = 0;
    for (std::size_t ix = 0; ix < N; ix++) {
      data.at(index++) = p0.at(ix);
      data.at(index++) = p1.at(ix);
    }
    return data;
  };

  auto lambda_interlace_3 = [](auto p0, auto p1, auto p2) -> std::vector<REAL> {
    std::vector<REAL> data(p0.size() + p1.size() + p2.size());
    EXPECT_EQ(p0.size(), p1.size());
    EXPECT_EQ(p0.size(), p2.size());
    const auto N = p0.size();
    std::size_t index = 0;
    for (std::size_t ix = 0; ix < N; ix++) {
      data.at(index++) = p0.at(ix);
      data.at(index++) = p1.at(ix);
      data.at(index++) = p2.at(ix);
    }
    return data;
  };

  auto lambda_rel_error = [](auto a, auto b) {
    const auto err_abs = std::abs(a - b);
    const auto mag = std::abs(a);
    const auto err_rel = mag > 0.0 ? err_abs / mag : err_abs;
    const REAL tol = 1.0e-14;
    if (err_rel > tol) {
      nprint("Error:", a, b);
    }
    EXPECT_TRUE(err_rel <= tol);
  };

  std::vector<REAL> div_space0(num_phys0 * stride);
  std::vector<REAL> div_space1(num_phys1 * stride);
  std::vector<REAL> div_space2(num_phys2 * stride);
  std::generate(div_space0.begin(), div_space0.end(), lambda_rng);
  std::generate(div_space1.begin(), div_space1.end(), lambda_rng);
  std::generate(div_space2.begin(), div_space2.end(), lambda_rng);

  auto func0 = lambda_make_phys_vals();
  auto func1 = lambda_make_phys_vals();
  auto func2 = lambda_make_phys_vals();

  std::vector<std::size_t> strides = {1, stride};
  for (auto test_stride : strides) {

    // 2D, 1 function
    {
      const REAL correct = Bary::compute_dir_10(num_phys0, num_phys1,
                                                func0.data(), div_space0.data(),
                                                div_space1.data(), test_stride);

      REAL to_test[1];
      Bary::compute_dir_10_interlaced<1>(num_phys0, num_phys1, func0.data(),
                                         div_space0.data(), div_space1.data(),
                                         to_test, test_stride);
      lambda_rel_error(correct, to_test[0]);

      Bary::compute_dir_10_interlaced(1, num_phys0, num_phys1, func0.data(),
                                      div_space0.data(), div_space1.data(),
                                      to_test, test_stride);
      lambda_rel_error(correct, to_test[0]);
    }

    // 2D, 2 functions
    {
      auto func01 = lambda_interlace_2(func0, func1);
      REAL to_test0[2];
      Bary::compute_dir_10_interlaced<2>(num_phys0, num_phys1, func01.data(),
                                         div_space0.data(), div_space1.data(),
                                         to_test0, test_stride);

      REAL *tmp_data[2] = {func0.data(), func1.data()};
      for (int dx = 0; dx < 2; dx++) {
        const REAL correct = Bary::compute_dir_10(
            num_phys0, num_phys1, tmp_data[dx], div_space0.data(),
            div_space1.data(), test_stride);
        lambda_rel_error(correct, to_test0[dx]);
      }

      REAL to_test1[2];
      Bary::compute_dir_10_interlaced(2, num_phys0, num_phys1, func01.data(),
                                      div_space0.data(), div_space1.data(),
                                      to_test1, test_stride);

      lambda_rel_error(to_test0[0], to_test1[0]);
      lambda_rel_error(to_test0[1], to_test1[1]);
    }

    // 2D, 3 functions
    {
      auto func012 = lambda_interlace_3(func0, func1, func2);
      REAL to_test0[3];
      Bary::compute_dir_10_interlaced<3>(num_phys0, num_phys1, func012.data(),
                                         div_space0.data(), div_space1.data(),
                                         to_test0, test_stride);

      REAL *tmp_data[3] = {func0.data(), func1.data(), func2.data()};
      for (int dx = 0; dx < 3; dx++) {
        const REAL correct = Bary::compute_dir_10(
            num_phys0, num_phys1, tmp_data[dx], div_space0.data(),
            div_space1.data(), test_stride);
        lambda_rel_error(correct, to_test0[dx]);
      }

      REAL to_test1[3];
      Bary::compute_dir_10_interlaced(3, num_phys0, num_phys1, func012.data(),
                                      div_space0.data(), div_space1.data(),
                                      to_test1, test_stride);

      lambda_rel_error(to_test0[0], to_test1[0]);
      lambda_rel_error(to_test0[1], to_test1[1]);
      lambda_rel_error(to_test0[2], to_test1[2]);
    }

    // 3D, 1 function
    {
      const REAL correct = Bary::compute_dir_210(
          num_phys0, num_phys1, num_phys2, func0.data(), div_space0.data(),
          div_space1.data(), div_space2.data(), test_stride);

      REAL to_test[1];
      Bary::compute_dir_210_interlaced<1>(
          num_phys0, num_phys1, num_phys2, func0.data(), div_space0.data(),
          div_space1.data(), div_space2.data(), to_test, test_stride);
      lambda_rel_error(correct, to_test[0]);

      Bary::compute_dir_210_interlaced(
          1, num_phys0, num_phys1, num_phys2, func0.data(), div_space0.data(),
          div_space1.data(), div_space2.data(), to_test, test_stride);
      lambda_rel_error(correct, to_test[0]);
    }

    // 3D, 2 functions
    {
      auto func01 = lambda_interlace_2(func0, func1);
      REAL to_test0[2];
      Bary::compute_dir_210_interlaced<2>(
          num_phys0, num_phys1, num_phys2, func01.data(), div_space0.data(),
          div_space1.data(), div_space2.data(), to_test0, test_stride);

      REAL *tmp_data[2] = {func0.data(), func1.data()};
      for (int dx = 0; dx < 2; dx++) {
        const REAL correct = Bary::compute_dir_210(
            num_phys0, num_phys1, num_phys2, tmp_data[dx], div_space0.data(),
            div_space1.data(), div_space2.data(), test_stride);
        lambda_rel_error(correct, to_test0[dx]);
      }

      REAL to_test1[2];
      Bary::compute_dir_210_interlaced(
          2, num_phys0, num_phys1, num_phys2, func01.data(), div_space0.data(),
          div_space1.data(), div_space2.data(), to_test1, test_stride);

      lambda_rel_error(to_test0[0], to_test1[0]);
      lambda_rel_error(to_test0[1], to_test1[1]);
    }

    // 3D, 3 functions
    {
      auto func012 = lambda_interlace_3(func0, func1, func2);
      REAL to_test0[3];
      Bary::compute_dir_210_interlaced<3>(
          num_phys0, num_phys1, num_phys2, func012.data(), div_space0.data(),
          div_space1.data(), div_space2.data(), to_test0, test_stride);

      REAL *tmp_data[3] = {func0.data(), func1.data(), func2.data()};
      for (int dx = 0; dx < 3; dx++) {
        const REAL correct = Bary::compute_dir_210(
            num_phys0, num_phys1, num_phys2, tmp_data[dx], div_space0.data(),
            div_space1.data(), div_space2.data(), test_stride);
        lambda_rel_error(correct, to_test0[dx]);
      }

      REAL to_test1[3];
      Bary::compute_dir_210_interlaced(
          3, num_phys0, num_phys1, num_phys2, func012.data(), div_space0.data(),
          div_space1.data(), div_space2.data(), to_test1, test_stride);

      lambda_rel_error(to_test0[0], to_test1[0]);
      lambda_rel_error(to_test0[1], to_test1[1]);
      lambda_rel_error(to_test0[2], to_test1[2]);
    }
  }
}
