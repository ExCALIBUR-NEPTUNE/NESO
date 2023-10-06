#include "main_evaluation.hpp"

int main_evaluation(int argc, char *argv[],
                    LibUtilities::SessionReaderSharedPtr session) {
  int err = 0;

  auto graph = SpatialDomains::MeshGraph::Read(session);
  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  const int ndim = mesh->get_ndim();

  MPI_Comm comm;
  comm = mesh->get_comm();
  int rank, size;
  MPICHK(MPI_Comm_rank(comm, &rank));
  MPICHK(MPI_Comm_size(comm, &size));

  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());
  if (rank == 0) {
    sycl_target->print_device_info();
  }
  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);
  auto esw = std::make_shared<EquationSystemWrapper>(session, graph);

  auto fields = esw->UpdateFields();
  NESOASSERT(fields.size(), "No field found to evaluate.");
  auto field = std::dynamic_pointer_cast<DisContField>(fields[0]);

  // Create a function to evaluate
  NESOASSERT(ndim == 2 || ndim == 3, "Unexpected number of dimensions");
  if (ndim == 2) {
    auto lambda_f = [&](const NekDouble x, const NekDouble y) {
      return std::pow((x + 1.0) * (x - 1.0) * (y + 1.0) * (y - 1.0), 4);
    };
    interpolate_onto_nektar_field_2d(lambda_f, field);
  } else {
    auto lambda_f = [&](const NekDouble x, const NekDouble y,
                        const NekDouble z) {
      return std::pow((x + 1.0) * (x - 1.0) * (y + 1.0) * (y - 1.0) *
                          (z + 1.0) * (z - 1.0),
                      4);
    };
    interpolate_onto_nektar_field_3d(lambda_f, field);
  }

  // create particle system
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<REAL>("E"), 1)};
  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  // method to translate from Nektar cells to NESO cells
  auto cell_id_translation =
      std::make_shared<CellIDTranslation>(sycl_target, A->cell_id_dat, mesh);

  // Determine total number of particles
  int num_particles_total;
  const int num_particles_total_default = 16000;
  session->LoadParameter("num_particles_total", num_particles_total,
                         num_particles_total_default);

  // Determine how many particles this rank initialises
  int rstart, rend;
  get_decomp_1d(size, num_particles_total, rank, &rstart, &rend);
  const int N = rend - rstart;
  int num_particles_check = -1;
  MPICHK(MPI_Allreduce(&N, &num_particles_check, 1, MPI_INT, MPI_SUM,
                       MPI_COMM_WORLD));
  NESOASSERT(num_particles_check == num_particles_total,
             "Error creating particles");

  // Initialise particles on this rank
  int seed;
  const int seed_default = 34672835;
  session->LoadParameter("seed", seed, seed_default);
  std::mt19937 rng_pos(seed + rank);

  // uniformly distributed positions
  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
  if (N > 0) {
    auto positions =
        uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);

    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
      }
    }
    A->add_particles_local(initial_distribution);
  }

  // send the particles to the correct rank
  // parallel_advection_initialisation(A, 32);
  pbc.execute();
  A->hybrid_move();
  cell_id_translation->execute();
  A->cell_move();

  // create the evaluation instance
  auto field_evaluate = std::make_shared<BenchmarkEvaluate<DisContField>>(
      field, mesh, cell_id_translation);
  const auto global_coeffs = field->GetCoeffs();
  field_evaluate->evaluate_init(global_coeffs);

  // Determine total number of warmup steps
  int num_steps_warmup;
  const int num_steps_warmup_default = 2;
  session->LoadParameter("num_steps_warmup", num_steps_warmup,
                         num_steps_warmup_default);
  // Determine total number of steps
  int num_steps;
  const int num_steps_default = 10;
  session->LoadParameter("num_steps", num_steps, num_steps_default);

  for (int stepx = 0; stepx < num_steps_warmup; stepx++) {
    field_evaluate->evaluate_quadrilaterals(A, Sym<REAL>("E"), 0);
    field_evaluate->evaluate_triangles(A, Sym<REAL>("E"), 0);
    field_evaluate->evaluate_hexahedrons(A, Sym<REAL>("E"), 0);
    field_evaluate->evaluate_prisms(A, Sym<REAL>("E"), 0);
    field_evaluate->evaluate_tetrahedrons(A, Sym<REAL>("E"), 0);
    field_evaluate->evaluate_pyramids(A, Sym<REAL>("E"), 0);
  }

  std::vector<double> times_local = {0, 0, 0, 0, 0, 0};
  for (int stepx = 0; stepx < num_steps; stepx++) {
    times_local.at(0) +=
        field_evaluate->evaluate_quadrilaterals(A, Sym<REAL>("E"), 0);
    times_local.at(1) +=
        field_evaluate->evaluate_triangles(A, Sym<REAL>("E"), 0);
    times_local.at(2) +=
        field_evaluate->evaluate_hexahedrons(A, Sym<REAL>("E"), 0);
    times_local.at(3) += field_evaluate->evaluate_prisms(A, Sym<REAL>("E"), 0);
    times_local.at(4) +=
        field_evaluate->evaluate_tetrahedrons(A, Sym<REAL>("E"), 0);
    times_local.at(5) +=
        field_evaluate->evaluate_pyramids(A, Sym<REAL>("E"), 0);
  }

  std::vector<int> flops_count(6);
  std::vector<int> particle_count(6);
  field_evaluate->get_flop_counts(flops_count);
  field_evaluate->get_particle_counts(A, particle_count);
  std::vector<double> flops_local(6);

  for (int cx = 0; cx < 6; cx++) {
    if (particle_count[cx] > 0) {
      flops_local[cx] = ((double)(particle_count[cx]) *
                         ((double)flops_count[cx]) * ((double)num_steps)) /
                        times_local[cx];
    } else {
      flops_local[cx] = 0;
    }
  }

  std::vector<double> flops_global(6 * size);
  MPICHK(MPI_Gather(flops_local.data(), 6, MPI_DOUBLE, flops_global.data(), 6,
                    MPI_DOUBLE, 0, comm));

  if (rank == 0) {
    for (int rx = 0; rx < size; rx++) {
      printf(
          "%6.4e, %6.4e, %6.4e, %6.4e, %6.4e, %6.4e\n",
          flops_global[rx * 6 + 0] * 1.0e-9, flops_global[rx * 6 + 1] * 1.0e-9,
          flops_global[rx * 6 + 2] * 1.0e-9, flops_global[rx * 6 + 3] * 1.0e-9,
          flops_global[rx * 6 + 4] * 1.0e-9, flops_global[rx * 6 + 5] * 1.0e-9);
    }

    std::string filename = "bench_flops_evaluation.h5";
    hid_t file =
        H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dims[2] = {static_cast<hsize_t>(size), 6};
    hid_t dataspace = H5Screate_simple(2, dims, nullptr);
    hid_t dataset =
        H5Dcreate2(file, "flops_evaluation", H5T_NATIVE_DOUBLE, dataspace,
                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    H5CHK(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                   flops_global.data()));

    H5CHK(H5Dclose(dataset));
    H5CHK(H5Sclose(dataspace));

    auto lambda_write_int = [&](const char *key, const int value) {
      hid_t dataspace = H5Screate(H5S_SCALAR);
      hid_t attribute = H5Acreate(file, key, H5T_NATIVE_INT, dataspace,
                                  H5P_DEFAULT, H5P_DEFAULT);
      herr_t status = H5Awrite(attribute, H5T_NATIVE_INT, &value);
      H5Aclose(attribute);
      H5Sclose(dataspace);
    };

    lambda_write_int("MPI_SIZE", size);
    lambda_write_int("num_particles_total", num_particles_total);
    lambda_write_int("num_modes", field_evaluate->get_num_modes());

    H5CHK(H5Fclose(file));
  }

  A->free();
  sycl_target->free();
  mesh->free();
  return err;
}
