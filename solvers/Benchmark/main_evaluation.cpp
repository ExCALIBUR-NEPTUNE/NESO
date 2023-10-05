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
  parallel_advection_initialisation(A, 32);
  cell_id_translation->execute();
  A->cell_move();

  // create the evaluation instance
  auto field_evaluate = std::make_shared<BenchmarkEvaluate<DisContField>>(
      field, mesh, cell_id_translation);
  const auto global_coeffs = field->GetCoeffs();
  field_evaluate->evaluate_init(global_coeffs);

  auto time_quads =
      field_evaluate->evaluate_quadrilaterals(A, Sym<REAL>("E"), 0);
  auto time_tris = field_evaluate->evaluate_triangles(A, Sym<REAL>("E"), 0);
  auto time_hexs = field_evaluate->evaluate_hexahedrons(A, Sym<REAL>("E"), 0);
  auto time_prisms = field_evaluate->evaluate_prisms(A, Sym<REAL>("E"), 0);
  auto time_tets = field_evaluate->evaluate_tetrahedrons(A, Sym<REAL>("E"), 0);
  auto time_pyrs = field_evaluate->evaluate_pyramids(A, Sym<REAL>("E"), 0);

  nprint(time_quads, time_tris, time_hexs, time_prisms, time_tets, time_pyrs);

  A->free();
  sycl_target->free();
  mesh->free();
  return err;
}
