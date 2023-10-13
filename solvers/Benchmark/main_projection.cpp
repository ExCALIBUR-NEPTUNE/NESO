#include "main_projection.hpp"

int main_projection(int argc, char *argv[],
                    LibUtilities::SessionReaderSharedPtr session,
                    MeshGraphSharedPtr graph) {
  int err = 0;

  auto particle_system =
      std::make_shared<EvaluateProjectSystem>(session, graph);
  MPI_Comm comm = particle_system->comm;
  const int rank = particle_system->rank;
  const int size = particle_system->size;
  const int num_particles_total = particle_system->num_particles_total;

  auto field_project = std::make_shared<BenchmarkProject<DisContField>>(
      particle_system->field, particle_system->mesh,
      particle_system->cell_id_translation);
  const auto global_coeffs = particle_system->field->GetCoeffs();
  field_project->project_init(global_coeffs);

  // Determine total number of warmup steps
  int num_steps_warmup;
  const int num_steps_warmup_default = 2;
  session->LoadParameter("num_steps_warmup", num_steps_warmup,
                         num_steps_warmup_default);
  // Determine total number of steps
  int num_steps;
  const int num_steps_default = 10;
  session->LoadParameter("num_steps", num_steps, num_steps_default);
  auto A = particle_system->particle_group;

  for (int stepx = 0; stepx < num_steps_warmup; stepx++) {
    if (rank == 0) {
      nprint("WARMUP:", stepx);
    }
    field_project->project_quadrilaterals(A, Sym<REAL>("E"), 0);
    field_project->project_triangles(A, Sym<REAL>("E"), 0);
    field_project->project_hexahedrons(A, Sym<REAL>("E"), 0);
    field_project->project_prisms(A, Sym<REAL>("E"), 0);
    field_project->project_tetrahedrons(A, Sym<REAL>("E"), 0);
    field_project->project_pyramids(A, Sym<REAL>("E"), 0);
    particle_system->advect();
  }

  std::vector<double> times_local = {0, 0, 0, 0, 0, 0};
  for (int stepx = 0; stepx < num_steps; stepx++) {
    if (rank == 0) {
      nprint("RUN:", stepx);
    }
    times_local.at(0) +=
        field_project->project_quadrilaterals(A, Sym<REAL>("E"), 0);
    times_local.at(1) += field_project->project_triangles(A, Sym<REAL>("E"), 0);
    times_local.at(2) +=
        field_project->project_hexahedrons(A, Sym<REAL>("E"), 0);
    times_local.at(3) += field_project->project_prisms(A, Sym<REAL>("E"), 0);
    times_local.at(4) +=
        field_project->project_tetrahedrons(A, Sym<REAL>("E"), 0);
    times_local.at(5) += field_project->project_pyramids(A, Sym<REAL>("E"), 0);
    particle_system->advect();
  }

  std::vector<int> flops_count(6);
  std::vector<int> particle_count(6);
  field_project->get_flop_counts(flops_count);
  field_project->get_particle_counts(A, particle_count);
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

    HDF5ResultsWriter results_writer("benchmark_flops_projection.h5");
    results_writer.add_array("flops_projection", size, 6, flops_global);
    results_writer.add_parameter("mpi_size", size);
    results_writer.add_parameter("num_particles_total", num_particles_total);
    results_writer.add_parameter("num_modes", field_project->get_num_modes());
  }

  particle_system->free();
  return err;
}
