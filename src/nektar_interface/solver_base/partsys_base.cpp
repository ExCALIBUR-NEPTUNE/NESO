#include "../../../include/nektar_interface/solver_base/partsys_base.hpp"
#include <SolverUtils/EquationSystem.h>
#include <mpi.h>
#include <nektar_interface/geometry_transport/halo_extension.hpp>
#include <nektar_interface/particle_interface.hpp>
#include <neso_particles.hpp>
#include <type_traits>

namespace NESO::Particles {

ParticleSystemFactory &GetParticleSystemFactory() {
  static ParticleSystemFactory instance;
  return instance;
}

PartSysBase::PartSysBase(const ParticleReaderSharedPtr session,
                         const SD::MeshGraphSharedPtr graph, MPI_Comm comm,
ParticleSystemFactory &GetParticleSystemFactory() {
  static ParticleSystemFactory instance;
  return instance;
}

PartSysBase::PartSysBase(const ParticleReaderSharedPtr session,
                         const SD::MeshGraphSharedPtr graph, MPI_Comm comm,
                         PartSysOptions options)
    : session(session), graph(graph), comm(comm),
      ndim(graph->GetSpaceDimension()) {

  read_params();
  // Store options
  this->options = options;

  // Create interface between particles and nektar++
  this->particle_mesh_interface =
      std::make_shared<ParticleMeshInterface>(graph, 0, this->comm);
  extend_halos_fixed_offset(this->options.extend_halos_offset,
                            this->particle_mesh_interface);
  this->sycl_target =
      std::make_shared<SYCLTarget>(0, particle_mesh_interface->get_comm());
  this->nektar_graph_local_mapper = std::make_shared<NektarGraphLocalMapper>(
      this->sycl_target, this->particle_mesh_interface);
  this->domain = std::make_shared<Domain>(this->particle_mesh_interface,
                                          this->nektar_graph_local_mapper);

  // Set up map between cell indices
  this->cell_id_translation = std::make_shared<CellIDTranslation>(
      this->sycl_target, this->particle_group->cell_id_dat,
      this->particle_mesh_interface);
}

/**
 * @details For each entry in the param_vals map (constructed via report_param),
 * write the value to stdout
 * @see also report_param()
 */
void PartSysBase::add_params_report() {

  std::cout << "Particle settings:" << std::endl;
  for (auto const &[param_lbl, param_str_val] : this->param_vals_to_report) {
    std::cout << "  " << param_lbl << ": " << param_str_val << std::endl;
  }
  std::cout << "============================================================="
               "=========="
            << std::endl
            << std::endl;
}

void PartSysBase::free() {
  if (this->h5part) {
    this->h5part->close();
  }
  this->particle_group->free();
  this->sycl_target->free();
  this->particle_mesh_interface->free();
}

bool PartSysBase::is_output_step(int step) {
  return this->output_freq > 0 && (step % this->output_freq) == 0;
}

void PartSysBase::read_params() {

  // Read total number of particles / number per cell from config
  int num_parts_per_cell, num_parts_tot;
  this->session->LoadParameter(NUM_PARTS_TOT_STR, num_parts_tot, -1);
  this->session->LoadParameter(NUM_PARTS_PER_CELL_STR, num_parts_per_cell, -1);

  if (num_parts_tot > 0) {
    this->num_parts_tot = num_parts_tot;
    if (num_parts_per_cell > 0) {
      nprint("Ignoring value of '" + NUM_PARTS_PER_CELL_STR +
             "' because  "
             "'" +
             NUM_PARTS_TOT_STR + "' was specified.");
    }
  } else {
    if (num_parts_per_cell > 0) {
      // Determine the global number of elements
      const int num_elements_local = this->graph->GetNumElements();
      int num_elements_global;
      MPICHK(MPI_Allreduce(&num_elements_local, &num_elements_global, 1,
                           MPI_INT, MPI_SUM, this->comm));

      // compute the global number of particles
      this->num_parts_tot = ((int64_t)num_elements_global) * num_parts_per_cell;

      report_param("Number of particles per cell/element", num_parts_per_cell);
    } else {
      nprint("Particles disabled (Neither '" + NUM_PARTS_TOT_STR +
             "' or "
             "'" +
             NUM_PARTS_PER_CELL_STR + "' are set)");
    }
  }
  report_param("Total number of particles", this->num_parts_tot);

  // Output frequency
  // ToDo Should probably be unsigned, but complicates use of LoadParameter
  this->session->LoadParameter(PART_OUTPUT_FREQ_STR, this->output_freq, 0);
  report_param("Output frequency (steps)", this->output_freq);
}

void PartSysBase::write(const int step) {
  if (this->h5part) {
    if (this->sycl_target->comm_pair.rank_parent == 0) {
      nprint("Writing particle properties at step", step);
    }
    this->h5part->write();
  } else {
    if (this->sycl_target->comm_pair.rank_parent == 0) {
      nprint("Ignoring call to write particle data because an output file "
             "wasn't set up. init_output() not called?");
    }
  }
};

void PartSysBase::InitSpec() { this->particle_spec = ParticleSpec{}; }

void PartSysBase::InitObject() {
  this->session->LoadParameter(PART_OUTPUT_FREQ_STR, this->output_freq, 0);
  report_param("Output frequency (steps)", this->output_freq);

  // Create ParticleSpec
  this->InitSpec();
  // Create ParticleGroup
  this->particle_group = std::make_shared<ParticleGroup>(
      this->domain, this->particle_spec, this->sycl_target);

  this->SetUpParticles();
}


} // namespace NESO::Particles
