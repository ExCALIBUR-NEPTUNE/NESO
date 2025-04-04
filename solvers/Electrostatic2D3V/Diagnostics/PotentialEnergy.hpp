#ifndef __NESOSOLVERS_ELECTROSTATIC2D3V_POTENTIALENERGY_HPP__
#define __NESOSOLVERS_ELECTROSTATIC2D3V_POTENTIALENERGY_HPP__

#include <memory>
#include <mpi.h>

#include "FieldMean.hpp"
#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <nektar_interface/function_evaluation.hpp>
#include <neso_particles.hpp>

namespace NP = NESO::Particles;
using Nektar::Array;
using Nektar::NekDouble;
using Nektar::OneD;

namespace NESO::Solvers::Electrostatic2D3V {

/**
 *  Class to compute and write to a HDF5 file the integral of a function
 *  squared.
 */
template <typename T> class PotentialEnergy {
private:
  Array<OneD, NekDouble> phys_values;
  int num_quad_points;

  std::shared_ptr<FieldEvaluate<T>> field_evaluate;
  std::shared_ptr<FieldMean<T>> field_mean;

public:
  /// The Nektar++ field of interest.
  std::shared_ptr<T> field;
  /// In use ParticleGroup
  NP::ParticleGroupSharedPtr particle_group;
  /// The MPI communicator used by this instance.
  MPI_Comm comm;
  /// The last field energy that was computed on call to write.
  double energy;
  /*
   *  Create new instance.
   *
   *  @param field Nektar++ field (DisContField, ContField) to use.
   *  @param particle_group ParticleGroup to use.
   *  @param cell_id_translation CellIDTranslation to use.
   *  @param filename Filename of HDF5 output file.
   *  @param comm MPI communicator (default MPI_COMM_WORLD).
   */
  PotentialEnergy(std::shared_ptr<T> field,
                  NP::ParticleGroupSharedPtr particle_group,
                  std::shared_ptr<CellIDTranslation> cell_id_translation,
                  MPI_Comm comm = MPI_COMM_WORLD)
      : field(field), particle_group(particle_group), comm(comm) {

    int flag;
    MPICHK(MPI_Initialized(&flag));
    ASSERTL1(flag, "MPI is not initialised");

    // create space to store u^2
    this->num_quad_points = this->field->GetNpoints();
    this->phys_values = Array<OneD, NekDouble>(num_quad_points);

    this->particle_group->add_particle_dat(
        NP::ParticleDat(this->particle_group->sycl_target,
                        NP::ParticleProp(NP::Sym<NP::REAL>("ELEC_PIC_PE"), 1),
                        this->particle_group->domain->mesh->get_cell_count()));

    this->field_evaluate = std::make_shared<FieldEvaluate<T>>(
        this->field, this->particle_group, cell_id_translation, false);

    this->field_mean = std::make_shared<FieldMean<T>>(this->field);
  }

  /**
   *  Compute the current energy of the field.
   */
  inline double compute() {

    this->field_evaluate->evaluate(NP::Sym<NP::REAL>("ELEC_PIC_PE"));

    auto t0 = NP::profile_timestamp();
    auto ga_energy = std::make_shared<NP::GlobalArray<NP::REAL>>(
        this->particle_group->sycl_target, 1, 0.0);
    const double k_potential_shift = -this->field_mean->get_mean();

    NP::particle_loop(
        "PotentialEnergy::compute", this->particle_group,
        [=](auto k_Q, auto k_PHI, auto k_ga_energy) {
          const NP::REAL phi = k_PHI.at(0);
          const NP::REAL q = k_Q.at(0);
          const NP::REAL tmp_contrib = q * (phi + k_potential_shift);
          k_ga_energy.add(0, tmp_contrib);
        },
        NP::Access::read(NP::Sym<NP::REAL>("Q")),
        NP::Access::read(NP::Sym<NP::REAL>("ELEC_PIC_PE")),
        NP::Access::add(ga_energy))
        ->execute();

    // The factor of 1/2 in the electrostatic potential energy calculation.
    this->energy = ga_energy->get().at(0) * 0.5;
    return this->energy;
  }
};

} // namespace NESO::Solvers::Electrostatic2D3V

#endif // __NESOSOLVERS_ELECTROSTATIC2D3V_POTENTIALENERGY_HPP__
