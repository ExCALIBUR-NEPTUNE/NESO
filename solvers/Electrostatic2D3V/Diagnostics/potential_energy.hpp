#ifndef __NESOSOLVERS_ELECTROSTATIC2D3V_POTENTIALENERGY_HPP__
#define __NESOSOLVERS_ELECTROSTATIC2D3V_POTENTIALENERGY_HPP__

#include <memory>
#include <mpi.h>

#include <nektar_interface/function_evaluation.hpp>
#include <neso_particles.hpp>
using namespace NESO;
using namespace NESO::Particles;

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
using namespace Nektar;

#include "field_mean.hpp"

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
  ParticleGroupSharedPtr particle_group;
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
                  ParticleGroupSharedPtr particle_group,
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
        ParticleDat(this->particle_group->sycl_target,
                    ParticleProp(Sym<REAL>("ELEC_PIC_PE"), 1),
                    this->particle_group->domain->mesh->get_cell_count()));

    this->field_evaluate = std::make_shared<FieldEvaluate<T>>(
        this->field, this->particle_group, cell_id_translation, false);

    this->field_mean = std::make_shared<FieldMean<T>>(this->field);
  }

  /**
   *  Compute the current energy of the field.
   */
  inline double compute() {

    this->field_evaluate->evaluate(Sym<REAL>("ELEC_PIC_PE"));

    auto t0 = profile_timestamp();
    auto ga_energy = std::make_shared<GlobalArray<REAL>>(
        this->particle_group->sycl_target, 1, 0.0);
    const double k_potential_shift = -this->field_mean->get_mean();

    particle_loop(
        "PotentialEnergy::compute", this->particle_group,
        [=](auto k_Q, auto k_PHI, auto k_ga_energy) {
          const REAL phi = k_PHI.at(0);
          const REAL q = k_Q.at(0);
          const REAL tmp_contrib = q * (phi + k_potential_shift);
          k_ga_energy.add(0, tmp_contrib);
        },
        Access::read(Sym<REAL>("Q")), Access::read(Sym<REAL>("ELEC_PIC_PE")),
        Access::add(ga_energy))
        ->execute();

    // The factor of 1/2 in the electrostatic potential energy calculation.
    this->energy = ga_energy->get().at(0) * 0.5;
    return this->energy;
  }
};

#endif // __NESOSOLVERS_ELECTROSTATIC2D3V_POTENTIALENERGY_HPP__
