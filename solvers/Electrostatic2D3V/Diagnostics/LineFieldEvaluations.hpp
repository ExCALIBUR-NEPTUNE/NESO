#ifndef __NESOSOLVERS_ELECTROSTATIC2D3V_LINEFIELDEVALUATIONS_HPP__
#define __NESOSOLVERS_ELECTROSTATIC2D3V_LINEFIELDEVALUATIONS_HPP__

#include "../ParticleSystems/ChargedParticles.hpp"
#include <memory>
#include <mpi.h>
#include <neso_particles.hpp>

#include "FieldMean.hpp"

namespace NP = NESO::Particles;

namespace NESO::Solvers::Electrostatic2D3V {
/**
 * Evaluate the value or derivative of the potential field at a set of points
 * that form two lines. One line is in the x direction the other line is in the
 * y direction. Output is a particle trajectory where the particles are fixed
 * along one of the two lines. A particle property INDEX (INT, 1 component)
 * indicates which line the point resides on:
 *  - In 1D (two lines, crossed in a plus sign arrangment):
 *    - 0 for the x direction line, and the index in the y direction
 *    - the index along the x direction line, and 1 for the x direction line
 *  - In a 2D grid:
 *    - the index in x direction and index in the y direction
 * The field evaluations/derivatives are stored on the
 * FIELD_EVALUATION particle property (REAL, 1 component for evaluations, 2
 * components for derivatives).
 */
template <typename T> class LineFieldEvaluations {
private:
  int step;
  bool mean_shift;
  NP::SYCLTargetSharedPtr sycl_target;
  std::shared_ptr<CellIDTranslation> cell_id_translation;
  std::shared_ptr<T> field;
  std::shared_ptr<FieldEvaluate<T>> field_evaluate;
  std::shared_ptr<NP::H5Part> h5part;
  std::unique_ptr<FieldMean<T>> field_mean;

public:
  /// NP::ParticleGroup of interest.
  NP::ParticleGroupSharedPtr particle_group;
  /// The MPI communicator used by this instance.
  MPI_Comm comm;

  /**
   *  Create new instance.
   *
   * @param field Nektar++ field to sample the field or field derivative of
   * along lines in x and y.
   * @param charged_particles A ChargedParticles instance from which the domain
   * will be used.
   * @param nx Number of sample points in the x direction.
   * @param ny Number of sample points in the y direction.
   * @bool derivative Bool to evaluate derivatives instead of field value.
   * @bool mean_shift Bool to enable shifting of evaluations by minus the mean.
   */
  LineFieldEvaluations(std::shared_ptr<T> field,
                       std::shared_ptr<ChargedParticles> charged_particles,
                       const int nx, const int ny,
                       const bool derivative = false,
                       const bool mean_shift = false)
      : field(field), step(0), mean_shift(mean_shift) {

    int flag;
    MPICHK(MPI_Initialized(&flag));
    ASSERTL1(flag, "MPI is not initialised");

    auto domain = charged_particles->particle_group->domain;
    this->sycl_target = charged_particles->sycl_target;

    const int ncomp = (derivative) ? domain->mesh->get_ndim() : 1;

    ParticleSpec particle_spec{
        NP::ParticleProp(NP::Sym<NP::REAL>("P"), 2, true),
        NP::ParticleProp(NP::Sym<NP::INT>("CELL_ID"), 1, true),
        NP::ParticleProp(NP::Sym<NP::REAL>("FIELD_EVALUATION"), ncomp),
        NP::ParticleProp(NP::Sym<NP::INT>("INDEX"), 2)};

    NESOASSERT(nx >= 0, "LineFieldEvaluations: bad nx count");
    NESOASSERT(ny >= 0, "LineFieldEvaluations: bad ny count");

    this->particle_group = std::make_shared<NP::ParticleGroup>(
        domain, particle_spec, this->sycl_target);

    // Setup map between cell indices
    this->cell_id_translation = std::make_shared<CellIDTranslation>(
        this->sycl_target, this->particle_group->cell_id_dat,
        charged_particles->particle_mesh_interface);

    this->field_evaluate = std::make_shared<FieldEvaluate<T>>(
        field, this->particle_group, this->cell_id_translation, derivative);

    const int rank = this->sycl_target->comm_pair.rank_parent;
    if (rank == 0) {
      if (false) {
        NP::ParticleSet initial_distribution(
            nx + ny, this->particle_group->get_particle_spec());

        const double extentx =
            charged_particles->boundary_conditions->global_extent[0];
        const double extenty =
            charged_particles->boundary_conditions->global_extent[1];
        const double hx = extentx / ((double)nx);
        const double hy = extenty / ((double)ny);

        // get the first location
        double tmp_pos =
            charged_particles->boundary_conditions->global_origin[0];
        tmp_pos += 0.5 * hx;
        double tmp_other_dim =
            charged_particles->boundary_conditions->global_origin[1] +
            0.5 * extenty;

        for (int px = 0; px < nx; px++) {
          initial_distribution[NP::Sym<NP::REAL>("P")][px][0] = tmp_pos;
          initial_distribution[NP::Sym<NP::REAL>("P")][px][1] = tmp_other_dim;
          tmp_pos += hx;
          initial_distribution[NP::Sym<NP::INT>("INDEX")][px][0] = 0;
          initial_distribution[NP::Sym<NP::INT>("INDEX")][px][1] = px;
        }

        tmp_pos = charged_particles->boundary_conditions->global_origin[1];
        tmp_pos += 0.5 * hy;
        tmp_other_dim =
            charged_particles->boundary_conditions->global_origin[0] +
            0.5 * extentx;

        for (int px = nx; px < (nx + ny); px++) {
          initial_distribution[NP::Sym<NP::REAL>("P")][px][0] = tmp_other_dim;
          initial_distribution[NP::Sym<NP::REAL>("P")][px][1] = tmp_pos;
          tmp_pos += hy;
          initial_distribution[NP::Sym<NP::INT>("INDEX")][px][0] = 1;
          initial_distribution[NP::Sym<NP::INT>("INDEX")][px][1] = px - nx;
        }

        this->particle_group->add_particles_local(initial_distribution);
      } else {
        NP::ParticleSet initial_distribution(
            nx * ny, this->particle_group->get_particle_spec());

        const double extentx =
            charged_particles->boundary_conditions->global_extent[0];
        const double hx = extentx / ((double)nx);

        const double extenty =
            charged_particles->boundary_conditions->global_extent[1];
        const double hy = extenty / ((double)ny);

        // get the first location
        double init_pos_x =
            charged_particles->boundary_conditions->global_origin[0];
        init_pos_x += 0.5 * hx;

        double init_pos_y =
            charged_particles->boundary_conditions->global_origin[1];
        init_pos_y += 0.5 * hy;

        int ip = 0;
        for (int px = 0; px < nx; px++) {
          for (int py = 0; py < ny; py++) {
            initial_distribution[NP::Sym<NP::REAL>("P")][ip][0] =
                init_pos_x + px * hx;
            initial_distribution[NP::Sym<NP::REAL>("P")][ip][1] =
                init_pos_y + py * hy;
            initial_distribution[NP::Sym<NP::INT>("INDEX")][ip][0] = px;
            initial_distribution[NP::Sym<NP::INT>("INDEX")][ip][1] = py;
            ip += 1;
          }
        }

        this->particle_group->add_particles_local(initial_distribution);
      }
    }

    this->particle_group->hybrid_move();
    this->cell_id_translation->execute();
    this->particle_group->cell_move();

    std::string filename;
    if (derivative) {
      filename = "Electrostatic2D3V_line_field_deriv_evaluations.h5part";
    } else {
      filename = "Electrostatic2D3V_line_field_evaluations.h5part";
    }

    this->h5part = std::make_shared<NP::H5Part>(
        filename, this->particle_group, NP::Sym<NP::INT>("INDEX"),
        NP::Sym<NP::REAL>("FIELD_EVALUATION"));

    if (this->mean_shift) {
      this->field_mean = std::make_unique<FieldMean<T>>(this->field);
    }
  }

  /**
   * Evaluate the derivative of the potential at the points and write a new
   * step to the trajectory.
   *
   * @param step_in Optional override of particle step in trajectory.
   */
  inline void write(int step_in = -1) {
    if (step_in > -1) {
      this->step = step_in;
    }

    // get the field deriv evaluations
    this->field_evaluate->evaluate(NP::Sym<NP::REAL>("FIELD_EVALUATION"));

    if (this->mean_shift) {
      const double k_mean = this->field_mean->get_mean();
      NP::particle_loop(
          "LineFieldEvaluations::write", this->particle_group,
          [=](auto k_EVAL) { k_EVAL.at(0) -= k_mean; },
          NP::Access::write(NP::Sym<NP::REAL>("FIELD_EVALUATION")))
          ->execute();
    }

    this->h5part->write(this->step);
    this->step++;
  }

  /**
   *  Close the output file and free local members as required. Must be called.
   */
  inline void close() {
    this->h5part->close();
    this->particle_group->free();
  }
};

} // namespace NESO::Solvers::Electrostatic2D3V

#endif // __NESOSOLVERS_ELECTROSTATIC2D3V_LINEFIELDEVALUATIONS_HPP__
