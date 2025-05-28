#ifndef __FUNCTION_EVALUATION_H_
#define __FUNCTION_EVALUATION_H_

#include <map>
#include <memory>

#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <MultiRegions/ContField.h>
#include <MultiRegions/DisContField.h>

#include <neso_particles.hpp>

#include "function_bary_evaluation.hpp"
#include "function_basis_evaluation.hpp"
#include "particle_interface.hpp"

using namespace Nektar::LibUtilities;
using namespace NESO::Particles;

namespace NESO {

/**
 *  Class to evaluate a Nektar++ field at a set of particle locations. It is
 *  assumed that the reference coordinates for the particles have already been
 *  computed by NektarGraphLocalMapper.
 */
template <typename T> class FieldEvaluate {

private:
  std::shared_ptr<T> field;
  ParticleGroupSharedPtr particle_group;
  SYCLTargetSharedPtr sycl_target;
  CellIDTranslationSharedPtr cell_id_translation;

  const bool derivative;

  // used to compute derivatives
  std::shared_ptr<BaryEvaluateBase<T>> bary_evaluate_base;

  // used for scalar values
  std::shared_ptr<FunctionEvaluateBasis<T>> function_evaluate_basis;

public:
  ~FieldEvaluate(){};

  /**
   *  Construct new evaluation object. FieldEvaluate allows a Nektar++ field
   *  (or the derivative) to be evaluated at particle locations.
   *
   *  @param field Nektar++ field to evaluate at particle positions.
   *  @param particle_group ParticleGroup with positions mapped by
   *  NektarGraphLocalMapper.
   *  @param cell_id_translation CellIDTranslation used to map between NESO
   *  cell ids and Nektar++ geometry object ids.
   *  @param derivative This evaluation object should evaluate the derivative of
   * the field (default false).
   */
  FieldEvaluate(std::shared_ptr<T> field, ParticleGroupSharedPtr particle_group,
                CellIDTranslationSharedPtr cell_id_translation,
                const bool derivative = false)
      : field(field), particle_group(particle_group),
        sycl_target(particle_group->sycl_target),
        cell_id_translation(cell_id_translation), derivative(derivative) {

    if (this->derivative) {
      auto particle_mesh_interface =
          std::dynamic_pointer_cast<ParticleMeshInterface>(
              particle_group->domain->mesh);
      NESOASSERT((particle_mesh_interface->ndim == 2) ||
                     (particle_mesh_interface->ndim == 3),
                 "Derivative evaluation supported in 2D and 3D only.");
      this->bary_evaluate_base = std::make_shared<BaryEvaluateBase<T>>(
          field, particle_mesh_interface, cell_id_translation);
    } else {
      auto mesh = std::dynamic_pointer_cast<ParticleMeshInterface>(
          particle_group->domain->mesh);
      this->function_evaluate_basis =
          std::make_shared<FunctionEvaluateBasis<T>>(field, mesh,
                                                     cell_id_translation);
    }
  };

  /**
   *  Evaluate the field at the particle locations and place the result in the
   *  ParticleDat indexed by the passed symbol. This call assumes that the
   *  reference positions of particles have already been computed and stored in
   *  the NESO_REFERENCE_POSITIONS ParticleDat. This computation of reference
   *  positions is done as part of the cell binning process implemented in
   *  NektarGraphLocalMapper.
   *
   *  @param particle_sub_group ParticleSubGroup created from the ParticleGroup
   *  this evaluation instance was created from or the original ParticleGroup.
   *  @param sym ParticleDat in the ParticleGroup of this object in which to
   *  place the evaluations.
   */
  template <typename GROUP_TYPE, typename U>
  inline void evaluate(std::shared_ptr<GROUP_TYPE> particle_sub_group,
                       Sym<U> sym) {

    if (this->derivative) {
      const auto ndim = this->particle_group->domain->mesh->get_ndim();
      const auto ncomp = this->particle_group->get_dat(sym)->ncomp;
      NESOASSERT(ncomp >= ndim, "Output ParticleDat does not have a sufficient "
                                "number of components.");

      auto global_physvals = this->field->GetPhys();
      const int num_quadrature_points = this->field->GetTotPoints();

      std::vector<Array<OneD, NekDouble>> deriv_physvals(ndim);
      for (int dx = 0; dx < ndim; dx++) {
        deriv_physvals.at(dx) = Array<OneD, NekDouble>(num_quadrature_points);
      }
      for (int dx = 0; dx < ndim; dx++) {
        this->field->PhysDeriv(dx, global_physvals, deriv_physvals.at(dx));
      }

      std::vector<Sym<U>> syms(ndim);
      std::vector<int> components(ndim);
      std::vector<Array<OneD, NekDouble> *> deriv_physvals_ptrs(ndim);
      for (int dx = 0; dx < ndim; dx++) {
        syms.at(dx) = sym;
        components.at(dx) = dx;
        deriv_physvals_ptrs.at(dx) = &deriv_physvals.at(dx);
      }
      this->bary_evaluate_base->evaluate(particle_sub_group, syms, components,
                                         deriv_physvals_ptrs);

    } else {
      auto global_coeffs = this->field->GetCoeffs();
      this->function_evaluate_basis->evaluate(particle_sub_group, sym, 0,
                                              global_coeffs);
    }
  }

  /**
   *  Evaluate the field at the particle locations and place the result in the
   *  ParticleDat indexed by the passed symbol. This call assumes that the
   *  reference positions of particles have already been computed and stored in
   *  the NESO_REFERENCE_POSITIONS ParticleDat. This computation of reference
   *  positions is done as part of the cell binning process implemented in
   *  NektarGraphLocalMapper.
   *
   *  @param sym ParticleDat in the ParticleGroup of this object in which to
   *  place the evaluations.
   */
  template <typename U> inline void evaluate(Sym<U> sym) {
    this->evaluate(this->particle_group, sym);
  }
};

extern template void
FieldEvaluate<MultiRegions::DisContField>::evaluate(Sym<REAL> sym);
extern template void
FieldEvaluate<MultiRegions::ContField>::evaluate(Sym<REAL> sym);
extern template void FieldEvaluate<MultiRegions::DisContField>::evaluate(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym);
extern template void FieldEvaluate<MultiRegions::ContField>::evaluate(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym);
} // namespace NESO

#endif
