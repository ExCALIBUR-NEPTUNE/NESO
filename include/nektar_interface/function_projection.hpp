#ifndef __FUNCTION_PROJECTION_H_
#define __FUNCTION_PROJECTION_H_

#include <cmath>
#include <map>
#include <memory>

#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <MultiRegions/DisContField.h>
#include <neso_particles.hpp>

#include "particle_interface.hpp"

using namespace Nektar::MultiRegions;
using namespace Nektar::LibUtilities;
using namespace NESO::Particles;

namespace NESO {

/**
 *  Generic function to solve a mass matrix system for use in projection Mx=b.
 *
 *  @param field Nektar++ field class.
 *  @param inarray RHS of system, b.
 *  @param outarray solution vector, x.
 */
template <typename T>
inline void
multiply_by_inverse_mass_matrix(std::shared_ptr<T> &field,
                                const Array<OneD, const NekDouble> &inarray,
                                Array<OneD, NekDouble> &outarray) {
  field->MultiplyByInvMassMatrix(inarray, outarray);
}

/**
 *  Function to solve a mass matrix system for use in projection Mx=b.
 *  Specialised verstion for DisContField.
 *
 *  @param field Nektar++ field class.
 *  @param inarray RHS of system, b.
 *  @param outarray solution vector, x.
 */
template <>
inline void
multiply_by_inverse_mass_matrix(std::shared_ptr<DisContField> &field,
                                const Array<OneD, const NekDouble> &inarray,
                                Array<OneD, NekDouble> &outarray) {
  field->MultiplyByElmtInvMass(inarray, outarray);
}

/**
 * Class to project properties stored on ParticleDat objects onto a Nektar++
 * Field instance. Projection is achieved by considering each particle as a
 * Dirac Delta at the particle location with a weight set by the particle
 * property. A standard L2 Galerkin projection is performed to compute the DOFs
 * which are then stored on the Nektar++ object.
 */
template <typename T> class FieldProject {

private:
  std::shared_ptr<T> field;
  ParticleGroupSharedPtr particle_group;
  SYCLTargetSharedPtr sycl_target;
  CellIDTranslationSharedPtr cell_id_translation;

  // map from Nektar++ geometry ids to Nektar++ expanions ids for the field
  std::map<int, int> geom_to_exp;

public:
  ~FieldProject(){};

  /**
   * Construct a new instance to project particle data from the given
   * ParticleGroup on the given Nektar++ field.
   *
   * @param field Nektar++ field to project particle data onto, e.g. a
   * DisContField instance.
   * @param particle_group ParticleGroup which is the source of particle data.
   * @param cell_id_translation CellIDTranslation instance (provides the map
   * from particle cell indices to Nektar++ geometry ids).
   */
  FieldProject(std::shared_ptr<T> field, ParticleGroupSharedPtr particle_group,
               CellIDTranslationSharedPtr cell_id_translation)
      : field(field), particle_group(particle_group),
        sycl_target(particle_group->sycl_target),
        cell_id_translation(cell_id_translation) {

    // build the map from geometry ids to expansion ids
    auto expansions = this->field->GetExp();
    const int num_expansions = (*expansions).size();
    for (int ex = 0; ex < num_expansions; ex++) {
      auto exp = (*expansions)[ex];
      // The indexing in Nektar++ source suggests that ex is the important
      // index if these do not match in future.
      NESOASSERT(ex == exp->GetElmtId(),
                 "expected expansion id to match element id?");
      int geom_gid = exp->GetGeom()->GetGlobalID();
      this->geom_to_exp[geom_gid] = ex;
    }
  };

  /**
   * Project the particle data from the given ParticleDat onto the Nektar++
   * field. It is assumed that the reference positions of particles have aleady
   * been computed and are stored on the particles. This reference position
   * computation is performed as particle of the cell binning process
   * implemented in NektarGraphLocalMapperT.
   *
   * @param sym ParticleDat in the ParticleGroup to use as the particle weights.
   */
  template <typename U> inline void project(Sym<U> sym) {

    auto input_dat = (*this->particle_group)[sym];
    auto ref_position_dat =
        (*this->particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")];

    const int nrow_max =
        this->particle_group->mpi_rank_dat->cell_dat.get_nrow_max();
    const int ncol = input_dat->ncomp;
    const int particle_ndim = ref_position_dat->ncomp;

    Array<OneD, NekDouble> local_coord(particle_ndim);

    const int ncoeffs = this->field->GetNcoeffs();
    Array<OneD, NekDouble> global_coeffs = Array<OneD, NekDouble>(ncoeffs);
    Array<OneD, NekDouble> global_phi = Array<OneD, NekDouble>(ncoeffs);

    CellDataT<U> input_tmp(this->sycl_target, nrow_max, ncol);
    CellDataT<REAL> ref_positions_tmp(this->sycl_target, nrow_max,
                                      particle_ndim);
    EventStack event_stack;

    const int neso_cell_count =
        this->particle_group->domain->mesh->get_cell_count();
    for (int neso_cellx = 0; neso_cellx < neso_cell_count; neso_cellx++) {
      // Get the source values.
      input_dat->cell_dat.get_cell_async(neso_cellx, input_tmp, event_stack);
      // Get the reference positions from the particle in the cell
      ref_position_dat->cell_dat.get_cell_async(neso_cellx, ref_positions_tmp,
                                                event_stack);

      // Get the nektar++ geometry id that corresponds to this NESO cell id
      const int nektar_geom_id =
          this->cell_id_translation->map_to_nektar[neso_cellx];

      // Map from the geometry id to the expansion id for the field.
      NESOASSERT(this->geom_to_exp.count(nektar_geom_id),
                 "Could not find expansion id for geom id");
      const int nektar_expansion_id = this->geom_to_exp[nektar_geom_id];
      const int coordim = this->field->GetCoordim(nektar_expansion_id);
      NESOASSERT(particle_ndim >= coordim, "mismatch in coordinate size");

      // Get the expansion object that corresponds to this expansion id
      auto nektar_expansion = this->field->GetExp(nektar_expansion_id);

      // get the number of modes in this expansion
      const int num_modes = nektar_expansion->GetNcoeffs();

      auto phi = global_phi + this->field->GetCoeff_Offset(nektar_expansion_id);

      // zero the output array
      for (int modex = 0; modex < num_modes; modex++) {
        phi[modex] = 0.0;
      }

      // wait for the copy of particle data to host
      event_stack.wait();
      const int nrow = input_dat->cell_dat.nrow[neso_cellx];
      // for each particle in the cell
      for (int rowx = 0; rowx < nrow; rowx++) {

        const REAL quantity = input_tmp[0][rowx];

        // read the reference position from the particle
        for (int dimx = 0; dimx < particle_ndim; dimx++) {
          local_coord[dimx] = ref_positions_tmp[dimx][rowx];
        }

        // for each mode in the expansion evaluate the basis function
        // corresponding to that node at the location of the particle, then
        // re-weight with the value on the particle.
        for (int modex = 0; modex < num_modes; modex++) {
          const double phi_j =
              nektar_expansion->PhysEvaluateBasis(local_coord, modex);
          phi[modex] += phi_j * quantity;
        }
      }
    }

    for (int cx = 0; cx < ncoeffs; cx++) {
      NESOASSERT(std::isfinite(global_phi[cx]), "A projection RHS value is nan.");
      global_coeffs[cx] = 0.0;
    }

    // Solve the mass matrix system
    multiply_by_inverse_mass_matrix(this->field, global_phi, global_coeffs);

    for (int cx = 0; cx < ncoeffs; cx++) {
      NESOASSERT(std::isfinite(global_coeffs[cx]),
                 "A projection LHS value is nan.");
      // set the coefficients on the function
      this->field->SetCoeff(cx, global_coeffs[cx]);
    }

    // set the values at the quadrature points of the function to correspond to
    // the DOFs we just computed.
    const int tot_points = this->field->GetTotPoints();
    Array<OneD, NekDouble> global_phys(tot_points);
    for (int cx = 0; cx < tot_points; cx++) {
      global_phys[cx] = 0.0;
    }
    this->field->BwdTrans(global_coeffs, global_phys);
    this->field->SetPhys(global_phys);
  }
};

} // namespace NESO
#endif
