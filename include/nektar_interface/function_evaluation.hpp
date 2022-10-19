#ifndef __FUNCTION_EVALUATION_H_
#define __FUNCTION_EVALUATION_H_

#include <map>
#include <memory>

#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <neso_particles.hpp>

#include "particle_interface.hpp"

using namespace Nektar::LibUtilities;
using namespace NESO::Particles;

namespace NESO {

/**
 *  Class to evaluate a Nektar++ field at a set of particle locations. It is
 *  assumed that the reference coordinates for the particles have alreadt been
 *  computed by NektarGraphLocalMapperT.
 */
template <typename T> class FieldEvaluate {

private:
  std::shared_ptr<T> field;
  ParticleGroupSharedPtr particle_group;
  SYCLTargetSharedPtr sycl_target;
  CellIDTranslationSharedPtr cell_id_translation;

  // map from Nektar++ geometry ids to Nektar++ expanions ids for the field
  std::map<int, int> geom_to_exp;

public:
  ~FieldEvaluate(){};

  /**
   *  Construct new evaluation object.
   *
   *  @param field Nektar++ field to evaluate at particle positions.
   *  @param particle_group ParticleGroup with positions mapped by
   *  NektarGraphLocalMapperT.
   *  @param cell_id_translation CellIDTranslation used to map between NESO
   *  cell ids and Nektar++ geometry object ids.
   */
  FieldEvaluate(std::shared_ptr<T> field, ParticleGroupSharedPtr particle_group,
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
   *  Evaluate the field at the particle locations and place the result in the
   *  ParticleDat indexed by the passed symbol.
   *
   *  @param sym ParticleDat in the ParticleGroup of this object in which to
   *  place the evaluations.
   */
  template <typename U> inline void evaluate(Sym<U> sym) {

    auto output_dat = (*this->particle_group)[sym];
    auto ref_position_dat =
        (*this->particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")];

    const int nrow_max =
        this->particle_group->mpi_rank_dat->cell_dat.get_nrow_max();
    const int ncol = output_dat->ncomp;
    NESOASSERT(ncol >= 1, "Expected evaluated field to be scalar valued");
    const int particle_ndim = ref_position_dat->ncomp;

    Array<OneD, NekDouble> local_coord(particle_ndim);

    // Get the physvals from the Nektar++ field.
    auto global_physvals = this->field->GetPhys();

    CellDataT<U> output_tmp(this->sycl_target, nrow_max, ncol);
    CellDataT<REAL> ref_positions_tmp(this->sycl_target, nrow_max,
                                      particle_ndim);
    EventStack event_stack;

    const int neso_cell_count =
        this->particle_group->domain->mesh->get_cell_count();
    for (int neso_cellx = 0; neso_cellx < neso_cell_count; neso_cellx++) {
      // Get the reference positions from the particle in the cell
      ref_position_dat->cell_dat.get_cell_async(neso_cellx, ref_positions_tmp,
                                                event_stack);
      event_stack.wait();

      // Get the nektar++ geometry id that corresponds to this NESO cell id
      const int nektar_geom_id =
          this->cell_id_translation->map_to_nektar[neso_cellx];

      // Map from the geometry id to the expansion id for the field.
      NESOASSERT(this->geom_to_exp.count(nektar_geom_id),
                 "Could not find expansion id for geom id");
      const int nektar_expansion_id = this->geom_to_exp[nektar_geom_id];
      NESOASSERT(particle_ndim >= this->field->GetCoordim(nektar_expansion_id),
                 "mismatch in coordinate size");

      // Get the expansion object that corresponds to this expansion id
      auto nektar_expansion = this->field->GetExp(nektar_expansion_id);

      // Get the physvals required to evaluate the function in the expansion
      // object that corresponds to the nektar++ geom/NESO cell
      auto physvals =
          global_physvals + this->field->GetPhys_Offset(nektar_expansion_id);

      const int nrow = output_dat->cell_dat.nrow[neso_cellx];
      for (int rowx = 0; rowx < nrow; rowx++) {

        // read the reference position from the particle
        for (int dimx = 0; dimx < particle_ndim; dimx++) {
          local_coord[dimx] = ref_positions_tmp[dimx][rowx];
        }

        // evaluate the field at the reference position of the particle
        const U phys_eval =
            nektar_expansion->StdPhysEvaluate(local_coord, physvals);

        for (int dimx = 0; dimx < ncol; dimx++) {
          output_tmp[dimx][rowx] = phys_eval;
        }
      }

      // write the function evaluations back to the particle
      output_dat->cell_dat.set_cell_async(neso_cellx, output_tmp, event_stack);
    }
    event_stack.wait();
  }
};

} // namespace NESO

#endif
