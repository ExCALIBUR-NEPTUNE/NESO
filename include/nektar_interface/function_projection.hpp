#ifndef __FUNCTION_PROJECTION_H_
#define __FUNCTION_PROJECTION_H_

#include <cmath>
#include <map>
#include <memory>

#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <MultiRegions/DisContField.h>
#include <neso_particles.hpp>

#include "function_basis_projection.hpp"
#include "function_coupling_base.hpp"
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
template <typename T> class FieldProject : GeomToExpansionBuilder {

private:
  std::vector<std::shared_ptr<T>> fields;
  std::vector<ParticleGroupSharedPtr> particle_groups;
  CellIDTranslationSharedPtr cell_id_translation;

  // map from Nektar++ geometry ids to Nektar++ expanions ids for the field
  std::map<int, int> geom_to_exp;

  std::shared_ptr<FunctionProjectBasis<T>> function_project_basis;

  bool is_testing;
  std::vector<double> testing_device_rhs;
  std::vector<double> testing_host_rhs;

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
  FieldProject(std::shared_ptr<T> field,
               ParticleGroupSharedPtr particle_group,
               CellIDTranslationSharedPtr cell_id_translation)
      : FieldProject(std::vector<std::shared_ptr<T>>({field}),
                     std::vector<ParticleGroupSharedPtr>({particle_group}),
                     cell_id_translation) { };

  /**
   * Construct a new instance to project particle data from the given
   * ParticleGroup on the given Nektar++ field.
   *
   * @param fields Nektar++ fields to project particle data onto, e.g. a
   * DisContField instance.
   * @param particle_group ParticleGroup which is the source of particle data.
   * @param cell_id_translation CellIDTranslation instance (provides the map
   * from particle cell indices to Nektar++ geometry ids).
   */
  FieldProject(std::vector<std::shared_ptr<T>> fields,
               ParticleGroupSharedPtr particle_group,
               CellIDTranslationSharedPtr cell_id_translation)
      : FieldProject(fields,
                     std::vector<ParticleGroupSharedPtr>({particle_group}),
                     cell_id_translation) { };
  /**
   * Construct a new instance to project particle data from the given
   * ParticleGroup on the given Nektar++ field.
   *
   * @param field Nektar++ field to project particle data onto, e.g. a
   * DisContField instance.
   * @param particle_groups ParticleGroup vector which is the source of particle data.
   * @param cell_id_translation CellIDTranslation instance (provides the map
   * from particle cell indices to Nektar++ geometry ids).
   */
  FieldProject(std::shared_ptr<T> field,
               std::vector<ParticleGroupSharedPtr> particle_groups,
               CellIDTranslationSharedPtr cell_id_translation)
      : FieldProject(std::vector<std::shared_ptr<T>>({fields}),
                     particle_groups,
                     cell_id_translation) { };


  /**
   * Construct a new instance to project particle data from the given
   * ParticleGroup on the given Nektar++ field.
   *
   * @param fields Nektar++ fields to project particle data onto, e.g. a
   * DisContField instance.
   * @param particle_groups ParticleGroup vector which is the source of particle data.
   * @param cell_id_translation CellIDTranslation instance (provides the map
   * from particle cell indices to Nektar++ geometry ids).
   */
  FieldProject(std::vector<std::shared_ptr<T>> fields,
               std::vector<ParticleGroupSharedPtr> particle_groups,
               CellIDTranslationSharedPtr cell_id_translation)
      : fields(fields), particle_groups(particle_groups),
        cell_id_translation(cell_id_translation) {

    NESOASSERT(this->fields.size() > 0, "No fields passed.");

    auto mesh = std::dynamic_pointer_cast<ParticleMeshInterface>(
        particle_groups[0]->domain->mesh);
    this->function_project_basis = std::make_shared<FunctionProjectBasis<T>>(
        this->fields[0], mesh, cell_id_translation);
    this->is_testing = false;

    // build the map from geometry ids to expansion ids
    build_geom_to_expansion_map(this->fields[0], this->geom_to_exp);

    // build the map from geometry ids to expansion ids
    auto expansions = this->fields[0]->GetExp();
    const int num_expansions = (*expansions).size();
    if (this->fields.size() > 1) {
      // check all the fields have the same properties

      const int ncoeffs = this->fields[0]->GetNcoeffs();
      const int npoints = this->fields[0]->GetTotPoints();

      for (auto &fieldx : this->fields) {
        auto test_expansion = fieldx->GetExp();

        NESOASSERT(fieldx->GetNcoeffs() == ncoeffs, "Missmatch of ncoeffs.");

        NESOASSERT(fieldx->GetTotPoints() == npoints, "Missmatch of npoints.");

        NESOASSERT(test_expansion->size() == num_expansions,
                   "Missmatch of expansion count between fields.");

        for (int ex = 0; ex < num_expansions; ex++) {
          const int ndim = (*expansions)[0]->GetCoordim();

          NESOASSERT((*test_expansion)[0]->GetCoordim() == ndim,
                     "Missmatch of field dimension count.");

          NESOASSERT((*test_expansion)[ex]->GetGeom()->GetGlobalID() ==
                         (*expansions)[ex]->GetGeom()->GetGlobalID(),
                     "Mesh missmatch between fields (Geom IDs).");

          for (int dx = 0; dx < ndim; dx++) {
            NESOASSERT((*test_expansion)[ex]->GetBasisNumModes(dx) ==
                           (*expansions)[ex]->GetBasisNumModes(dx),
                       "Mesh missmatch between number of modes.");

            NESOASSERT((*test_expansion)[ex]->GetBasis(dx)->GetBasisType() ==
                           (*expansions)[ex]->GetBasis(dx)->GetBasisType(),
                       "Basis type missmatch between fields.");
          }
        }
      }
    }
  };

  /**
   * Enable recording of computed values for testing.
   */
  inline void testing_enable() { this->is_testing = true; }

  /**
   * Get the last computed rhs vectors for host and device.
   *
   * @param rhs_host (output) Pointer to last computed RHS values in field
   * order on host.
   * @param rhs_device (output) Pointer to last computed RHS values in field
   * order on SYCL Device.
   */
  inline void testing_get_rhs(double **rhs_host, double **rhs_device) {
    NESOASSERT(this->is_testing, "Calling testing_get_rhs without calling "
                                 "testing_enable will not work.");
    *rhs_host = this->testing_host_rhs.data();
    *rhs_device = this->testing_device_rhs.data();
  }

  /**
   * Project the particle data from the given ParticleDats onto the Nektar++
   * fields. It is assumed that the reference positions of particles have aleady
   * been computed and are stored on the particles. This reference position
   * computation is performed as part of the cell binning process
   * implemented in NektarGraphLocalMapperT.
   *
   * @param syms Vector of ParticleDats in the ParticleGroup to use as the
   * particle weights.
   * @param components Vector of components to index into the ParticleDats, i.e.
   * if the ParticleDat has two components a 1 in this vector extracts the
   * second component.
   */
  template <typename U>
  inline void project_host(std::vector<Sym<U>> syms,
                           std::vector<int> components,
                           const int particle_group_index,
                           const bool is_first_pass,
                           const bool is_final_pass) {

    const int nfields = this->fields.size();
    NESOASSERT(syms.size() == nfields, "Bad number of Sym objects passed. i.e. "
                                       "Does not match number of fields.");
    NESOASSERT(components.size() == nfields,
               "Bad number of components passed. i.e. Does not match number of "
               "fields.");

    auto ref_position_dat =
        (*this->particle_groups[particle_group_index])[Sym<REAL>("NESO_REFERENCE_POSITIONS")];

    // This is the same for all the ParticleDats
    const int nrow_max =
        this->particle_groups[particle_group_index]->mpi_rank_dat->cell_dat.get_nrow_max();

    // space to store the reference positions for each particle
    const int particle_ndim = ref_position_dat->ncomp;
    const auto sycl_target = this->particle_groups[particle_group_index]->sycl_target;
    CellDataT<REAL> ref_positions_tmp(sycl_target, nrow_max, particle_ndim);

    // space on host to store the values TODO find a way to fetch only one
    // component
    std::vector<std::unique_ptr<CellDataT<U>>> input_tmp;
    input_tmp.reserve(nfields);
    // space on host for the reference positions
    std::vector<ParticleDatSharedPtr<U>> input_dats;
    input_dats.reserve(nfields);

    // should be the same for all fields
    const int ncoeffs = this->fields[0]->GetNcoeffs();

    // space for the new RHS values for the projection
    std::vector<std::unique_ptr<Array<OneD, NekDouble>>> global_phi;
    global_phi.reserve(nfields);

    for (int symx = 0; symx < nfields; symx++) {
      auto dat_tmp = (*this->particle_groups[particle_group_index])[syms[symx]];
      input_dats.push_back(dat_tmp);
      const int ncol = dat_tmp->ncomp;
      NESOASSERT((0 <= components[symx]) && (components[symx] < ncol),
                 "Component to project out of range.");

      // allocate space to store the particle values
      input_tmp.push_back(
          std::make_unique<CellDataT<U>>(sycl_target, nrow_max, ncol));

      // allocate space to store the RHS values of the projection
      global_phi.push_back(std::make_unique<Array<OneD, NekDouble>>(ncoeffs));
    }

    // zero the output arrays
    for (int fieldx = 0; fieldx < nfields; fieldx++) {
      for (int cx = 0; cx < ncoeffs; cx++) {
        (*global_phi[fieldx])[cx] = 0.0;
      }
    }

    // EvaluateBasis is called with this argument holding the reference position
    Array<OneD, NekDouble> local_coord(particle_ndim);

    // event stack for copy operations
    EventStack event_stack;

    // Number of mesh cells containing particles
    const int neso_cell_count =
        this->particle_groups[particle_group_index]->domain->mesh->get_cell_count();

    // For each cell in the mesh
    for (int neso_cellx = 0; neso_cellx < neso_cell_count; neso_cellx++) {

      // Get the source values.
      for (int fieldx = 0; fieldx < nfields; fieldx++) {
        input_dats[fieldx]->cell_dat.get_cell_async(
            neso_cellx, *input_tmp[fieldx], event_stack);
      }
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
      const int coordim = this->fields[0]->GetCoordim(nektar_expansion_id);
      NESOASSERT(particle_ndim >= coordim, "mismatch in coordinate size");

      // Get the expansion object that corresponds to the first expansion
      auto nektar_expansion_0 = this->fields[0]->GetExp(nektar_expansion_id);
      // get the number of modes in this expansion
      const int num_modes = nektar_expansion_0->GetNcoeffs();
      // get the offset in the expansion values for this mesh cell
      const auto expansion_offset =
          this->fields[0]->GetCoeff_Offset(nektar_expansion_id);

      for (int fieldx = 0; fieldx < nfields; fieldx++) {
        NESOASSERT(this->fields[fieldx]->GetCoeff_Offset(nektar_expansion_id) ==
                       expansion_offset,
                   "Missmatch in expansion offset.");
      }

      // wait for the copy of particle data to host
      event_stack.wait();
      const int nrow = input_dats[0]->cell_dat.nrow[neso_cellx];
      // for each particle in the cell
      for (int rowx = 0; rowx < nrow; rowx++) {
        // read the reference position from the particle
        for (int dimx = 0; dimx < particle_ndim; dimx++) {
          local_coord[dimx] = ref_positions_tmp[dimx][rowx];
        }

        // for each mode in the expansion evaluate the basis function
        // corresponding to that node at the location of the particle, then
        // re-weight with the value on the particle.
        for (int modex = 0; modex < num_modes; modex++) {
          const double phi_j =
              nektar_expansion_0->PhysEvaluateBasis(local_coord, modex);
          // for each field reuse the computed basis function value
          for (int fieldx = 0; fieldx < nfields; fieldx++) {
            const int componentx = components[fieldx];
            // read the component from the particle for the nth field
            const auto quantity = (*input_tmp[fieldx])[componentx][rowx];

            // offset to this dof in this field
            auto phi = (*global_phi[fieldx]) + expansion_offset;
            phi[modex] += phi_j * quantity;
          }
        }
      }
    }

    if (this->is_testing) {
      this->testing_host_rhs.clear();
      this->testing_host_rhs.reserve(nfields * ncoeffs);
    }

    this->finalise_projection(global_phi, ncoeffs, is_first_pass, is_final_pass);
  } // project host

  inline void finalise_projection(
    std::vector<std::unique_ptr<Array<OneD, NekDouble>>>& global_phi,
    int ncoeffs,
    bool is_first_pass = true,
    bool is_final_pass = true) {

    // solve mass matrix system to do projections
    Array<OneD, NekDouble> global_coeffs = Array<OneD, NekDouble>(ncoeffs);
    const int tot_points = this->fields[0]->GetTotPoints();
    Array<OneD, NekDouble> global_phys(tot_points);
    const int nfields = this->fields.size();
    for (int fieldx = 0; fieldx < nfields; fieldx++) {
      for (int cx = 0; cx < ncoeffs; cx++) {
        const double rhs_tmp = (*global_phi[fieldx])[cx];
        NESOASSERT(std::isfinite(rhs_tmp), "A projection RHS value is nan.");

        if (this->is_testing) {
          this->testing_host_rhs.push_back(rhs_tmp);
        }
      }

      // Solve the mass matrix system
      multiply_by_inverse_mass_matrix(this->fields[fieldx],
                                      *global_phi[fieldx],
                                      global_coeffs);

      // make sure that the coeffs are incremented by adding back the initial values
      if (!is_first_pass) {
        const auto global_coeffs_init_values = this->fields[fieldx]->GetCoeffs();
        for (int cx = 0; cx < ncoeffs; cx++) {
          global_coeffs[cx] += global_coeffs_init_values[cx];
        }
      }

      // set the coeffs back on the field
      for (int cx = 0; cx < ncoeffs; cx++) {
        NESOASSERT(std::isfinite(global_coeffs[cx]),
                   "A projection LHS value is nan.");
        // set the coefficients on the function
        this->fields[fieldx]->SetCoeff(cx, global_coeffs[cx]);
      }

      // on final pass make sure the phys values are set
      if (is_final_pass) {
        // set the values at the quadrature points of the function to correspond
        // to the DOFs we just computed.
        for (int cx = 0; cx < tot_points; cx++) {
          global_phys[cx] = 0.0;
        }
        this->fields[fieldx]->BwdTrans(global_coeffs, global_phys);
        this->fields[fieldx]->SetPhys(global_phys);
      }
    }
  } // project_finalise

  /**
   * Project the particle data from the given ParticleDat onto the Nektar++
   * field. It is assumed that the reference positions of particles have aleady
   * been computed and are stored on the particles. This reference position
   * computation is performed as part of the cell binning process
   * implemented in NektarGraphLocalMapperT.
   *
   * @param sym ParticleDat in the ParticleGroup to use as the particle weights.
   */
  template <typename U> inline void project(std::vector<Sym<U>> syms,
                                            std::vector<int> components = {0}) {
    for (int i = 0; i < this->particle_groups.size(); ++i) {
      const int particle_group_index = i;
      const bool is_first_pass = (i == 0);
      const bool is_final_pass = (i == this->particle_groups.size() - 1);
      this->project(syms,
                    components,
                    particle_group_index,
                    is_first_pass,
                    is_final_pass);
    }
  }

  template <typename U> inline void project(Sym<U> sym) {
    std::vector<Sym<U>> syms(1, sym);
    this->project(syms);
  }

  /**
   * Project the particle data from the given ParticleDats onto the Nektar++
   * fields. It is assumed that the reference positions of particles have aleady
   * been computed and are stored on the particles. This reference position
   * computation is performed as part of the cell binning process
   * implemented in NektarGraphLocalMapperT.
   *
   * @param syms Vector of ParticleDats in the ParticleGroup to use as the
   * particle weights.
   * @param components Vector of components to index into the ParticleDats, i.e.
   * if the ParticleDat has two components a 1 in this vector extracts the
   * second component.
   */
  template <typename U>
  inline void project(std::vector<Sym<U>> syms,
                      std::vector<int> components,
                      const int particle_group_index,
                      const bool is_first_pass,
                      const bool is_final_pass) {

    const int nfields = this->fields.size();
    NESOASSERT(syms.size() == nfields, "Bad number of Sym objects passed. i.e. "
                                       "Does not match number of fields.");
    NESOASSERT(components.size() == nfields,
               "Bad number of components passed. i.e. Does not match number of "
               "fields.");

    // space on host for the reference positions
    std::vector<ParticleDatSharedPtr<U>> input_dats;
    input_dats.reserve(nfields);

    // should be the same for all fields
    const int ncoeffs = this->fields[0]->GetNcoeffs();

    // space for the new RHS values for the projection
    std::vector<std::unique_ptr<Array<OneD, NekDouble>>> global_phi;
    global_phi.reserve(nfields);

    for (int symx = 0; symx < nfields; symx++) {
      auto dat_tmp = (*this->particle_groups[particle_group_index])[syms[symx]];
      input_dats.push_back(dat_tmp);
      const int ncol = dat_tmp->ncomp;
      NESOASSERT((0 <= components[symx]) && (components[symx] < ncol),
                 "Component to project out of range.");
      // allocate space to store the RHS values of the projection
      global_phi.push_back(std::make_unique<Array<OneD, NekDouble>>(ncoeffs));
    }

    for (int fieldx = 0; fieldx < nfields; fieldx++) {
      this->function_project_basis->project(this->particle_groups[particle_group_index],
                                            syms[fieldx],
                                            components[fieldx],
                                            *global_phi[fieldx]);
    }

    if (this->is_testing) {
      this->testing_device_rhs.clear();
      this->testing_device_rhs.reserve(nfields * ncoeffs);
    }

    this->finalise_projection(global_phi, ncoeffs, is_first_pass, is_final_pass);
  } // project

  /**
   * Project the particle data from the given ParticleDat onto the Nektar++
   * field. It is assumed that the reference positions of particles have aleady
   * been computed and are stored on the particles. This reference position
   * computation is performed as part of the cell binning process
   * implemented in NektarGraphLocalMapperT.
   *
   * @param syms ParticleDat vector in the ParticleGroup to use as the particle weights.
   */
  template <typename U> inline void project_host(std::vector<Sym<U>> syms,
                                                 std::vector<int> components = {0}) {
    for (int i = 0; i < this->particle_groups.size(); ++i) {
      const int particle_group_index = i;
      const bool is_first_pass = (i == 0);
      const bool is_final_pass = (i == this->particle_groups.size() - 1);
      this->project_host(syms,
                         components,
                         particle_group_index,
                         is_first_pass,
                         is_final_pass);
    }
  }

  /**
   * Project the particle data from the given ParticleDat onto the Nektar++
   * field. It is assumed that the reference positions of particles have aleady
   * been computed and are stored on the particles. This reference position
   * computation is performed as part of the cell binning process
   * implemented in NektarGraphLocalMapperT.
   *
   * @param sym ParticleDat in the ParticleGroup to use as the particle weights.
   */
  template <typename U> inline void project_host(Sym<U> sym) {
    std::vector<Sym<U>> syms(1, sym);
    this->project_host(syms);
  }



};

} // namespace NESO
#endif
