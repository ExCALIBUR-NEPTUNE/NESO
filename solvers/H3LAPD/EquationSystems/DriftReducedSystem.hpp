#ifndef H3LAPD_DRIFT_REDUCED_SYSTEM_H
#define H3LAPD_DRIFT_REDUCED_SYSTEM_H

#include "../ParticleSystems/NeutralParticleSystem.hpp"

#include "nektar_interface/utilities.hpp"

#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <SolverUtils/AdvectionSystem.h>
#include <SolverUtils/EquationSystem.h>
#include <SolverUtils/Forcing/Forcing.h>
#include <SolverUtils/RiemannSolvers/RiemannSolver.h>

#include <solvers/solver_callback_handler.hpp>

namespace LU = Nektar::LibUtilities;
namespace MR = Nektar::MultiRegions;
namespace SD = Nektar::SpatialDomains;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::H3LAPD {

/**
 * @brief Abstract base class for drift-reduced equation systems, including
 * Hasegawa-Wakatani and LAPD.
 *
 */
class DriftReducedSystem : virtual public SU::AdvectionSystem {
public:
  friend class MemoryManager<DriftReducedSystem>;

  /// Name of class
  static std::string class_name;

  /// Free particle system memory on destruction
  virtual ~DriftReducedSystem() { m_particle_sys->free(); }

protected:
  DriftReducedSystem(const LU::SessionReaderSharedPtr &session,
                     const SD::MeshGraphSharedPtr &graph);

  /// Advection object used in the electron density equation
  SU::AdvectionSharedPtr m_adv_elec;
  /// Storage for ne advection velocities
  Array<OneD, Array<OneD, NekDouble>> m_adv_vel_elec;
  /// Advection type
  std::string m_adv_type;
  /// Advection object used in the vorticity equation
  SU::AdvectionSharedPtr m_adv_vort;
  /// Magnetic field vector
  std::vector<NekDouble> m_B;
  /// Magnitude of the magnetic field
  NekDouble m_Bmag;
  /// Normalised magnetic field vector
  std::vector<NekDouble> m_b_unit;
  /** Source fields cast to DisContFieldSharedPtr, indexed by name, for use in
   * particle evaluation/projection methods
   */
  std::map<std::string, MR::DisContFieldSharedPtr> m_discont_fields;
  /// Storage for physical values of the electric field
  Array<OneD, Array<OneD, NekDouble>> m_E;
  /// Storage for ExB drift velocity
  Array<OneD, Array<OneD, NekDouble>> m_ExB_vel;
  /// Field name => index mapper
  NESO::NektarFieldIndexMap m_field_to_index;
  /// Names of fields that will be time integrated
  std::vector<std::string> m_int_fld_names;
  /// Factor used to set the density floor (n_floor = m_n_floor_fac * m_n_ref)
  NekDouble m_n_floor_fac;
  /// Reference number density
  NekDouble m_n_ref;
  /// Storage for electron parallel velocities
  Array<OneD, NekDouble> m_par_vel_elec;
  /// Particles system
  std::shared_ptr<NeutralParticleSystem> m_particle_sys;
  /// List of field names required by the solver
  std::vector<std::string> m_required_flds;
  /// Riemann solver type (used for all advection terms)
  std::string m_riemann_solver_type;

  void add_adv_terms(
      std::vector<std::string> field_names,
      const SU::AdvectionSharedPtr adv_obj,
      const Array<OneD, Array<OneD, NekDouble>> &adv_vel,
      const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
      Array<OneD, Array<OneD, NekDouble>> &out_arr, const NekDouble time,
      std::vector<std::string> eqn_labels = std::vector<std::string>());

  void add_density_source(Array<OneD, Array<OneD, NekDouble>> &out_arr);

  void add_particle_sources(std::vector<std::string> target_fields,
                            Array<OneD, Array<OneD, NekDouble>> &out_arr);

  virtual void
  calc_E_and_adv_vels(const Array<OneD, const Array<OneD, NekDouble>> &in_arr);

  virtual void
  explicit_time_int(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                    Array<OneD, Array<OneD, NekDouble>> &out_arr,
                    const NekDouble time) = 0;

  Array<OneD, NekDouble> &
  get_adv_vel_norm(Array<OneD, NekDouble> &trace_vel_norm,
                   const Array<OneD, Array<OneD, NekDouble>> &adv_vel);

  void get_flux_vector(const Array<OneD, Array<OneD, NekDouble>> &fields_vals,
                       const Array<OneD, Array<OneD, NekDouble>> &adv_vel,
                       Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);

  virtual void
  get_phi_solve_rhs(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                    Array<OneD, NekDouble> &rhs) = 0;
  virtual void load_params();
  void solve_phi(const Array<OneD, const Array<OneD, NekDouble>> &in_arr);

  virtual void v_InitObject(bool DeclareField) override;
  virtual bool v_PostIntegrate(int step) override;
  virtual bool v_PreIntegrate(int step) override;

  void zero_out_array(Array<OneD, Array<OneD, NekDouble>> &out_arr);

  //---------------------------------------------------------------------------
  // Debugging
  void print_arr_vals(const Array<OneD, NekDouble> &arr, int num,
                      int stride = 1, std::string label = "",
                      bool all_tasks = false);
  void print_arr_size(const Array<OneD, NekDouble> &arr, std::string label = "",
                      bool all_tasks = false);

private:
  /// d00 coefficient for Helmsolve
  NekDouble m_d00;
  /// d11 coefficient for Helmsolve
  NekDouble m_d11;
  /// d22 coefficient for Helmsolve
  NekDouble m_d22;
  /// Storage for component of ne advection velocity normal to trace elements
  Array<OneD, NekDouble> m_norm_vel_elec;
  /// Storage for component of w advection velocity normal to trace elements
  Array<OneD, NekDouble> m_norm_vel_vort;
  /// Number of particle timesteps per fluid timestep.
  int m_num_part_substeps;
  /// Number of time steps between particle trajectory step writes.
  int m_num_write_particle_steps;
  /// Particle timestep size.
  double m_part_timestep;
  /// Riemann solver object used in electron advection
  SU::RiemannSolverSharedPtr m_riemann_elec;
  /// Riemann solver object used in vorticity advection
  SU::RiemannSolverSharedPtr m_riemann_vort;

  void
  do_ode_projection(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                    Array<OneD, Array<OneD, NekDouble>> &out_arr,
                    const NekDouble time);
  Array<OneD, NekDouble> &get_adv_vel_norm_elec();
  Array<OneD, NekDouble> &get_adv_vel_norm_vort();

  void get_flux_vector_diff(
      const Array<OneD, Array<OneD, NekDouble>> &in_arr,
      const Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &q_field,
      Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &viscous_tensor);
  void
  get_flux_vector_elec(const Array<OneD, Array<OneD, NekDouble>> &fields_vals,
                       Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);
  void
  get_flux_vector_vort(const Array<OneD, Array<OneD, NekDouble>> &fields_vals,
                       Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);

  void validate_fields();
};

} // namespace NESO::Solvers::H3LAPD
#endif // H3LAPD_DRIFT_REDUCED_SYSTEM_H
