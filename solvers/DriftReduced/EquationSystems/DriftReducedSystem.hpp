#ifndef __NESOSOLVERS_DRIFTREDUCED_DRIFTREDUCEDSYSTEM_HPP__
#define __NESOSOLVERS_DRIFTREDUCED_DRIFTREDUCEDSYSTEM_HPP__

#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <SolverUtils/AdvectionSystem.h>
#include <SolverUtils/Core/Misc.h>
#include <SolverUtils/EquationSystem.h>
#include <SolverUtils/Forcing/Forcing.h>
#include <SolverUtils/RiemannSolvers/RiemannSolver.h>
#include <nektar_interface/solver_base/time_evolved_eqnsys_base.hpp>
#include <nektar_interface/utilities.hpp>
#include <solvers/solver_callback_handler.hpp>

#include "../ParticleSystems/NeutralParticleSystem.hpp"

namespace LU = Nektar::LibUtilities;
namespace MR = Nektar::MultiRegions;
namespace SD = Nektar::SpatialDomains;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::DriftReduced {

/**
 * @brief Abstract base class for drift-reduced equation systems, including
 * Hasegawa-Wakatani and LAPD.
 *
 */
class DriftReducedSystem
    : public TimeEvoEqnSysBase<SU::UnsteadySystem, NeutralParticleSystem> {
public:
  friend class Nektar::MemoryManager<DriftReducedSystem>;

  /// Free particle system memory on destruction
  inline virtual ~DriftReducedSystem() {}

protected:
  DriftReducedSystem(const LU::SessionReaderSharedPtr &session,
                     const SD::MeshGraphSharedPtr &graph);

  /// Advection object used in the electron density equation
  SU::AdvectionSharedPtr adv_elec;
  /// Storage for ne advection velocities
  Array<OneD, Array<OneD, NekDouble>> adv_vel_elec =
      Array<OneD, Array<OneD, NekDouble>>(3);
  /// Advection type
  std::string adv_type;
  /// Advection object used in the vorticity equation
  SU::AdvectionSharedPtr adv_vort;
  /// Magnetic field vector
  std::vector<NekDouble> Bvec{3};
  /// Magnitude of the magnetic field
  NekDouble Bmag;
  /// Normalised magnetic field vector
  std::vector<NekDouble> b_unit;
  /** Source fields cast to DisContFieldSharedPtr, indexed by name, for use in
   * particle evaluation/projection methods
   */
  std::map<std::string, MR::DisContFieldSharedPtr> discont_fields;
  /// Storage for physical values of the electric field
  Array<OneD, Array<OneD, NekDouble>> Evec{3};
  /// Storage for ExB drift velocity
  Array<OneD, Array<OneD, NekDouble>> ExB_vel{3};
  /// Factor used to set the density floor (n_floor = n_floor_fac * n_ref)
  NekDouble n_floor_fac;
  /// Reference number density
  NekDouble n_ref;
  /// Storage for electron parallel velocities
  Array<OneD, NekDouble> par_vel_elec;
  /// Riemann solver type (used for all advection terms)
  std::string riemann_solver_type;

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
  virtual void load_params() override;
  void solve_phi(const Array<OneD, const Array<OneD, NekDouble>> &in_arr);

  virtual void v_GenerateSummary(SU::SummaryList &s) override;
  virtual void v_InitObject(bool DeclareField) override;
  virtual bool v_PostIntegrate(int step) override;
  virtual bool v_PreIntegrate(int step) override;

  void zero_out_array(Array<OneD, Array<OneD, NekDouble>> &out_arr);

private:
  /// d00 coefficient for Helmsolve
  NekDouble d00;
  /// d11 coefficient for Helmsolve
  NekDouble d11;
  /// d22 coefficient for Helmsolve
  NekDouble d22;
  /// Storage for component of ne advection velocity normal to trace elements
  Array<OneD, NekDouble> norm_vel_elec;
  /// Storage for component of w advection velocity normal to trace elements
  Array<OneD, NekDouble> norm_vel_vort;
  /// Number of particle timesteps per fluid timestep.
  int num_part_substeps;
  /// Particle timestep size.
  double part_timestep;
  /// Number of time steps between particle trajectory step writes.
  int num_write_particle_steps;
  /// Riemann solver object used in electron advection
  SU::RiemannSolverSharedPtr riemann_elec;
  /// Riemann solver object used in vorticity advection
  SU::RiemannSolverSharedPtr riemann_vort;

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
};

} // namespace NESO::Solvers::DriftReduced
#endif // __NESOSOLVERS_DRIFTREDUCED_DRIFTREDUCEDSYSTEM_HPP__
