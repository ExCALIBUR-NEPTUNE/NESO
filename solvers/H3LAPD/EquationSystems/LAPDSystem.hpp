
#ifndef H3LAPD_LAPD_SYSTEM_H
#define H3LAPD_LAPD_SYSTEM_H

#include "DriftReducedSystem.hpp"
#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <SolverUtils/EquationSystem.h>

namespace LU = Nektar::LibUtilities;
namespace SD = Nektar::SpatialDomains;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::H3LAPD {

/**
 * @brief Initial version of full LAPD equation system.
 */
class LAPDSystem : public DriftReducedSystem {
public:
  friend class MemoryManager<LAPDSystem>;

  /**
   * @brief Creates an instance of this class.
   */
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &session,
         const SD::MeshGraphSharedPtr &graph) {
    SU::EquationSystemSharedPtr p =
        MemoryManager<LAPDSystem>::AllocateSharedPtr(session, graph);
    p->InitObject();
    return p;
  }

  /// Name of class
  static std::string class_name;

protected:
  LAPDSystem(const LU::SessionReaderSharedPtr &session,
             const SD::MeshGraphSharedPtr &graph);
  virtual void
  explicit_time_int(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                    Array<OneD, Array<OneD, NekDouble>> &out_arr,
                    const NekDouble time) override;
  virtual void
  get_phi_solve_rhs(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                    Array<OneD, NekDouble> &rhs) override;
  virtual void load_params() final;
  virtual void v_InitObject(bool DeclareField) override;

private:
  /// Advection object used in the ion momentum equation
  SU::AdvectionSharedPtr adv_ions;
  /// Advection object used for polarisation drift advection
  SU::AdvectionSharedPtr adv_PD;
  /// Storage for ion advection velocities
  Array<OneD, Array<OneD, NekDouble>> adv_vel_ions;
  /** Storage for difference between elec, ion parallel velocities. Has size
   ndim so that it can be used in advection operation */
  Array<OneD, Array<OneD, NekDouble>> adv_vel_PD;
  /// Charge unit
  NekDouble charge_e;
  /// Density-independent part of the Coulomb logarithm; read from config
  NekDouble coulomb_log_const;
  /// Ion mass;
  NekDouble md;
  /// Electron mass;
  NekDouble me;
  /// Factor to convert densities (back) to SI; used in Coulomb logarithm calc
  NekDouble n_to_SI;
  /// Storage for component of Gd advection velocity normal to trace
  /// elements
  Array<OneD, NekDouble> norm_vel_ions;
  /// Storage for component of polarisation drift velocity normal to trace
  /// elements
  Array<OneD, NekDouble> norm_vel_PD;
  /// Pre-factor used when calculating collision frequencies; read from config
  NekDouble nu_ei_const;
  /// Storage for ion parallel velocities
  Array<OneD, NekDouble> par_vel_ions;
  /// Riemann solver object used in ion advection
  SU::RiemannSolverSharedPtr riemann_ions;
  /// Riemann solver object used in polarisation drift advection
  SU::RiemannSolverSharedPtr riemann_PD;
  /// Ion temperature in eV
  NekDouble Td;
  /// Electron temperature in eV
  NekDouble Te;

  void
  add_collision_terms(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                      Array<OneD, Array<OneD, NekDouble>> &out_arr);
  void add_E_par_terms(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                       Array<OneD, Array<OneD, NekDouble>> &out_arr);
  void add_grad_P_terms(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                        Array<OneD, Array<OneD, NekDouble>> &out_arr);
  void calc_collision_freqs(const Array<OneD, NekDouble> &ne,
                            Array<OneD, NekDouble> &coeffs);
  void calc_coulomb_logarithm(const Array<OneD, NekDouble> &ne,
                              Array<OneD, NekDouble> &LogLambda);
  virtual void calc_E_and_adv_vels(
      const Array<OneD, const Array<OneD, NekDouble>> &in_arr) override;
  Array<OneD, NekDouble> &get_adv_vel_norm_ions();
  Array<OneD, NekDouble> &get_adv_vel_norm_PD();
  void
  get_flux_vector_ions(const Array<OneD, Array<OneD, NekDouble>> &physfield,
                       Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);
  void
  get_flux_vector_PD(const Array<OneD, Array<OneD, NekDouble>> &physfield,
                     Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);
};

} // namespace NESO::Solvers::H3LAPD
#endif // H3LAPD_LAPD_SYSTEM_H