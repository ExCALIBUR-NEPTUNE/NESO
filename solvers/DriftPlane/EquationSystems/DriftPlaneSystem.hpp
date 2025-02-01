#ifndef DRIFTPLANE_DRIFTPLANESYSTEM_H
#define DRIFTPLANE_DRIFTPLANESYSTEM_H

#include "nektar_interface/solver_base/empty_partsys.hpp"
#include "nektar_interface/solver_base/time_evolved_eqnsys_base.hpp"

#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <SolverUtils/AdvectionSystem.h>
#include <SolverUtils/Core/Misc.h>
#include <SolverUtils/RiemannSolvers/RiemannSolver.h>

#include <solvers/solver_callback_handler.hpp>

namespace LU = Nektar::LibUtilities;
namespace MR = Nektar::MultiRegions;
namespace SD = Nektar::SpatialDomains;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::DriftPlane {

/**
 * @brief Abstract base class for drift-plane equation systems.
 *
 */
class DriftPlaneSystem
    : public TimeEvoEqnSysBase<SU::UnsteadySystem, Particles::EmptyPartSys> {
public:
  friend class MemoryManager<DriftPlaneSystem>;

  inline virtual ~DriftPlaneSystem() {}

protected:
  DriftPlaneSystem(const LU::SessionReaderSharedPtr &session,
                   const SD::MeshGraphSharedPtr &graph);
                   
  /// Advection object
  SU::AdvectionSharedPtr adv_obj;
  /// Advection type
  std::string adv_type;

  /// Storage for the sheath divergence.
  Array<OneD, NekDouble> div_sheath;

  /// Storage for the drift velocity. The outer index is dimension, and inner
  /// index the solution nodes (in physical space).
  Array<OneD, Array<OneD, NekDouble>> drift_vel;

  /// Flag to move dndy into lhs convective term; maybe remove
  bool dndy;

  SU::RiemannSolverSharedPtr riemann_solver;

  /// Riemann solver type
  std::string riemann_type;

  /// Storage for the dot product of drift velocity with element edge normals,
  /// required for the DG formulation.
  Array<OneD, NekDouble> trace_vnorm;

  /// Model params
  NekDouble B;
  NekDouble e;
  NekDouble Lpar;
  NekDouble Rxy;
  NekDouble T_e;

  Array<OneD, NekDouble> calc_div_sheath_closure(
      const Array<OneD, const Array<OneD, NekDouble>> &in_arr);

  void calc_drift_velocity();

  virtual void create_riemann_solver();

  void
  do_ode_projection(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                    Array<OneD, Array<OneD, NekDouble>> &out_arr,
                    const NekDouble time);

  virtual void
  explicit_time_int(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                    Array<OneD, Array<OneD, NekDouble>> &out_arr,
                    const NekDouble time) = 0;

  void get_flux_vector(const Array<OneD, Array<OneD, NekDouble>> &physfield,
                       Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);
  Array<OneD, NekDouble> &get_normal_velocity();
  Array<OneD, NekDouble> &get_trace_norm_y();

  virtual void load_params() override;

  void solve_phi(const Array<OneD, const Array<OneD, NekDouble>> &in_arr);

  virtual void v_GenerateSummary(SU::SummaryList &s) override;
  virtual void v_InitObject(bool DeclareField) override;
};

} // namespace NESO::Solvers::DriftPlane
#endif // H3LAPD_DRIFTPLANESYSTEM_H
