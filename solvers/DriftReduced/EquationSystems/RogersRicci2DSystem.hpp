#ifndef __NESOSOLVERS_DRIFTREDUCED_ROGERSRICCI2DSYSTEM_HPP__
#define __NESOSOLVERS_DRIFTREDUCED_ROGERSRICCI2DSYSTEM_HPP__

#include "DriftReducedSystem.hpp"

#include <solvers/helpers/implicit_helper.hpp>

namespace LU = Nektar::LibUtilities;
namespace MR = Nektar::MultiRegions;
namespace SD = Nektar::SpatialDomains;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::DriftReduced {

/**
 * @brief An equation system for Rogers and Ricci's simplied 2D LAPD model.
 */
class RogersRicci2D : public DriftReducedSystem {
public:
  /// Allow the memory manager to allocate shared pointers of this class.
  friend class Nektar::MemoryManager<RogersRicci2D>;

  /// Function to create an instance of this class
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &session,
         const SD::MeshGraphSharedPtr &graph) {
    SU::EquationSystemSharedPtr p =
        Nektar::MemoryManager<RogersRicci2D>::AllocateSharedPtr(session, graph);
    p->InitObject();
    return p;
  }

  /// Name of class, used to statically initialise a function pointer for the
  /// create method above.
  static std::string className;

  /// Require a fixed variable order; use these indices for clarity
  static constexpr int n_idx = 0;
  static constexpr int Te_idx = 1;
  static constexpr int w_idx = 2;
  static constexpr int phi_idx = 3;

  /// Default destructor.
  virtual ~RogersRicci2D() = default;

protected:
  /// Protected constructor (Class is instantiated through the factory only)
  RogersRicci2D(const LU::SessionReaderSharedPtr &session,
                const SD::MeshGraphSharedPtr &graph);

  /// Advection object, which abstracts the calculation of the
  /// \f$ \nabla\cdot\mathbf{F} \f$ operator using different approaches.
  SU::AdvectionSharedPtr adv_obj;
  /// Storage for the drift velocities at each quad point
  Array<OneD, Array<OneD, NekDouble>> drift_vel;
  /// Helper object for fully-implicit solve.
  std::shared_ptr<ImplicitHelper> implicit_helper;
  /// Storage for radial coords of each quad point; used in source terms
  Array<OneD, NekDouble> r;
  /// A Riemann solver object to solve numerical fluxes arising from DG
  SU::RiemannSolverSharedPtr riemann_solver;
  /// Storage for the dot product of drift velocity with element edge normals,
  /// required for the DG formulation.
  Array<OneD, NekDouble> trace_norm_vels;

  virtual void v_InitObject(bool DeclareField = true) override final;

  void
  do_ode_projection(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                    Array<OneD, Array<OneD, NekDouble>> &outarray,
                    const NekDouble time);

  virtual void
  explicit_time_int(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                    Array<OneD, Array<OneD, NekDouble>> &outarray,
                    const NekDouble time) override final;
  void get_flux_vector(const Array<OneD, Array<OneD, NekDouble>> &physfield,
                       Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);

  Array<OneD, NekDouble> &get_norm_vel();

  virtual void
  get_phi_solve_rhs(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                    Array<OneD, NekDouble> &rhs) override final;

  virtual void load_params() override final;

private:
  // Model params
  NekDouble coulomb_log;
  NekDouble L_s;
  NekDouble rho_s0;
  NekDouble T_eps;
  NekDouble r_s;
};

} // namespace NESO::Solvers::DriftReduced

#endif // __NESOSOLVERS_DRIFTREDUCED_ROGERSRICCI2DSYSTEM_HPP__
