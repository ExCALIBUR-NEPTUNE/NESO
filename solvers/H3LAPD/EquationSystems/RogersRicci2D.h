#ifndef H3LAPD_ROGERSRICCI2D_SYSTEM_H
#define H3LAPD_ROGERSRICCI2D_SYSTEM_H
#include <SolverUtils/AdvectionSystem.h>
#include <SolverUtils/RiemannSolvers/RiemannSolver.h>

#include "DriftReducedSystem.hpp"

#include <solvers/helpers/implicit_helper.hpp>

namespace LU = Nektar::LibUtilities;
namespace MR = Nektar::MultiRegions;
namespace SD = Nektar::SpatialDomains;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::H3LAPD {

/**
 * @brief An equation system for the drift-wave solver.
 */
class RogersRicci2D : public SU::AdvectionSystem {
public:
  // Friend class to allow the memory manager to allocate shared pointers of
  // this class.
  friend class MemoryManager<RogersRicci2D>;

  /// Creates an instance of this class. This static method is registered with
  /// a factory.
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &session,
         const SD::MeshGraphSharedPtr &graph) {
    SU::EquationSystemSharedPtr p =
        MemoryManager<RogersRicci2D>::AllocateSharedPtr(session, graph);
    p->InitObject();
    return p;
  }

  /// Name of class, used to statically initialise a function pointer for the
  /// create method above.
  static std::string className;

  // Require a fixed variable order; use these indices for clarity
  static constexpr int n_idx = 0;
  static constexpr int Te_idx = 1;
  static constexpr int w_idx = 2;
  static constexpr int phi_idx = 3;

  /// Default destructor.
  virtual ~RogersRicci2D() = default;

protected:
  /// Helper function to define constants.
  NekDouble c(std::string n) {
    auto it = m_c.find(n);

    ASSERTL0(it != m_c.end(), "Unknown constant");

    return it->second;
  }

  /// Map of known constants
  std::map<std::string, NekDouble> m_c = {
      {"T_e", 6.0},        {"L_z", 18.0},   {"n_0", 2.0e18}, {"m_i", 6.67e-27},
      {"omega_ci", 9.6e5}, {"lambda", 3.0}, {"R", 0.5}};

  /// Storage for the drift velocity. The outer index is dimension, and inner
  /// index the solution nodes (in physical space).
  Array<OneD, Array<OneD, NekDouble>> m_driftVel;
  /// Storage for the dot product of drift velocity with element edge normals,
  /// required for the DG formulation.
  Array<OneD, NekDouble> m_traceVn;
  /// A Advection object, which abstracts the calculation of the
  /// \f$ \nabla\cdot\mathbf{F} \f$ operator using different approaches.
  SU::AdvectionSharedPtr m_advObject;
  /// A Riemann solver object to solve numerical fluxes arising from DG: in
  /// this case a simple upwind.
  SU::RiemannSolverSharedPtr m_riemannSolver;
  /// Helper object for fully-implicit solve.
  std::shared_ptr<ImplicitHelper> m_implHelper;

  Array<OneD, NekDouble> m_r;

  /// Protected constructor. Since we use a factory pattern, objects should be
  /// constructed via the EquationSystem factory.
  RogersRicci2D(const LU::SessionReaderSharedPtr &session,
                const SD::MeshGraphSharedPtr &graph);

  virtual void v_InitObject(bool DeclareField = true) override;

  void InitialiseNonlinSysSolver(void);

  void ExplicitTimeInt(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                       Array<OneD, Array<OneD, NekDouble>> &outarray,
                       const NekDouble time);
  void DoOdeProjection(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                       Array<OneD, Array<OneD, NekDouble>> &outarray,
                       const NekDouble time);
  void GetFluxVector(const Array<OneD, Array<OneD, NekDouble>> &physfield,
                     Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);

  Array<OneD, NekDouble> &GetNormalVelocity();

  int m_npts;
  int m_ndims;
};

} // namespace NESO::Solvers::H3LAPD

#endif
