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
 * @brief An equation system for Rogers and Ricci's simplied 2D LAPD model.
 */
class RogersRicci2D : public SU::AdvectionSystem {
public:
  /// Allow the memory manager to allocate shared pointers of this class.
  friend class MemoryManager<RogersRicci2D>;

  /// Function to create an instance of this class
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
  /// Store mesh dims and number of quad points as member vars for convenience
  int ndims;
  int npts;
  /// Storage for radial coords of each quad point; used in source terms
  Array<OneD, NekDouble> r;
  /// A Riemann solver object to solve numerical fluxes arising from DG
  SU::RiemannSolverSharedPtr riemann_solver;
  /// Storage for the dot product of drift velocity with element edge normals,
  /// required for the DG formulation.
  Array<OneD, NekDouble> trace_norm_vels;

  virtual void v_InitObject(bool DeclareField = true) override;

  void
  do_ode_projection(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                    Array<OneD, Array<OneD, NekDouble>> &outarray,
                    const NekDouble time);

  void
  explicit_time_int(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                    Array<OneD, Array<OneD, NekDouble>> &outarray,
                    const NekDouble time);
  void get_flux_vector(const Array<OneD, Array<OneD, NekDouble>> &physfield,
                       Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);

  Array<OneD, NekDouble> &get_norm_vel();
};

} // namespace NESO::Solvers::H3LAPD

#endif
