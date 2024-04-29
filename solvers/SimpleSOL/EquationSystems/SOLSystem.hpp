#ifndef __SIMPLESOL_SOLSYSTEM_H_
#define __SIMPLESOL_SOLSYSTEM_H_

#include "../ParticleSystems/neutral_particles.hpp"
#include "nektar_interface/solver_base/time_evolved_eqnsys_base.hpp"
#include "nektar_interface/utilities.hpp"
#include <CompressibleFlowSolver/Misc/VariableConverter.h>
#include <LocalRegions/Expansion2D.h>
#include <LocalRegions/Expansion3D.h>
#include <MultiRegions/GlobalMatrixKey.h>
#include <SolverUtils/Advection/Advection.h>
#include <SolverUtils/Diffusion/Diffusion.h>
#include <SolverUtils/Filters/FilterInterfaces.hpp>
#include <SolverUtils/Forcing/Forcing.h>
#include <SolverUtils/RiemannSolvers/RiemannSolver.h>
#include <SolverUtils/UnsteadySystem.h>
#include <boost/core/ignore_unused.hpp>

namespace LU = Nektar::LibUtilities;
namespace MR = Nektar::MultiRegions;
namespace SD = Nektar::SpatialDomains;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers {

class SOLSystem
    : public TimeEvoEqnSysBase<SU::UnsteadySystem, NeutralParticleSystem> {
public:
  friend class MemoryManager<SOLSystem>;

  /// Creates an instance of this class.
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &session,
         const SD::MeshGraphSharedPtr &graph) {
    SU::EquationSystemSharedPtr equation_sys =
        MemoryManager<SOLSystem>::AllocateSharedPtr(session, graph);
    equation_sys->InitObject();
    return equation_sys;
  }

  /// Name of class.
  static std::string class_name;

  virtual ~SOLSystem();

protected:
  SOLSystem(const LU::SessionReaderSharedPtr &session,
            const SD::MeshGraphSharedPtr &graph);

  SU::AdvectionSharedPtr m_adv;
  // Forcing terms
  std::vector<SU::ForcingSharedPtr> m_forcing;
  NekDouble m_gamma;
  // Auxiliary object to convert variables
  VariableConverterSharedPtr m_var_converter;
  Array<OneD, Array<OneD, NekDouble>> m_vec_locs;

  void do_advection(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                    Array<OneD, Array<OneD, NekDouble>> &out_arr,
                    const NekDouble time,
                    const Array<OneD, const Array<OneD, NekDouble>> &fwd,
                    const Array<OneD, const Array<OneD, NekDouble>> &bwd);

  void
  do_ode_projection(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                    Array<OneD, Array<OneD, NekDouble>> &out_arr,
                    const NekDouble time);

  virtual void
  explicit_time_int(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                    Array<OneD, Array<OneD, NekDouble>> &out_arr,
                    const NekDouble time);

  void
  get_flux_vector(const Array<OneD, const Array<OneD, NekDouble>> &physfield,
                  TensorOfArray3D<NekDouble> &flux);

  NekDouble get_gamma() { return m_gamma; }

  const Array<OneD, const Array<OneD, NekDouble>> &get_trace_norms() {
    return m_traceNormals;
  }

  /**
   * Tells the Riemann solver the location of any "auxiliary" vectors
   * (velocity field indices, in this case)
   */
  const Array<OneD, const Array<OneD, NekDouble>> &get_vec_locs() {
    return m_vec_locs;
  }

  void init_advection();

  virtual void v_InitObject(bool DeclareField) override;
};

} // namespace NESO::Solvers
#endif
