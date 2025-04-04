#ifndef __NESOSOLVERS_SIMPLESOL_SOLSYSTEM_HPP__
#define __NESOSOLVERS_SIMPLESOL_SOLSYSTEM_HPP__

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
#include <nektar_interface/solver_base/time_evolved_eqnsys_base.hpp>
#include <nektar_interface/utilities.hpp>

#include "../ParticleSystems/NeutralParticleSystem.hpp"

namespace LU = Nektar::LibUtilities;
namespace MR = Nektar::MultiRegions;
namespace SD = Nektar::SpatialDomains;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::SimpleSOL {

class SOLSystem
    : public TimeEvoEqnSysBase<SU::UnsteadySystem, NeutralParticleSystem> {
public:
  friend class Nektar::MemoryManager<SOLSystem>;

  /// Creates an instance of this class.
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &session,
         const SD::MeshGraphSharedPtr &graph) {
    SU::EquationSystemSharedPtr equation_sys =
        Nektar::MemoryManager<SOLSystem>::AllocateSharedPtr(session, graph);
    equation_sys->InitObject();
    return equation_sys;
  }

  /// Name of class.
  static std::string class_name;

protected:
  SOLSystem(const LU::SessionReaderSharedPtr &session,
            const SD::MeshGraphSharedPtr &graph);

  /// Advection object
  SU::AdvectionSharedPtr adv_obj;
  /// Fluid forcing / source terms
  std::vector<SU::ForcingSharedPtr> fluid_src_terms;
  /// Gamma value, read from config
  NekDouble gamma;
  /// Auxiliary object to convert variables
  VariableConverterSharedPtr var_converter;
  Array<OneD, Array<OneD, NekDouble>> vel_fld_indices;

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

  NekDouble get_gamma() { return this->gamma; }

  const Array<OneD, const Array<OneD, NekDouble>> &get_trace_norms() {
    return m_traceNormals;
  }

  /**
   * Tells the Riemann solver the location of any "auxiliary" vectors
   * (velocity field indices, in this case)
   */
  const Array<OneD, const Array<OneD, NekDouble>> &get_vec_locs() {
    return this->vel_fld_indices;
  }

  void init_advection();

  virtual void v_InitObject(bool DeclareField) override;
};

} // namespace NESO::Solvers::SimpleSOL
#endif // __NESOSOLVERS_SIMPLESOL_SOLSYSTEM_HPP__
