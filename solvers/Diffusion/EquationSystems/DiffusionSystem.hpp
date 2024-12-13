#ifndef DIFFUSION_DIFFUSIONSYSTEM_H
#define DIFFUSION_DIFFUSIONSYSTEM_H

#include <SolverUtils/Diffusion/Diffusion.h>
#include <SolverUtils/UnsteadySystem.h>

#include "nektar_interface/solver_base/time_evolved_eqnsys_base.hpp"

namespace LU = Nektar::LibUtilities;
namespace SD = Nektar::SpatialDomains;
namespace SR = Nektar::StdRegions;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::Diffusion {
class DiffusionSystem : public SU::UnsteadySystem {
public:
  friend class MemoryManager<DiffusionSystem>;

  /// Creates an instance of this class
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &pSession,
         const SD::MeshGraphSharedPtr &pGraph) {
    SU::EquationSystemSharedPtr p =
        MemoryManager<DiffusionSystem>::AllocateSharedPtr(pSession, pGraph);
    p->InitObject();
    return p;
  }
  /// Name of class
  static std::string className;

  /// Destructor
  virtual ~DiffusionSystem();

protected:
  bool m_useSpecVanVisc;
  NekDouble
      m_sVVCutoffRatio;     // Cut-off ratio from which to start decaying modes
  NekDouble m_sVVDiffCoeff; // Diffusion coefficient of SVV modes
  SU::DiffusionSharedPtr m_diffusion;
  SU::RiemannSolverSharedPtr m_riemannSolver;

  DiffusionSystem(const LU::SessionReaderSharedPtr &pSession,
                  const SD::MeshGraphSharedPtr &pGraph);

  virtual void v_InitObject(bool DeclareField = true) override;
  virtual void v_GenerateSummary(SU::SummaryList &s) override;

  void DoOdeProjection(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                       Array<OneD, Array<OneD, NekDouble>> &outarray,
                       const NekDouble time);
  void DoImplicitSolve(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                       Array<OneD, Array<OneD, NekDouble>> &outarray,
                       NekDouble time, NekDouble lambda);

private:
  NekDouble m_kperp;
  NekDouble m_kpar;
  NekDouble m_theta;
  NekDouble m_n;
  NekDouble m_epsilon;
  SR::VarCoeffMap m_varcoeff;
  SR::ConstFactorMap m_factors;
};
} // namespace NESO::Solvers::Diffusion

#endif
