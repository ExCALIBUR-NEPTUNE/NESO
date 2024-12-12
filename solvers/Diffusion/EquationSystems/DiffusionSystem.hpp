#ifndef DIFFUSION_DIFFUSIONSYSTEM_H
#define DIFFUSION_DIFFUSIONSYSTEM_H

#include <SolverUtils/Diffusion/Diffusion.h>
#include <SolverUtils/UnsteadySystem.h>

using namespace Nektar::SolverUtils;

namespace Nektar {
class UnsteadyDiffusion : public UnsteadySystem {
public:
  friend class MemoryManager<UnsteadyDiffusion>;

  /// Creates an instance of this class
  static EquationSystemSharedPtr
  create(const LibUtilities::SessionReaderSharedPtr &pSession,
         const SpatialDomains::MeshGraphSharedPtr &pGraph) {
    EquationSystemSharedPtr p =
        MemoryManager<UnsteadyDiffusion>::AllocateSharedPtr(pSession, pGraph);
    p->InitObject();
    return p;
  }
  /// Name of class
  static std::string className;

  /// Destructor
  virtual ~UnsteadyDiffusion();

protected:
  bool m_useSpecVanVisc;
  NekDouble
      m_sVVCutoffRatio;     // Cut-off ratio from which to start decaying modes
  NekDouble m_sVVDiffCoeff; // Diffusion coefficient of SVV modes
  SolverUtils::DiffusionSharedPtr m_diffusion;
  SolverUtils::RiemannSolverSharedPtr m_riemannSolver;

  UnsteadyDiffusion(const LibUtilities::SessionReaderSharedPtr &pSession,
                    const SpatialDomains::MeshGraphSharedPtr &pGraph);

  virtual void v_InitObject(bool DeclareField = true) override;
  virtual void v_GenerateSummary(SummaryList &s) override;

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
  StdRegions::VarCoeffMap m_varcoeff;
  StdRegions::ConstFactorMap m_factors;
};
} // namespace Nektar

#endif
