#ifndef DIFFUSION_CWIPIRECEIVE_H
#define DIFFUSION_CWIPIRECEIVE_H

#include <SolverUtils/Core/Coupling.h>

#include "DiffusionSystem.hpp"

namespace NESO::Solvers::Diffusion {
class CwipiReceiveDiffTensorAndDiffuse : public DiffusionSystem {
public:
  friend class MemoryManager<CwipiReceiveDiffTensorAndDiffuse>;

  /// Creates an instance of this class
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &session,
         const SD::MeshGraphSharedPtr &graph) {
    SU::EquationSystemSharedPtr p =
        MemoryManager<CwipiReceiveDiffTensorAndDiffuse>::AllocateSharedPtr(
            session, graph);
    p->InitObject();
    return p;
  }
  /// Name of class
  static std::string className;

  /// Destructor
  virtual ~CwipiReceiveDiffTensorAndDiffuse();

protected:
  CwipiReceiveDiffTensorAndDiffuse(const LU::SessionReaderSharedPtr &session,
                                   const SD::MeshGraphSharedPtr &graph);

  virtual void v_GenerateSummary(SU::SummaryList &s) override final;
  virtual void v_InitObject(bool DeclareField = true) override final;
  virtual bool v_PreIntegrate(int step) override final;

private:
  SU::CouplingSharedPtr coupling;
};
} // namespace NESO::Solvers::Diffusion

#endif