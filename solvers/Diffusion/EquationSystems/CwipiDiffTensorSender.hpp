#ifndef DIFFUSION_CWIPISEND_H
#define DIFFUSION_CWIPISEND_H

#include <SolverUtils/Core/Coupling.h>

#include "DiffusionSystem.hpp"

namespace NESO::Solvers::Diffusion {
class CwipiDiffTensorSender : public DiffusionSystem {
public:
  friend class MemoryManager<CwipiDiffTensorSender>;

  /// Creates an instance of this class
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &session,
         const SD::MeshGraphSharedPtr &graph) {
    SU::EquationSystemSharedPtr p =
        MemoryManager<CwipiDiffTensorSender>::AllocateSharedPtr(session, graph);
    p->InitObject();
    return p;
  }
  /// Name of class
  static std::string className;

  /// Destructor
  virtual ~CwipiDiffTensorSender();

protected:
  CwipiDiffTensorSender(const LU::SessionReaderSharedPtr &session,
                        const SD::MeshGraphSharedPtr &graph);

  virtual void v_GenerateSummary(SU::SummaryList &s) override final;
  virtual void v_InitObject(bool DeclareField = true) override final;
  virtual bool v_PreIntegrate(int step) override final;

private:
  SU::CouplingSharedPtr coupling;

  void do_nothing(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                  Array<OneD, Array<OneD, NekDouble>> &out_arr,
                  const NekDouble time, const NekDouble lambda);
};
} // namespace NESO::Solvers::Diffusion

#endif