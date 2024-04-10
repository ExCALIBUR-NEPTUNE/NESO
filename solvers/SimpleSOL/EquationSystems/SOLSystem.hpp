#ifndef SOLSYSTEM_H
#define SOLSYSTEM_H

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

class SOLSystem : public SU::UnsteadySystem {
public:
  friend class MemoryManager<SOLSystem>;

  /// Creates an instance of this class.
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &pSession,
         const SD::MeshGraphSharedPtr &pGraph) {
    SU::EquationSystemSharedPtr p =
        MemoryManager<SOLSystem>::AllocateSharedPtr(pSession, pGraph);
    p->InitObject();
    return p;
  }

  /// Name of class.
  static std::string className;

  virtual ~SOLSystem();

  void GetDensity(const Array<OneD, const Array<OneD, NekDouble>> &physfield,
                  Array<OneD, NekDouble> &density);

  /// Function to get estimate of min h/p factor per element
  Array<OneD, NekDouble> GetElmtMinHP(void);

  void GetPressure(const Array<OneD, const Array<OneD, NekDouble>> &physfield,
                   Array<OneD, NekDouble> &pressure);

  void GetVelocity(const Array<OneD, const Array<OneD, NekDouble>> &physfield,
                   Array<OneD, Array<OneD, NekDouble>> &velocity);

  bool HasConstantDensity() { return false; }

protected:
  SOLSystem(const LU::SessionReaderSharedPtr &pSession,
            const SD::MeshGraphSharedPtr &pGraph);

  SU::AdvectionSharedPtr m_advObject;
  SU::DiffusionSharedPtr m_diffusion;
  NektarFieldIndexMap m_field_to_index;
  // Forcing term
  std::vector<SU::ForcingSharedPtr> m_forcing;
  NekDouble m_gamma;
  /// Names of fields that will be time integrated
  std::vector<std::string> m_int_fld_names;
  // List of field names required by the solver
  std::vector<std::string> m_required_flds;
  // Auxiliary object to convert variables
  VariableConverterSharedPtr m_varConv;
  Array<OneD, Array<OneD, NekDouble>> m_vecLocs;

  void DoAdvection(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                   Array<OneD, Array<OneD, NekDouble>> &outarray,
                   const NekDouble time,
                   const Array<OneD, const Array<OneD, NekDouble>> &pFwd,
                   const Array<OneD, const Array<OneD, NekDouble>> &pBwd);

  void DoOdeProjection(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                       Array<OneD, Array<OneD, NekDouble>> &outarray,
                       const NekDouble time);

  virtual void
  DoOdeRhs(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
           Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time);

  void GetElmtTimeStep(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                       Array<OneD, NekDouble> &tstep);

  void GetFluxVector(const Array<OneD, const Array<OneD, NekDouble>> &physfield,
                     TensorOfArray3D<NekDouble> &flux);

  NekDouble GetGamma() { return m_gamma; }

  const Array<OneD, const Array<OneD, NekDouble>> &GetNormals() {
    return m_traceNormals;
  }

  const Array<OneD, const Array<OneD, NekDouble>> &GetVecLocs() {
    return m_vecLocs;
  }

  void InitAdvection();

  virtual void v_InitObject(bool DeclareField) override;
  void ValidateFieldList();
};

} // namespace NESO::Solvers
#endif
