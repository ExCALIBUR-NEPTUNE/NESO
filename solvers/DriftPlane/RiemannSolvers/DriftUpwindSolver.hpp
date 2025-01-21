#ifndef DRIFTPLANE_DRIFTUPWINDSOLVER_H
#define DRIFTPLANE_DRIFTUPWINDSOLVER_H

#include <SolverUtils/RiemannSolvers/RiemannSolver.h>

namespace LU = Nektar::LibUtilities;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::DriftPlane {

class DriftUpwindSolver : public SU::RiemannSolver {
public:
  DriftUpwindSolver(const LU::SessionReaderSharedPtr &pSession, NekDouble c)
      : m_c(c) {}

protected:
  NekDouble m_c;

  virtual void
  v_Solve(const int nDim, const Array<OneD, const Array<OneD, NekDouble>> &Fwd,
          const Array<OneD, const Array<OneD, NekDouble>> &Bwd,
          Array<OneD, Array<OneD, NekDouble>> &flux) override final {
    boost::ignore_unused(nDim);

    ASSERTL1(CheckScalars("Vn"), "Vn not defined.");
    const Array<OneD, NekDouble> &traceVel = m_scalars["Vn"]();
    const Array<OneD, NekDouble> &ny = m_scalars["ny"]();

    for (int j = 0; j < traceVel.size(); ++j) {
      const Array<OneD, const Array<OneD, NekDouble>> &tmp =
          traceVel[j] >= 0 ? Fwd : Bwd;
      for (int i = 0; i < Fwd.size(); ++i) {
        flux[i][j] = traceVel[j] * tmp[i][j];
      }

      // subtract dn/dy term - dot with ny component
      flux[1][j] -= m_c * ny[j] * tmp[0][j];
    }
  }
};
}
#endif