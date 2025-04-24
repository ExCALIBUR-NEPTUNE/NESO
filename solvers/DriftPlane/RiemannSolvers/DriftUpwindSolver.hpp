#ifndef __NESOSOLVERS_DRIFTPLANE_DRIFTUPWINDSOLVER_HPP__
#define __NESOSOLVERS_DRIFTPLANE_DRIFTUPWINDSOLVER_HPP__

#include <SolverUtils/RiemannSolvers/RiemannSolver.h>

namespace NESO::Solvers::DriftPlane {

/**
 * @brief Custom RiemannSolver that does upwinding that accounts for dn/dy term.
 *
 */
class DriftUpwindSolver : public Nektar::SolverUtils::RiemannSolver {
public:
  DriftUpwindSolver(const Nektar::LibUtilities::SessionReaderSharedPtr &session,
                    NekDouble c, int n_idx, int w_idx)
      : c(c), n_idx(n_idx), w_idx(w_idx) {}

protected:
  /**
   * @brief Calculate up-winded fluxes.
   *
   * @param[in] Fwd Forward trace values
   * @param[in] Bwd Backward trace values
   * @param[out] flux Calculated fluxes
   */
  virtual void
  v_Solve(const int nDim, const Array<OneD, const Array<OneD, NekDouble>> &Fwd,
          const Array<OneD, const Array<OneD, NekDouble>> &Bwd,
          Array<OneD, Array<OneD, NekDouble>> &flux) override final {
    boost::ignore_unused(nDim);

    ASSERTL1(CheckScalars("Vn"), "Vn not defined.");
    const Array<OneD, NekDouble> &traceVel = m_scalars["Vn"]();
    const Array<OneD, NekDouble> &ny = m_scalars["ny"]();

    for (int ipt = 0; ipt < traceVel.size(); ++ipt) {
      const Array<OneD, const Array<OneD, NekDouble>> &tmp =
          traceVel[ipt] >= 0 ? Fwd : Bwd;
      // Standard upwind flux for all fields
      for (int ifld = 0; ifld < Fwd.size(); ++ifld) {
        flux[ifld][ipt] = traceVel[ipt] * tmp[ifld][ipt];
      }

      // For w flux, subtract dn/dy term - dot with ny component
      flux[this->w_idx][ipt] -= this->c * ny[ipt] * tmp[this->n_idx][ipt];
    }
  }

private:
  /// Constant used in flux calc
  NekDouble c;
  /// Field indices
  int n_idx;
  int w_idx;
};
} // namespace NESO::Solvers::DriftPlane
#endif // __NESOSOLVERS_DRIFTPLANE_DRIFTUPWINDSOLVER_HPP__