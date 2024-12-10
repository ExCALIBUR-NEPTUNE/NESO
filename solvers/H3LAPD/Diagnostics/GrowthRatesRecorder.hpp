#ifndef H3LAPD_GROWTH_RATES_RECORDER_H
#define H3LAPD_GROWTH_RATES_RECORDER_H

#include <fstream>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <neso_particles.hpp>

#include "../ParticleSystems/NeutralParticleSystem.hpp"
#include <LibUtilities/BasicUtils/ErrorUtil.hpp>

namespace LU = Nektar::LibUtilities;

namespace NESO::Solvers::H3LAPD {
/**
 * @brief Class to manage recording of energy and enstrophy growth rates
 * for Hasegawa-Wakatani-based equation systems.
 *
 */
template <typename T> class GrowthRatesRecorder {
protected:
  /// Pointer to number density field
  std::shared_ptr<T> n;
  /// Pointer to electric potential field
  std::shared_ptr<T> phi;
  /// Pointer to vorticity field
  std::shared_ptr<T> w;

  /// HW α constant
  double alpha;
  /// File handle for recording output
  std::ofstream fh;
  /// HW κ constant
  double kappa;
  /// Number of quad points associated with fields n, phi and w
  int npts;
  // Space dimension of the problem (not of the domain)
  const int prob_ndims;
  /// MPI rank
  int rank;
  /// Sets recording frequency (value of 0 disables recording)
  int recording_step;
  /// Pointer to session object
  const LU::SessionReaderSharedPtr session;

public:
  GrowthRatesRecorder(const LU::SessionReaderSharedPtr session, int prob_ndims,
                      std::shared_ptr<T> n, std::shared_ptr<T> w,
                      std::shared_ptr<T> phi, int npts, double alpha,
                      double kappa)
      : session(session), prob_ndims(prob_ndims), n(n), w(w), phi(phi),
        alpha(alpha), kappa(kappa), npts(npts) {

    // Store recording frequency for convenience
    this->session->LoadParameter("growth_rates_recording_step",
                                 this->recording_step, 0);

    // Store MPI rank for convenience
    this->rank = this->session->GetComm()->GetRank();

    // Fail for anything but prob_ndims=2 or 3
    NESOASSERT(this->prob_ndims == 2 || this->prob_ndims == 3,
               "GrowthRatesRecorder: invalid problem dimensionality; expected "
               "2 or 3.");

    // Write file header
    if ((this->rank == 0) && (this->recording_step > 0)) {
      this->fh.open("growth_rates.csv");
      this->fh << "step,E,W,dEdt_exp,dWdt_exp\n";
    }
  };

  ~GrowthRatesRecorder() {
    // Close file on destruct
    if ((this->rank == 0) && (this->recording_step > 0)) {
      this->fh.close();
    }
  }

  /**
   * Calculate Energy = 0.5 ∫ (n^2+|∇⊥ϕ|^2) dV
   */
  inline double compute_energy() {
    Array<OneD, NekDouble> integrand(this->npts);
    // First, set integrand = n^2
    Vmath::Vmul(this->npts, this->n->GetPhys(), 1, this->n->GetPhys(), 1,
                integrand, 1);

    // Compute ϕ derivs, square them and add to integrand
    Array<OneD, NekDouble> xderiv(this->npts), yderiv(this->npts),
        zderiv(this->npts);
    this->phi->PhysDeriv(this->phi->GetPhys(), xderiv, yderiv, zderiv);
    Vmath::Vvtvp(this->npts, xderiv, 1, xderiv, 1, integrand, 1, integrand, 1);
    Vmath::Vvtvp(this->npts, yderiv, 1, yderiv, 1, integrand, 1, integrand, 1);
    if (this->prob_ndims == 2) {
      /* Should be ∇⊥ϕ^2, so ∂ϕ/∂z ought to be excluded in both 2D and
       * 3D, but there's a small discrepancy in 2D without it. Energy 'leaking'
       * into orthogonal dimension?!? */
      Vmath::Vvtvp(this->npts, zderiv, 1, zderiv, 1, integrand, 1, integrand,
                   1);
    }

    return 0.5 * this->n->Integral(integrand);
  }

  /**
   * Calculate Enstrophy = 0.5 ∫ (n-w)^2 dV
   */
  inline double compute_enstrophy() {
    Array<OneD, NekDouble> integrand(this->npts);
    // Set integrand = n-w
    Vmath::Vsub(this->npts, this->n->GetPhys(), 1, this->w->GetPhys(), 1,
                integrand, 1);
    // Set integrand = (n-w)^2
    Vmath::Vmul(this->npts, integrand, 1, integrand, 1, integrand, 1);
    return 0.5 * this->n->Integral(integrand);
  }

  /**
   * Calculate Γα
   *  In 2D: Γα = α ∫ (n-ϕ)^2 dV​
   *  In 3D: Γα = α ∫ [∂/∂z(n-ϕ)]^2 dV
   */
  inline double compute_Gamma_a() {
    Array<OneD, NekDouble> integrand(this->npts);
    // Set integrand = n - ϕ
    Vmath::Vsub(this->npts, this->n->GetPhys(), 1, this->phi->GetPhys(), 1,
                integrand, 1);

    switch (this->prob_ndims) {
    case 2:
      // Set integrand = (n - ϕ)^2
      Vmath::Vmul(this->npts, integrand, 1, integrand, 1, integrand, 1);
      break;
    case 3:
      // Set integrand = ∂/∂z(n-ϕ)
      this->phi->PhysDeriv(2, integrand, integrand);
      // Set integrand = [∂/∂z(n-ϕ)]^2
      Vmath::Vmul(this->npts, integrand, 1, integrand, 1, integrand, 1);
      break;
    }
    return this->alpha * this->n->Integral(integrand);
  }

  /**
   * Calculate Γn = -κ ∫ n * ∂ϕ/∂y dV
   */
  inline double compute_Gamma_n() {
    Array<OneD, NekDouble> integrand(this->npts);

    // Set integrand = n * ∂ϕ/∂y
    this->phi->PhysDeriv(1, this->phi->GetPhys(), integrand);
    Vmath::Vmul(this->npts, integrand, 1, this->n->GetPhys(), 1, integrand, 1);
    return -this->kappa * this->n->Integral(integrand);
  }

  /**
   * Compute energy, enstrophy and gamma values and output to file
   */
  inline void compute(int step) {
    if (this->recording_step > 0) {
      if (step % this->recording_step == 0) {

        const double energy = compute_energy();
        const double enstrophy = compute_enstrophy();
        const double Gamma_n = compute_Gamma_n();
        const double Gamma_a = compute_Gamma_a();

        // Write values to file. In Debug, print to stdout too.
        if (this->rank == 0) {
          nprint(step, ",", energy, ",", enstrophy, ",", Gamma_n - Gamma_a, ",",
                 Gamma_n);
          this->fh << step << "," << std::setprecision(9) << energy << ","
                   << enstrophy << "," << Gamma_n - Gamma_a << "," << Gamma_n
                   << "\n";
        }
      }
    }
  }
};

} // namespace NESO::Solvers::H3LAPD

#endif // H3LAPD_GROWTH_RATES_RECORDER_H