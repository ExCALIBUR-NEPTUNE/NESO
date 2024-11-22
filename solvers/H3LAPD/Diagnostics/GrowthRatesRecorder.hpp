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
  std::shared_ptr<T> m_n;
  /// Pointer to electric potential field
  std::shared_ptr<T> m_phi;
  /// Pointer to vorticity field
  std::shared_ptr<T> m_w;

  /// HW α constant
  double m_alpha;
  /// File handle for recording output
  std::ofstream m_fh;
  /// HW κ constant
  double m_kappa;
  /// Number of quad points associated with fields m_n, m_phi and m_w
  int m_npts;
  /// MPI rank
  int m_rank;
  /// Sets recording frequency (value of 0 disables recording)
  int m_recording_step;
  /// Pointer to session object
  const LU::SessionReaderSharedPtr m_session;

public:
  GrowthRatesRecorder(const LU::SessionReaderSharedPtr session,
                      std::shared_ptr<T> n, std::shared_ptr<T> w,
                      std::shared_ptr<T> phi, int npts, double alpha,
                      double kappa)
      : m_session(session), m_n(n), m_w(w), m_phi(phi), m_alpha(alpha),
        m_kappa(kappa), m_npts(npts) {

    // Store recording frequency for convenience
    m_session->LoadParameter("growth_rates_recording_step", m_recording_step,
                             0);

    // Store MPI rank for convenience
    m_rank = m_session->GetComm()->GetRank();

    // Write file header
    if ((m_rank == 0) && (m_recording_step > 0)) {
      m_fh.open("growth_rates.csv");
      m_fh << "step,E,W,dEdt_exp,dWdt_exp\n";
    }
  };

  ~GrowthRatesRecorder() {
    // Close file on destruct
    if ((m_rank == 0) && (m_recording_step > 0)) {
      m_fh.close();
    }
  }

  /**
   * Calculate Energy = 0.5 ∫ (n^2+|∇ϕ|^2) dV
   */
  inline double compute_energy() {
    Array<OneD, NekDouble> integrand(m_npts);
    // First, set integrand = n^2
    Vmath::Vmul(m_npts, m_n->GetPhys(), 1, m_n->GetPhys(), 1, integrand, 1);

    // Compute phi derivs, square them and add to integrand
    Array<OneD, NekDouble> xderiv(m_npts), yderiv(m_npts), zderiv(m_npts);
    m_phi->PhysDeriv(m_phi->GetPhys(), xderiv, yderiv, zderiv);
    Vmath::Vvtvp(m_npts, xderiv, 1, xderiv, 1, integrand, 1, integrand, 1);
    Vmath::Vvtvp(m_npts, yderiv, 1, yderiv, 1, integrand, 1, integrand, 1);
    Vmath::Vvtvp(m_npts, zderiv, 1, zderiv, 1, integrand, 1, integrand, 1);

    // integrand *= 0.5
    Vmath::Smul(m_npts, 0.5, integrand, 1, integrand, 1);
    return m_n->Integral(integrand);
  }

  /**
   * Calculate Enstrophy = 0.5 ∫ (n-w)^2 dV
   */
  inline double compute_enstrophy() {
    Array<OneD, NekDouble> integrand(m_npts);
    // Set integrand = n-w
    Vmath::Vsub(m_npts, m_n->GetPhys(), 1, m_w->GetPhys(), 1, integrand, 1);
    // Set integrand = (n-w)^2
    Vmath::Vmul(m_npts, integrand, 1, integrand, 1, integrand, 1);
    // Set integrand = 0.5*(n-w)^2
    Vmath::Smul(m_npts, 0.5, integrand, 1, integrand, 1);
    return m_n->Integral(integrand);
  }

  /**
   * Calculate Gamma_alpha = alpha ∫ (n-phi)^2 dV
   */
  inline double compute_Gamma_a() {
    Array<OneD, NekDouble> integrand(m_npts);
    // Set integrand = n - phi
    Vmath::Vsub(m_npts, m_n->GetPhys(), 1, m_phi->GetPhys(), 1, integrand, 1);
    // Set integrand = (n - phi)^2
    Vmath::Vmul(m_npts, integrand, 1, integrand, 1, integrand, 1);
    // Set integrand = alpha*(n - phi)^2
    Vmath::Smul(m_npts, m_alpha, integrand, 1, integrand, 1);
    return m_n->Integral(integrand);
  }

  /**
   * Calculate Gamma_n = -kappa ∫ n * dphi/dy dV
   */
  inline double compute_Gamma_n() {
    Array<OneD, NekDouble> integrand(m_npts);

    // Set integrand = n * dphi/dy
    m_phi->PhysDeriv(1, m_phi->GetPhys(), integrand);
    Vmath::Vmul(m_npts, integrand, 1, m_n->GetPhys(), 1, integrand, 1);

    // Set integrand = -kappa * n * dphi/dy
    Vmath::Smul(m_npts, -1 * m_kappa, integrand, 1, integrand, 1);
    return m_n->Integral(integrand);
  }

  /**
   * Compute energy, enstrophy and gamma values and output to file
   */
  inline void compute(int step) {
    if (m_recording_step > 0) {
      if (step % m_recording_step == 0) {

        const double energy = compute_energy();
        const double enstrophy = compute_enstrophy();
        const double Gamma_n = compute_Gamma_n();
        const double Gamma_a = compute_Gamma_a();

        // Write values to file. In Debug, print to stdout too.
        if (m_rank == 0) {
          nprint(step, ",", energy, ",", enstrophy, ",", Gamma_n - Gamma_a, ",",
                 Gamma_n);
          m_fh << step << "," << std::setprecision(9) << energy << ","
               << enstrophy << "," << Gamma_n - Gamma_a << "," << Gamma_n
               << "\n";
        }
      }
    }
  }
};

} // namespace NESO::Solvers::H3LAPD

#endif // H3LAPD_GROWTH_RATES_RECORDER_H