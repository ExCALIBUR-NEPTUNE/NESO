#ifndef __H3LAPD_GROWTH_RATES_H_
#define __H3LAPD_GROWTH_RATES_H_

#include <memory>
#include <mpi.h>
#include <neso_particles.hpp>
using namespace NESO;
using namespace NESO::Particles;

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
using namespace Nektar;

#include "../ParticleSystems/NeutralParticleSystem.hpp"

#include <fstream>
#include <iostream>

template <typename T> class GrowthRatesRecorder {
protected:
  const LibUtilities::SessionReaderSharedPtr session;
  std::shared_ptr<NeutralParticleSystem> particle_sys;
  std::shared_ptr<T> n;
  std::shared_ptr<T> w;
  std::shared_ptr<T> phi;
  int nPts;
  double alpha;
  double kappa;
  SYCLTargetSharedPtr sycl_target;
  int growth_rates_recording_step;
  int rank;
  ofstream fh;

public:
  GrowthRatesRecorder(const LibUtilities::SessionReaderSharedPtr session,
                      std::shared_ptr<NeutralParticleSystem> particle_sys,
                      std::shared_ptr<T> n, std::shared_ptr<T> w,
                      std::shared_ptr<T> phi, int nPts, double alpha,
                      double kappa)
      : session(session), particle_sys(particle_sys), n(n), w(w), phi(phi),
        alpha(alpha), kappa(kappa), nPts(nPts),
        sycl_target(particle_sys->sycl_target) {

    session->LoadParameter("growth_rates_recording_step",
                           growth_rates_recording_step, 0);
    rank = sycl_target->comm_pair.rank_parent;
    if ((rank == 0) && (growth_rates_recording_step > 0)) {
      fh.open("growth_rates.csv");
      fh << "step,E,W,dEdt_exp,dWdt_exp\n";
    }
  };

  ~GrowthRatesRecorder() {
    if ((rank == 0) && (growth_rates_recording_step > 0)) {
      fh.close();
    }
  }

  /**
   * Energy = 0.5 ∫ (n^2+|∇ϕ|^2) dV
   */
  inline double compute_energy() {
    Array<OneD, NekDouble> integrand(this->nPts);
    // First, set integrand=n^2
    Vmath::Vmul(this->nPts, this->n->GetPhys(), 1, this->n->GetPhys(), 1,
                integrand, 1);

    // Compute phi derivs, square them and add to integrand
    Array<OneD, NekDouble> xderiv(nPts), yderiv(nPts), zderiv(nPts);
    this->phi->PhysDeriv(this->phi->GetPhys(), xderiv, yderiv, zderiv);
    Vmath::Vvtvp(nPts, xderiv, 1, xderiv, 1, integrand, 1, integrand, 1);
    Vmath::Vvtvp(nPts, yderiv, 1, yderiv, 1, integrand, 1, integrand, 1);
    Vmath::Vvtvp(nPts, zderiv, 1, zderiv, 1, integrand, 1, integrand, 1);

    // integrand *= 0.5
    Vmath::Smul(nPts, 0.5, integrand, 1, integrand, 1);
    return this->n->Integral(integrand);
  }

  /**
   * Enstrophy = 0.5 ∫ (n-w)^2 dV
   */
  inline double compute_enstrophy() {
    Array<OneD, NekDouble> integrand(this->nPts);
    // n-w
    Vmath::Vsub(this->nPts, this->n->GetPhys(), 1, this->w->GetPhys(), 1,
                integrand, 1);
    // (n-w)^2
    Vmath::Vmul(this->nPts, integrand, 1, integrand, 1, integrand, 1);
    // 0.5*(n-w)^2
    Vmath::Smul(nPts, 0.5, integrand, 1, integrand, 1);
    return this->n->Integral(integrand);
  }

  /**
   * Gamma_alpha = alpha ∫ (n-phi)^2 dV
   */
  inline double compute_Gamma_a() {
    Array<OneD, NekDouble> integrand(this->nPts);
    // n-phi
    Vmath::Vsub(this->nPts, this->n->GetPhys(), 1, this->phi->GetPhys(), 1,
                integrand, 1);
    // (n-phi)^2
    Vmath::Vmul(this->nPts, integrand, 1, integrand, 1, integrand, 1);
    // alpha*(n-phi)^2
    Vmath::Smul(nPts, this->alpha, integrand, 1, integrand, 1);
    return this->n->Integral(integrand);
  }

  /**
   * Gamma_n = -kappa ∫ n * dphi/dy dV
   */
  inline double compute_Gamma_n() {
    Array<OneD, NekDouble> integrand(this->nPts);

    // n * dphi/dy
    this->phi->PhysDeriv(1, this->phi->GetPhys(), integrand);
    Vmath::Vmul(this->nPts, integrand, 1, this->n->GetPhys(), 1, integrand, 1);

    // -kappa * n * dphi/dy
    Vmath::Smul(nPts, -1 * this->kappa, integrand, 1, integrand, 1);
    return this->n->Integral(integrand);
  }

  inline void compute(int step) {
    if (growth_rates_recording_step > 0) {
      if (step % growth_rates_recording_step == 0) {

        const double energy = compute_energy();
        const double enstrophy = compute_enstrophy();
        const double Gamma_n = compute_Gamma_n();
        const double Gamma_a = compute_Gamma_a();

        // Write values to file
        if (rank == 0) {
          nprint(step, ",", energy, ",", enstrophy, ",", Gamma_n - Gamma_a, ",",
                 Gamma_n);
          fh << step << "," << std::setprecision(9) << energy << ","
             << enstrophy << "," << Gamma_n - Gamma_a << "," << Gamma_n << "\n";
        }
      }
    }
  };
};

#endif