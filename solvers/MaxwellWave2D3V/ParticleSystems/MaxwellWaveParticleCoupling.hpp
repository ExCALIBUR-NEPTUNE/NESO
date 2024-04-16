#ifndef __MAXWELL_WAVE_PARTICLE_COUPLING_H_
#define __MAXWELL_WAVE_PARTICLE_COUPLING_H_

#include <memory>
#include <nektar_interface/function_evaluation.hpp>
#include <nektar_interface/function_projection.hpp>
#include <nektar_interface/particle_interface.hpp>
#include <nektar_interface/utilities.hpp>
#include <neso_particles.hpp>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Core/SessionFunction.h>
#include <SolverUtils/EquationSystem.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mpi.h>
#include <random>
#include <string>

#include "../EquationSystems/MaxwellWaveSystem.h"
#include "ChargedParticles.hpp"

using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace NESO::Particles;

template <typename T> class MaxwellWaveParticleCoupling {
private:
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  std::shared_ptr<ChargedParticles> charged_particles;

  std::shared_ptr<FieldProject<T>> rho_field_project;
  std::vector<std::shared_ptr<FieldProject<T>>> jx_field_projects;
  std::vector<std::shared_ptr<FieldProject<T>>> jy_field_projects;
  std::vector<std::shared_ptr<FieldProject<T>>> jz_field_projects;
  std::vector<std::shared_ptr<FieldEvaluate<T>>> phi_field_evaluates;
  std::vector<std::shared_ptr<FieldEvaluate<T>>> ax_field_evaluates;
  std::vector<std::shared_ptr<FieldEvaluate<T>>> ay_field_evaluates;
  std::vector<std::shared_ptr<FieldEvaluate<T>>> az_field_evaluates;
  std::vector<std::shared_ptr<FieldEvaluate<T>>> bx_field_evaluates;
  std::vector<std::shared_ptr<FieldEvaluate<T>>> by_field_evaluates;
  std::vector<std::shared_ptr<FieldEvaluate<T>>> bz_field_evaluates;
  std::vector<std::shared_ptr<FieldEvaluate<T>>> ex_field_evaluates;
  std::vector<std::shared_ptr<FieldEvaluate<T>>> ey_field_evaluates;
  std::vector<std::shared_ptr<FieldEvaluate<T>>> ez_field_evaluates;
  std::vector<std::shared_ptr<FieldEvaluate<T>>> gradax_field_evaluates;
  std::vector<std::shared_ptr<FieldEvaluate<T>>> graday_field_evaluates;
  std::vector<std::shared_ptr<FieldEvaluate<T>>> gradaz_field_evaluates;

  std::shared_ptr<MaxwellWaveSystem> m_maxwellWaveSys;
  double m_volume;

  // inline void add_neutralising_field() {
  //   // get the charge integral
  //   const double total_charge = this->rho_field->Integral();
  //   NESOASSERT(std::isfinite(total_charge),
  //              "Total charge is not finite (e.g. NaN or Inf/-Inf).");

  //  // get the jx current integral
  //  const double total_jx = this->jx_field->Integral();
  //  NESOASSERT(std::isfinite(total_jx),
  //             "Total jx is not finite (e.g. NaN or Inf/-Inf).");

  //  // get the jy current integral
  //  const double total_jy = this->jy_field->Integral();
  //  NESOASSERT(std::isfinite(total_jy),
  //             "Total jy is not finite (e.g. NaN or Inf/-Inf).");

  //  // get the jz current integral
  //  const double total_jz = this->jz_field->Integral();
  //  NESOASSERT(std::isfinite(total_jz),
  //             "Total jz is not finite (e.g. NaN or Inf/-Inf).");

  //  //// Modifiable reference to coeffs
  //  //auto coeffs = this->rho_field->UpdateCoeffs();
  //  //const int num_coeffs_rho = this->rho_field->GetNcoeffs();

  //  //for (int cx = 0; cx < num_coeffs_rho; cx++) {
  //  //  NESOASSERT(std::isfinite(coeffs[cx]),
  //  //             "A forcing coefficient is not finite (e.g. NaN or
  //  Inf/-Inf).");
  //  //  coeffs[cx] += this->ncd_coeff_values[cx] * average_charge_density;
  //  //}

  //  // Modifiable reference to phys values
  //  //auto phys_values = this->rho_field->UpdatePhys();
  //  //const int num_phys_rho = this->rho_field->GetTotPoints();
  //  //for (int cx = 0; cx < num_phys_rho; cx++) {
  //  //  NESOASSERT(std::isfinite(phys_values[cx]),
  //  //             "A phys value is not finite (e.g. NaN or Inf/-Inf).");
  //  //  phys_values[cx] += this->ncd_phys_values[cx] * average_charge_density;
  //  //}

  //  // integral should be approximately 0
  //  auto integral_ref_weight = this->charged_particles->particle_weight;
  //  const auto integral_forcing_func = this->rho_field->Integral() *
  //                                     integral_ref_weight;

  //  std::string error_msg =
  //      "RHS is not neutral, log10 error: " +
  //      std::to_string(std::log10(ABS(integral_forcing_func)));
  //  NESOASSERT(ABS(integral_forcing_func) < 1.0e-6, error_msg.c_str());
  //}

  inline void solve_equation_system(const double theta,
                                    const double dtMultiplier,
                                    const bool chargeConservationSwitch) {
    //    auto phys_rho = this->rho_field->UpdatePhys();
    //    auto coeffs_rho = this->rho_field->UpdateCoeffs();
    //    const double scaling_factor =
    //    -this->charged_particles->particle_weight; for (int cx = 0; cx <
    //    tot_points_rho; cx++) {
    //      phys_rho[cx] *= scaling_factor;
    //    }
    //    for (int cx = 0; cx < num_coeffs_rho; cx++) {
    //      coeffs_rho[cx] = scaling_factor;
    //    }

    this->m_maxwellWaveSys->setDtMultiplier(dtMultiplier);
    this->m_maxwellWaveSys->setTheta(theta);
    this->m_maxwellWaveSys->ChargeConservationSwitch(chargeConservationSwitch);

    this->m_maxwellWaveSys->DoSolve();
  }

public:
  /// The RHS of the maxwell_wave equation.
  std::shared_ptr<T> rho_field;
  std::shared_ptr<T> rho_minus_field;
  std::shared_ptr<T> jx_field;
  std::shared_ptr<T> jy_field;
  std::shared_ptr<T> jz_field;
  /// The solution function of the maxwell_wave equation.
  std::shared_ptr<T> phi_field;
  std::shared_ptr<T> phi_minus_field;
  std::shared_ptr<T> ax_field;
  std::shared_ptr<T> ay_field;
  std::shared_ptr<T> az_field;
  std::shared_ptr<T> ax_minus_field;
  std::shared_ptr<T> ay_minus_field;
  std::shared_ptr<T> az_minus_field;
  std::shared_ptr<T> bx_field;
  std::shared_ptr<T> by_field;
  std::shared_ptr<T> bz_field;
  std::shared_ptr<T> ex_field;
  std::shared_ptr<T> ey_field;
  std::shared_ptr<T> ez_field;

  MaxwellWaveParticleCoupling(
      LibUtilities::SessionReaderSharedPtr session,
      SpatialDomains::MeshGraphSharedPtr graph,
      std::shared_ptr<ChargedParticles> charged_particles)
      : session(session), graph(graph),
        charged_particles(charged_particles) {

    std::string eqnType = session->GetSolverInfo("EqType");
    EquationSystemSharedPtr eqnSystem = GetEquationSystemFactory().CreateInstance(
                    eqnType, session, graph);
    this->m_maxwellWaveSys = std::dynamic_pointer_cast<MaxwellWaveSystem>(
        eqnSystem);
    auto fields = this->m_maxwellWaveSys->UpdateFields();
    const int phi_index = this->m_maxwellWaveSys->GetFieldIndex("phi");
    const int phi_minus_index = this->m_maxwellWaveSys->GetFieldIndex("phi_minus");
    const int rho_index = this->m_maxwellWaveSys->GetFieldIndex("rho");
    const int rho_minus_index = this->m_maxwellWaveSys->GetFieldIndex("rho_minus");
    const int ax_index = this->m_maxwellWaveSys->GetFieldIndex("Ax");
    const int ay_index = this->m_maxwellWaveSys->GetFieldIndex("Ay");
    const int az_index = this->m_maxwellWaveSys->GetFieldIndex("Az");
    const int ax_minus_index = this->m_maxwellWaveSys->GetFieldIndex("Ax_minus");
    const int ay_minus_index = this->m_maxwellWaveSys->GetFieldIndex("Ay_minus");
    const int az_minus_index = this->m_maxwellWaveSys->GetFieldIndex("Az_minus");
    const int bx_index = this->m_maxwellWaveSys->GetFieldIndex("Bx");
    const int by_index = this->m_maxwellWaveSys->GetFieldIndex("By");
    const int bz_index = this->m_maxwellWaveSys->GetFieldIndex("Bz");
    const int ex_index = this->m_maxwellWaveSys->GetFieldIndex("Ex");
    const int ey_index = this->m_maxwellWaveSys->GetFieldIndex("Ey");
    const int ez_index = this->m_maxwellWaveSys->GetFieldIndex("Ez");
    const int jx_index = this->m_maxwellWaveSys->GetFieldIndex("Jx");
    const int jy_index = this->m_maxwellWaveSys->GetFieldIndex("Jy");
    const int jz_index = this->m_maxwellWaveSys->GetFieldIndex("Jz");

    // extract the expansion for the potential function phi, A
    this->phi_field = std::dynamic_pointer_cast<T>(fields[phi_index]);
    this->phi_minus_field = std::dynamic_pointer_cast<T>(fields[phi_minus_index]);
    this->ax_field = std::dynamic_pointer_cast<T>(fields[ax_index]);
    this->ay_field = std::dynamic_pointer_cast<T>(fields[ay_index]);
    this->az_field = std::dynamic_pointer_cast<T>(fields[az_index]);
    this->ax_minus_field = std::dynamic_pointer_cast<T>(fields[ax_minus_index]);
    this->ay_minus_field = std::dynamic_pointer_cast<T>(fields[ay_minus_index]);
    this->az_minus_field = std::dynamic_pointer_cast<T>(fields[az_minus_index]);

    // Extract the expansion that corresponds to the RHS of the maxwell_wave
    // equation
    this->rho_field = std::dynamic_pointer_cast<T>(fields[rho_index]);
    this->rho_minus_field = std::dynamic_pointer_cast<T>(fields[rho_minus_index]);
    this->jx_field = std::dynamic_pointer_cast<T>(fields[jx_index]);
    this->jy_field = std::dynamic_pointer_cast<T>(fields[jy_index]);
    this->jz_field = std::dynamic_pointer_cast<T>(fields[jz_index]);

    // electromagnetic field components
    this->bx_field = std::dynamic_pointer_cast<T>(fields[bx_index]);
    this->by_field = std::dynamic_pointer_cast<T>(fields[by_index]);
    this->bz_field = std::dynamic_pointer_cast<T>(fields[bz_index]);
    this->ex_field = std::dynamic_pointer_cast<T>(fields[ex_index]);
    this->ey_field = std::dynamic_pointer_cast<T>(fields[ey_index]);
    this->ez_field = std::dynamic_pointer_cast<T>(fields[ez_index]);

    for (auto &bx : this->phi_field->GetBndConditions()) {
      auto bc = bx->GetBoundaryConditionType();
      NESOASSERT(bc == ePeriodic, "Boundary condition is not periodic");
    }

    // Create evaluation object to compute the gradient of the potential field
    int pgi = 0;
    for (auto pg : this->charged_particles->particle_groups) {
      auto cit = this->charged_particles->cell_id_translations[pgi];
      this->phi_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->phi_field, pg, cit));
      this->ax_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->ax_field, pg, cit));
      this->ay_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->ay_field, pg, cit));
      this->az_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->az_field, pg, cit));

      this->bx_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->bx_field, pg, cit));
      this->by_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->by_field, pg, cit));
      this->bz_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->bz_field, pg, cit));

      this->ex_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->ex_field, pg, cit));
      this->ey_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->ey_field, pg, cit));
      this->ez_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->ez_field, pg, cit));
      this->gradax_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->ax_field, pg, cit));
      this->graday_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->ay_field, pg, cit));
      this->gradaz_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->az_field, pg, cit));

     // Create a projection object for the RHS.
     this->rho_field_project = std::make_shared<FieldProject<T>>(
         this->rho_field, this->charged_particles->particle_groups[pgi],
         this->charged_particles->cell_id_translations[pgi]);
     // Create a projection object for the RHS.
     this->jx_field_projects.push_back(std::make_shared<FieldProject<T>>(
         this->jx_field, this->charged_particles->particle_groups[pgi],
         this->charged_particles->cell_id_translations[pgi]));
     // Create a projection object for the RHS.
     this->jy_field_projects.push_back(std::make_shared<FieldProject<T>>(
         this->jy_field, this->charged_particles->particle_groups[pgi],
         this->charged_particles->cell_id_translations[pgi]));
     // Create a projection object for the RHS.
     this->jz_field_projects.push_back(std::make_shared<FieldProject<T>>(
         this->jz_field, this->charged_particles->particle_groups[pgi],
         this->charged_particles->cell_id_translations[pgi]));
      pgi += 1;
    }

    auto forcing_boundary_conditions = this->rho_field->GetBndConditions();
    for (auto &bx : forcing_boundary_conditions) {
      auto bc = bx->GetBoundaryConditionType();
      NESOASSERT(bc == ePeriodic,
                 "Boundary condition on forcing function is not periodic");
    }

    this->m_maxwellWaveSys->SetVolume(this->charged_particles->volume_nounits);

    // Don't need to neutralise the charge numerically
    //    // Compute the DOFs that correspond to a neutralising field of charge
    //    // density -1.0

    //    // First create the values at the quadrature points (uniform)
    //    this->ncd_phys_values = Array<OneD, NekDouble>(num_phys_rho);
    //    for (auto& coeff : this->ncd_phys_values) { coeff = -1.0; }

    //    this->volume = -this->rho_field->Integral(this->ncd_phys_values);

    //    // Transform the quadrature point values into DOFs
    //    this->ncd_coeff_values = Array<OneD, NekDouble>(num_coeffs_rho);
    //    for (auto& coeff : this->ncd_coeff_values) { coeff = 0.0; }

    //    this->rho_field->FwdTrans(this->ncd_phys_values,
    //    this->ncd_coeff_values);

    //    for (int cx = 0; cx < num_coeffs_rho; cx++) {
    //      NESOASSERT(std::isfinite(this->ncd_coeff_values[cx]),
    //                 "Neutralising coeff is not finite (e.g. NaN or
    //                 Inf/-Inf).");
    //    }
    //    for (auto& coeff : this->ncd_phys_values) { coeff = 0.0; }
    //    // Backward transform to ensure the quadrature point values are
    //    correct this->rho_field->BwdTrans(this->ncd_coeff_values,
    //                                     this->ncd_phys_values);

    //    for (auto& coeff : this->rho_field->UpdatePhys()) { coeff = -1.0; }

    //    const double l2_error =
    //        this->rho_field->L2(tmp_phys, this->ncd_phys_values) /
    //        this->volume;
    //
    //    std::string l2_error_msg =
    //        "This L2 error != 0 indicates a mesh/function space issue. Error:
    //        " + std::to_string(l2_error);
    //    NESOASSERT(l2_error < 1.0e-6, l2_error_msg.c_str());
    //
    //    for (int cx = 0; cx < tot_points_rho; cx++) {
    //      NESOASSERT(
    //          std::isfinite(this->ncd_phys_values[cx]),
    //          "Neutralising phys value is not finite (e.g. NaN or
    //          Inf/-Inf)..");
    //    }
  }

  inline void deposit_charge() {
    Vmath::Zero(this->rho_field->GetNcoeffs(), this->rho_field->UpdateCoeffs(), 1);
    Vmath::Zero(this->rho_field->GetNpoints(), this->rho_field->UpdatePhys(), 1);
    this->rho_field_project->project(this->charged_particles->get_rho_sym());
  }

  inline void deposit_current() {
    Vmath::Zero(this->jx_field->GetNcoeffs(), this->jx_field->UpdateCoeffs(), 1);
    Vmath::Zero(this->jx_field->GetNpoints(), this->jx_field->UpdatePhys(), 1);
    Vmath::Zero(this->jy_field->GetNcoeffs(), this->jy_field->UpdateCoeffs(), 1);
    Vmath::Zero(this->jy_field->GetNpoints(), this->jy_field->UpdatePhys(), 1);
    Vmath::Zero(this->jz_field->GetNcoeffs(), this->jz_field->UpdateCoeffs(), 1);
    Vmath::Zero(this->jz_field->GetNpoints(), this->jz_field->UpdatePhys(), 1);
    // TODO do this done better
    std::vector<Sym<REAL>> current_sym_vec = {
        this->charged_particles->get_current_sym()};
    std::vector<int> components = {0};
    for (auto jx_field_project : this->jx_field_projects) {
      jx_field_project->project(current_sym_vec, components); // 0th index component of Sym
    }
    components[0] += 1;
    for (auto jy_field_project : this->jy_field_projects) {
      jy_field_project->project(current_sym_vec, components); // 1st index component of Sym
    }
    components[0] += 1;
    for (auto jz_field_project : this->jz_field_projects) {
      jz_field_project->project(current_sym_vec, components); // 2nd index component of Sym
    }
  }

  inline void integrate_fields(const double theta, const double dtMultiplier,
      const bool chargeConservationSwitch) {
    const int rank =
        this->charged_particles->sycl_target->comm_pair.rank_parent;
    // MaxwellWave solve
    this->solve_equation_system(theta, dtMultiplier, chargeConservationSwitch);

    // Evaluate the derivative of the potential at the particle locations.
    for (uint32_t i = 0; i < this->phi_field_evaluates.size(); ++i) {
      // evalaute phi, A for potential energies
      this->phi_field_evaluates[i]->evaluate(Sym<REAL>("phi"));
      this->ax_field_evaluates[i]->evaluate(Sym<REAL>("A"), 0);
      this->ay_field_evaluates[i]->evaluate(Sym<REAL>("A"), 1);
      this->az_field_evaluates[i]->evaluate(Sym<REAL>("A"), 2);
      // evaluate B and E fields for Boris push
      this->gradax_field_evaluates[i]->evaluate(Sym<REAL>("GradAx"));
      this->graday_field_evaluates[i]->evaluate(Sym<REAL>("GradAy"));
      this->gradaz_field_evaluates[i]->evaluate(Sym<REAL>("GradAz"));
      this->ex_field_evaluates[i]->evaluate(Sym<REAL>("E"), 0);
      this->ey_field_evaluates[i]->evaluate(Sym<REAL>("E"), 1);
      this->ez_field_evaluates[i]->evaluate(Sym<REAL>("E"), 2);
    }
  }

  inline void write_fields(const int step) {
    const int rank =
        this->charged_particles->sycl_target->comm_pair.rank_parent;
    std::string name;
    name = "bx_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->bx_field, name, "bx");
    name = "by_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->by_field, name, "by");
    name = "bz_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->bz_field, name, "bz");
    name = "ex_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->ex_field, name, "ex");
    name = "ey_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->ey_field, name, "ey");
    name = "ez_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->ez_field, name, "ez");
  }


  inline void write_sources(const int step) {
    const int rank =
        this->charged_particles->sycl_target->comm_pair.rank_parent;
    std::string name;
    name = "rho_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->rho_field, name, "rho");
    name = "jx_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->jx_field, name, "jx");
    name = "jy_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->jy_field, name, "jy");
    name = "jz_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->jz_field, name, "jz");
  }

  inline void write_potentials(const int step) {
    const int rank =
        this->charged_particles->sycl_target->comm_pair.rank_parent;
    std::string name;
    name = "phi_minus_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->phi_minus_field, name, "phi_minus");
    name = "phi_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->phi_field, name, "phi");
    name = "ax_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->ax_field, name, "ax");
    name = "ay_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->ay_field, name, "ay");
    name = "az_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->az_field, name, "az");
  }

  /**
   *  Get the volume of the simulation domain.
   *
   *  @returns Volume of domain.
   */
  inline double get_volume() { return this->volume; }
};

#endif
