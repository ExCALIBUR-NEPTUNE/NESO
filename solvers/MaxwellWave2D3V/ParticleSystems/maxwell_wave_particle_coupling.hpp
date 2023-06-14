#ifndef __MAXWELL_WAVE_PARTICLE_COUPLING_H_
#define __MAXWELL_WAVE_PARTICLE_COUPLING_H_

#include <nektar_interface/function_evaluation.hpp>
#include <nektar_interface/function_projection.hpp>
#include <nektar_interface/particle_interface.hpp>
#include <nektar_interface/utilities.hpp>
#include <neso_particles.hpp>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Core/SessionFunction.h>
#include <SolverUtils/Driver.h>
#include <SolverUtils/EquationSystem.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mpi.h>
#include <random>
#include <string>

#include "../EquationSystems/MaxwellWavePIC.h"
#include "charged_particles.hpp"

using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace NESO::Particles;

template <typename T> class MaxwellWaveParticleCoupling {
private:
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  DriverSharedPtr driver;
  std::shared_ptr<ChargedParticles> charged_particles;

  std::shared_ptr<FieldProject<T>> rho_field_project;
  std::shared_ptr<FieldProject<T>> jx_field_project;
  std::shared_ptr<FieldProject<T>> jy_field_project;
  std::shared_ptr<FieldProject<T>> jz_field_project;
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

  Array<OneD, EquationSystemSharedPtr> equation_system;
  std::shared_ptr<MaxwellWavePIC> maxwell_wave_pic;

//  Array<OneD, NekDouble> ncd_phys_values;
//  Array<OneD, NekDouble> ncd_coeff_values;
  Array<OneD, NekDouble> rho_phys_array;
  Array<OneD, NekDouble> phi_phys_array;
  Array<OneD, NekDouble> rho_coeffs_array;
  Array<OneD, NekDouble> phi_coeffs_array;

  Array<OneD, NekDouble> jx_phys_array;
  Array<OneD, NekDouble> jy_phys_array;
  Array<OneD, NekDouble> jz_phys_array;
  Array<OneD, NekDouble> ax_phys_array;
  Array<OneD, NekDouble> ay_phys_array;
  Array<OneD, NekDouble> az_phys_array;
  Array<OneD, NekDouble> bx_phys_array;
  Array<OneD, NekDouble> by_phys_array;
  Array<OneD, NekDouble> bz_phys_array;
  Array<OneD, NekDouble> ex_phys_array;
  Array<OneD, NekDouble> ey_phys_array;
  Array<OneD, NekDouble> ez_phys_array;
  Array<OneD, NekDouble> jx_coeffs_array;
  Array<OneD, NekDouble> jy_coeffs_array;
  Array<OneD, NekDouble> jz_coeffs_array;
  Array<OneD, NekDouble> ax_coeffs_array;
  Array<OneD, NekDouble> ay_coeffs_array;
  Array<OneD, NekDouble> az_coeffs_array;
  Array<OneD, NekDouble> bx_coeffs_array;
  Array<OneD, NekDouble> by_coeffs_array;
  Array<OneD, NekDouble> bz_coeffs_array;
  Array<OneD, NekDouble> ex_coeffs_array;
  Array<OneD, NekDouble> ey_coeffs_array;
  Array<OneD, NekDouble> ez_coeffs_array;

  double volume;

  //inline void add_neutralising_field() {
  //  // get the charge integral
  //  const double total_charge = this->rho_function->Integral();
  //  NESOASSERT(std::isfinite(total_charge),
  //             "Total charge is not finite (e.g. NaN or Inf/-Inf).");

  //  // get the jx current integral
  //  const double total_jx = this->jx_function->Integral();
  //  NESOASSERT(std::isfinite(total_jx),
  //             "Total jx is not finite (e.g. NaN or Inf/-Inf).");

  //  // get the jy current integral
  //  const double total_jy = this->jy_function->Integral();
  //  NESOASSERT(std::isfinite(total_jy),
  //             "Total jy is not finite (e.g. NaN or Inf/-Inf).");

  //  // get the jz current integral
  //  const double total_jz = this->jz_function->Integral();
  //  NESOASSERT(std::isfinite(total_jz),
  //             "Total jz is not finite (e.g. NaN or Inf/-Inf).");

  //  //// Modifiable reference to coeffs
  //  //auto coeffs = this->rho_function->UpdateCoeffs();
  //  //const int num_coeffs_rho = this->rho_function->GetNcoeffs();

  //  //for (int cx = 0; cx < num_coeffs_rho; cx++) {
  //  //  NESOASSERT(std::isfinite(coeffs[cx]),
  //  //             "A forcing coefficient is not finite (e.g. NaN or Inf/-Inf).");
  //  //  coeffs[cx] += this->ncd_coeff_values[cx] * average_charge_density;
  //  //}

  //  // Modifiable reference to phys values
  //  //auto phys_values = this->rho_function->UpdatePhys();
  //  //const int num_phys_rho = this->rho_function->GetTotPoints();
  //  //for (int cx = 0; cx < num_phys_rho; cx++) {
  //  //  NESOASSERT(std::isfinite(phys_values[cx]),
  //  //             "A phys value is not finite (e.g. NaN or Inf/-Inf).");
  //  //  phys_values[cx] += this->ncd_phys_values[cx] * average_charge_density;
  //  //}

  //  // integral should be approximately 0
  //  auto integral_ref_weight = this->charged_particles->particle_weight;
  //  const auto integral_forcing_func = this->rho_function->Integral() *
  //                                     integral_ref_weight;

  //  std::string error_msg =
  //      "RHS is not neutral, log10 error: " +
  //      std::to_string(std::log10(ABS(integral_forcing_func)));
  //  NESOASSERT(ABS(integral_forcing_func) < 1.0e-6, error_msg.c_str());
  //}

  inline void solve_equation_system(const double theta,
                                    const double dtMultiplier) {
//    auto phys_rho = this->rho_function->UpdatePhys();
//    auto coeffs_rho = this->rho_function->UpdateCoeffs();
//    const double scaling_factor = -this->charged_particles->particle_weight;
//    for (int cx = 0; cx < tot_points_rho; cx++) {
//      phys_rho[cx] *= scaling_factor;
//    }
//    for (int cx = 0; cx < num_coeffs_rho; cx++) {
//      coeffs_rho[cx] = scaling_factor;
//    }

    this->maxwell_wave_pic->setDtMultiplier(dtMultiplier);
    this->maxwell_wave_pic->setTheta(theta);

    this->maxwell_wave_pic->DoSolve();
  }

public:
  /// The RHS of the maxwell_wave equation.
  std::shared_ptr<T> rho_function;
  std::shared_ptr<T> jx_function;
  std::shared_ptr<T> jy_function;
  std::shared_ptr<T> jz_function;
  /// The solution function of the maxwell_wave equation.
  std::shared_ptr<T> phi_function;
  std::shared_ptr<T> ax_function;
  std::shared_ptr<T> ay_function;
  std::shared_ptr<T> az_function;
  std::shared_ptr<T> bx_function;
  std::shared_ptr<T> by_function;
  std::shared_ptr<T> bz_function;
  std::shared_ptr<T> ex_function;
  std::shared_ptr<T> ey_function;
  std::shared_ptr<T> ez_function;

  MaxwellWaveParticleCoupling(
      LibUtilities::SessionReaderSharedPtr session,
      SpatialDomains::MeshGraphSharedPtr graph, DriverSharedPtr driver,
      std::shared_ptr<ChargedParticles> charged_particles)
      : session(session), graph(graph), driver(driver),
        charged_particles(charged_particles) {

    this->equation_system = this->driver->GetEqu();
    this->maxwell_wave_pic =
        std::dynamic_pointer_cast<MaxwellWavePIC>(this->equation_system[0]);
    auto fields = this->maxwell_wave_pic->UpdateFields();
    const int phi_index = this->maxwell_wave_pic->GetFieldIndex("phi");
    const int rho_index = this->maxwell_wave_pic->GetFieldIndex("rho");
    const int ax_index = this->maxwell_wave_pic->GetFieldIndex("Ax");
    const int ay_index = this->maxwell_wave_pic->GetFieldIndex("Ay");
    const int az_index = this->maxwell_wave_pic->GetFieldIndex("Az");
    const int bx_index = this->maxwell_wave_pic->GetFieldIndex("Bx");
    const int by_index = this->maxwell_wave_pic->GetFieldIndex("By");
    const int bz_index = this->maxwell_wave_pic->GetFieldIndex("Bz");
    const int ex_index = this->maxwell_wave_pic->GetFieldIndex("Ex");
    const int ey_index = this->maxwell_wave_pic->GetFieldIndex("Ey");
    const int ez_index = this->maxwell_wave_pic->GetFieldIndex("Ez");
    const int jx_index = this->maxwell_wave_pic->GetFieldIndex("Jx");
    const int jy_index = this->maxwell_wave_pic->GetFieldIndex("Jy");
    const int jz_index = this->maxwell_wave_pic->GetFieldIndex("Jz");

    // extract the expansion for the potential function u
    this->phi_function = std::dynamic_pointer_cast<T>(fields[phi_index]);
    this->ax_function = std::dynamic_pointer_cast<T>(fields[ax_index]);
    this->ay_function = std::dynamic_pointer_cast<T>(fields[ay_index]);
    this->az_function = std::dynamic_pointer_cast<T>(fields[az_index]);

    // Extract the expansion that corresponds to the RHS of the maxwell_wave
    // equation
    this->rho_function = std::dynamic_pointer_cast<T>(fields[rho_index]);
    this->jx_function = std::dynamic_pointer_cast<T>(fields[jx_index]);
    this->jy_function = std::dynamic_pointer_cast<T>(fields[jy_index]);
    this->jz_function = std::dynamic_pointer_cast<T>(fields[jz_index]);

    // electromagnetic field components
    this->bx_function = std::dynamic_pointer_cast<T>(fields[bx_index]);
    this->by_function = std::dynamic_pointer_cast<T>(fields[by_index]);
    this->bz_function = std::dynamic_pointer_cast<T>(fields[bz_index]);
    this->ex_function = std::dynamic_pointer_cast<T>(fields[ex_index]);
    this->ey_function = std::dynamic_pointer_cast<T>(fields[ey_index]);
    this->ez_function = std::dynamic_pointer_cast<T>(fields[ez_index]);

    const auto tot_points_phi = this->phi_function->GetTotPoints();
    const auto tot_points_rho = this->rho_function->GetTotPoints();
    const auto tot_points_ax = this->ax_function->GetTotPoints();
    const auto tot_points_ay = this->ay_function->GetTotPoints();
    const auto tot_points_az = this->az_function->GetTotPoints();
    const auto tot_points_bx = this->bx_function->GetTotPoints();
    const auto tot_points_by = this->by_function->GetTotPoints();
    const auto tot_points_bz = this->bz_function->GetTotPoints();
    const auto tot_points_ex = this->ex_function->GetTotPoints();
    const auto tot_points_ey = this->ey_function->GetTotPoints();
    const auto tot_points_ez = this->ez_function->GetTotPoints();
    const auto tot_points_jx = this->jx_function->GetTotPoints();
    const auto tot_points_jy = this->jy_function->GetTotPoints();
    const auto tot_points_jz = this->jz_function->GetTotPoints();

    const auto num_coeffs_phi = this->phi_function->GetNcoeffs();
    const auto num_coeffs_rho = this->rho_function->GetNcoeffs();
    const auto num_coeffs_ax = this->ax_function->GetNcoeffs();
    const auto num_coeffs_ay = this->ay_function->GetNcoeffs();
    const auto num_coeffs_az = this->az_function->GetNcoeffs();
    const auto num_coeffs_bx = this->bx_function->GetNcoeffs();
    const auto num_coeffs_by = this->by_function->GetNcoeffs();
    const auto num_coeffs_bz = this->bz_function->GetNcoeffs();
    const auto num_coeffs_ex = this->ex_function->GetNcoeffs();
    const auto num_coeffs_ey = this->ey_function->GetNcoeffs();
    const auto num_coeffs_ez = this->ez_function->GetNcoeffs();
    const auto num_coeffs_jx = this->jx_function->GetNcoeffs();
    const auto num_coeffs_jy = this->jy_function->GetNcoeffs();
    const auto num_coeffs_jz = this->jz_function->GetNcoeffs();

    for (auto &bx : this->phi_function->GetBndConditions()) {
      auto bc = bx->GetBoundaryConditionType();
      NESOASSERT(bc == ePeriodic, "Boundary condition is not periodic");
    }

    // Create evaluation object to compute the gradient of the potential field
    for (auto pg : this->charged_particles->particle_groups) {
      this->phi_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->phi_function, pg, this->charged_particles->cell_id_translation));
      this->ax_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->ax_function, pg, this->charged_particles->cell_id_translation));
      this->ay_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->ay_function, pg, this->charged_particles->cell_id_translation));
      this->az_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->az_function, pg, this->charged_particles->cell_id_translation));

      this->bx_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->bx_function, pg, this->charged_particles->cell_id_translation));
      this->by_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->by_function, pg, this->charged_particles->cell_id_translation));
      this->bz_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->bz_function, pg, this->charged_particles->cell_id_translation));

      this->ex_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->ex_function, pg, this->charged_particles->cell_id_translation));
      this->ey_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->ey_function, pg, this->charged_particles->cell_id_translation));
      this->ez_field_evaluates.push_back(std::make_shared<FieldEvaluate<T>>(
          this->ez_function, pg, this->charged_particles->cell_id_translation));
    }

    this->rho_phys_array = Array<OneD, NekDouble>(tot_points_rho);
    this->phi_phys_array = Array<OneD, NekDouble>(tot_points_phi);
    this->rho_coeffs_array = Array<OneD, NekDouble>(num_coeffs_rho);
    this->phi_coeffs_array = Array<OneD, NekDouble>(num_coeffs_phi);

    this->jx_phys_array = Array<OneD, NekDouble>(tot_points_jx);
    this->jy_phys_array = Array<OneD, NekDouble>(tot_points_jy);
    this->jz_phys_array = Array<OneD, NekDouble>(tot_points_jz);
    this->ax_phys_array = Array<OneD, NekDouble>(tot_points_ax);
    this->ay_phys_array = Array<OneD, NekDouble>(tot_points_ay);
    this->az_phys_array = Array<OneD, NekDouble>(tot_points_az);
    this->bx_phys_array = Array<OneD, NekDouble>(tot_points_bx);
    this->by_phys_array = Array<OneD, NekDouble>(tot_points_by);
    this->bz_phys_array = Array<OneD, NekDouble>(tot_points_bz);
    this->ex_phys_array = Array<OneD, NekDouble>(tot_points_ex);
    this->ey_phys_array = Array<OneD, NekDouble>(tot_points_ey);
    this->ez_phys_array = Array<OneD, NekDouble>(tot_points_ez);
    this->jx_coeffs_array = Array<OneD, NekDouble>(num_coeffs_jx);
    this->jy_coeffs_array = Array<OneD, NekDouble>(num_coeffs_jy);
    this->jz_coeffs_array = Array<OneD, NekDouble>(num_coeffs_jz);
    this->ax_coeffs_array = Array<OneD, NekDouble>(num_coeffs_ax);
    this->ay_coeffs_array = Array<OneD, NekDouble>(num_coeffs_ay);
    this->az_coeffs_array = Array<OneD, NekDouble>(num_coeffs_az);
    this->bx_coeffs_array = Array<OneD, NekDouble>(num_coeffs_bx);
    this->by_coeffs_array = Array<OneD, NekDouble>(num_coeffs_by);
    this->bz_coeffs_array = Array<OneD, NekDouble>(num_coeffs_bz);
    this->ex_coeffs_array = Array<OneD, NekDouble>(num_coeffs_ex);
    this->ey_coeffs_array = Array<OneD, NekDouble>(num_coeffs_ey);
    this->ez_coeffs_array = Array<OneD, NekDouble>(num_coeffs_ez);

    this->rho_function->SetPhysArray(this->rho_phys_array);
    this->phi_function->SetPhysArray(this->phi_phys_array);
    this->rho_function->SetCoeffsArray(this->rho_coeffs_array);
    this->phi_function->SetCoeffsArray(this->phi_coeffs_array);

    this->jx_function->SetPhysArray(this->jx_phys_array);
    this->jy_function->SetPhysArray(this->jy_phys_array);
    this->jz_function->SetPhysArray(this->jz_phys_array);
    this->ax_function->SetPhysArray(this->ax_phys_array);
    this->ay_function->SetPhysArray(this->ay_phys_array);
    this->az_function->SetPhysArray(this->az_phys_array);
    this->bx_function->SetPhysArray(this->bx_phys_array);
    this->by_function->SetPhysArray(this->by_phys_array);
    this->bz_function->SetPhysArray(this->bz_phys_array);
    this->ex_function->SetPhysArray(this->ex_phys_array);
    this->ey_function->SetPhysArray(this->ey_phys_array);
    this->ez_function->SetPhysArray(this->ez_phys_array);
    this->jx_function->SetCoeffsArray(this->jx_coeffs_array);
    this->jy_function->SetCoeffsArray(this->jy_coeffs_array);
    this->jz_function->SetCoeffsArray(this->jz_coeffs_array);
    this->ax_function->SetCoeffsArray(this->ax_coeffs_array);
    this->ay_function->SetCoeffsArray(this->ay_coeffs_array);
    this->az_function->SetCoeffsArray(this->az_coeffs_array);
    this->bx_function->SetCoeffsArray(this->bx_coeffs_array);
    this->by_function->SetCoeffsArray(this->by_coeffs_array);
    this->bz_function->SetCoeffsArray(this->bz_coeffs_array);
    this->ex_function->SetCoeffsArray(this->ex_coeffs_array);
    this->ey_function->SetCoeffsArray(this->ey_coeffs_array);
    this->ez_function->SetCoeffsArray(this->ez_coeffs_array);

    // Create a projection object for the RHS.
    this->rho_field_project = std::make_shared<FieldProject<T>>(
        this->rho_function, this->charged_particles->particle_groups,
        this->charged_particles->cell_id_translation);
    // Create a projection object for the RHS.
    this->jx_field_project = std::make_shared<FieldProject<T>>(
        this->jx_function, this->charged_particles->particle_groups,
        this->charged_particles->cell_id_translation);
    // Create a projection object for the RHS.
    this->jy_field_project = std::make_shared<FieldProject<T>>(
        this->jy_function, this->charged_particles->particle_groups,
        this->charged_particles->cell_id_translation);
    // Create a projection object for the RHS.
    this->jz_field_project = std::make_shared<FieldProject<T>>(
        this->jz_function, this->charged_particles->particle_groups,
        this->charged_particles->cell_id_translation);

    auto forcing_boundary_conditions =
        this->rho_function->GetBndConditions();
    for (auto &bx : forcing_boundary_conditions) {
      auto bc = bx->GetBoundaryConditionType();
      NESOASSERT(bc == ePeriodic,
                 "Boundary condition on forcing function is not periodic");
    }

// Don't need to neutralise the charge numerically
//    // Compute the DOFs that correspond to a neutralising field of charge
//    // density -1.0

//    // First create the values at the quadrature points (uniform)
//    this->ncd_phys_values = Array<OneD, NekDouble>(num_phys_rho);
//    for (auto& coeff : this->ncd_phys_values) { coeff = -1.0; }

//    this->volume = -this->rho_function->Integral(this->ncd_phys_values);

//    // Transform the quadrature point values into DOFs
//    this->ncd_coeff_values = Array<OneD, NekDouble>(num_coeffs_rho);
//    for (auto& coeff : this->ncd_coeff_values) { coeff = 0.0; }

//    this->rho_function->FwdTrans(this->ncd_phys_values, this->ncd_coeff_values);

//    for (int cx = 0; cx < num_coeffs_rho; cx++) {
//      NESOASSERT(std::isfinite(this->ncd_coeff_values[cx]),
//                 "Neutralising coeff is not finite (e.g. NaN or Inf/-Inf).");
//    }
//    for (auto& coeff : this->ncd_phys_values) { coeff = 0.0; }
//    // Backward transform to ensure the quadrature point values are correct
//    this->rho_function->BwdTrans(this->ncd_coeff_values,
//                                     this->ncd_phys_values);

//    for (auto& coeff : this->rho_function->UpdatePhys()) { coeff = -1.0; }

//    const double l2_error =
//        this->rho_function->L2(tmp_phys, this->ncd_phys_values) /
//        this->volume;
//
//    std::string l2_error_msg =
//        "This L2 error != 0 indicates a mesh/function space issue. Error: " +
//        std::to_string(l2_error);
//    NESOASSERT(l2_error < 1.0e-6, l2_error_msg.c_str());
//
//    for (int cx = 0; cx < tot_points_rho; cx++) {
//      NESOASSERT(
//          std::isfinite(this->ncd_phys_values[cx]),
//          "Neutralising phys value is not finite (e.g. NaN or Inf/-Inf)..");
//    }

    for (auto& coeff : this->rho_function->UpdatePhys()) { coeff = 0.0; }
    for (auto& coeff : this->phi_function->UpdatePhys()) { coeff = 0.0; }
    for (auto& coeff : this->rho_function->UpdateCoeffs()) { coeff = 0.0; }
    for (auto& coeff : this->phi_function->UpdateCoeffs()) { coeff = 0.0; }
    for (auto& coeff : this->ax_function->UpdateCoeffs()) { coeff = 0.0; }
    for (auto& coeff : this->ax_function->UpdatePhys()) { coeff = 0.0; }
    for (auto& coeff : this->ay_function->UpdateCoeffs()) { coeff = 0.0; }
    for (auto& coeff : this->ay_function->UpdatePhys()) { coeff = 0.0; }
    for (auto& coeff : this->az_function->UpdateCoeffs()) { coeff = 0.0; }
    for (auto& coeff : this->az_function->UpdatePhys()) { coeff = 0.0; }
    for (auto& coeff : this->jx_function->UpdateCoeffs()) { coeff = 0.0; }
    for (auto& coeff : this->jx_function->UpdatePhys()) { coeff = 0.0; }
    for (auto& coeff : this->jy_function->UpdateCoeffs()) { coeff = 0.0; }
    for (auto& coeff : this->jy_function->UpdatePhys()) { coeff = 0.0; }
    for (auto& coeff : this->jz_function->UpdateCoeffs()) { coeff = 0.0; }
    for (auto& coeff : this->jz_function->UpdatePhys()) { coeff = 0.0; }
  }

  inline void deposit_charge() {
    for (auto& coeff : this->rho_function->UpdateCoeffs()) { coeff = 0.0; }
    for (auto& coeff : this->rho_function->UpdatePhys()) { coeff = 0.0; }
    this->rho_field_project->project(this->charged_particles->get_rho_sym());
  }

  inline void deposit_current() {
    for (auto& coeff : this->jx_function->UpdateCoeffs()) { coeff = 0.0; }
    for (auto& coeff : this->jx_function->UpdatePhys()) { coeff = 0.0; }
    for (auto& coeff : this->jy_function->UpdateCoeffs()) { coeff = 0.0; }
    for (auto& coeff : this->jy_function->UpdatePhys()) { coeff = 0.0; }
    for (auto& coeff : this->jz_function->UpdateCoeffs()) { coeff = 0.0; }
    for (auto& coeff : this->jz_function->UpdatePhys()) { coeff = 0.0; }
    // TODO do this done better
    std::vector<Sym<REAL>> current_sym_vec = {this->charged_particles->get_current_sym()};
    std::vector<int> components = {0};
    this->jx_field_project->project(current_sym_vec, components);
    components[0] += 1;
    this->jy_field_project->project(current_sym_vec, components);
    components[0] += 1;
    this->jz_field_project->project(current_sym_vec, components);
  }

  inline void integrate_fields(const double theta, const double dtMultiplier) {
    // MaxwellWave solve
    this->solve_equation_system(theta, dtMultiplier);

    // Evaluate the derivative of the potential at the particle locations.
    for (uint32_t i = 0; i < this->phi_field_evaluates.size(); ++i) {
      this->phi_field_evaluates[i]->evaluate(Sym<REAL>("phi"));
      this->ax_field_evaluates[i]->evaluate(Sym<REAL>("A"), 0);
      this->ay_field_evaluates[i]->evaluate(Sym<REAL>("A"), 1);
      this->az_field_evaluates[i]->evaluate(Sym<REAL>("A"), 2);
      this->bx_field_evaluates[i]->evaluate(Sym<REAL>("B"), 0);
      this->by_field_evaluates[i]->evaluate(Sym<REAL>("B"), 1);
      this->bz_field_evaluates[i]->evaluate(Sym<REAL>("B"), 2);
      this->ex_field_evaluates[i]->evaluate(Sym<REAL>("E"), 0);
      this->ey_field_evaluates[i]->evaluate(Sym<REAL>("E"), 1);
      this->ez_field_evaluates[i]->evaluate(Sym<REAL>("E"), 2);
    }
  }

  inline void write_sources(const int step) {
    const int rank =
        this->charged_particles->sycl_target->comm_pair.rank_parent;
    std::string name =
        "rho_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->rho_function, name, "Rho");
    name = "jx_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->jx_function, name, "Jx");
    name = "jx_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->jy_function, name, "Jy");
    name = "jz_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->jz_function, name, "Jz");
  }

  inline void write_potentials(const int step) {
    const int rank =
        this->charged_particles->sycl_target->comm_pair.rank_parent;
    std::string name = "phi_" + std::to_string(rank) + "_" +
                       std::to_string(step) + ".vtu";
    write_vtu(this->phi_function, name, "phi");
    name = "ax_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->ax_function, name, "ax");
    name = "ay_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->ay_function, name, "ay");
    name = "az_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->az_function, name, "az");
  }

  /**
   *  Get the volume of the simulation domain.
   *
   *  @returns Volume of domain.
   */
  inline double get_volume() { return this->volume; }
};

#endif
