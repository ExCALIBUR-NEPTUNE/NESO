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

#include "../EquationSystems/WaveEquationPIC.h"
#include "charged_particles.hpp"

using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace NESO::Particles;

template <typename T> class MaxwellWaveParticleCoupling {
private:
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  DriverSharedPtr driver;
  std::vector<std::shared_ptr<ChargedParticles> > all_species;

  std::shared_ptr<FieldProject<T>> rho_field_project;
  std::shared_ptr<FieldProject<T>> Jx_field_project;
  std::shared_ptr<FieldProject<T>> Jy_field_project;
  std::shared_ptr<FieldProject<T>> Jz_field_project;
  std::shared_ptr<FieldEvaluate<T>> phi_field_evaluate;
  std::shared_ptr<FieldEvaluate<T>> Ax_field_evaluate;
  std::shared_ptr<FieldEvaluate<T>> Ay_field_evaluate;
  std::shared_ptr<FieldEvaluate<T>> Az_field_evaluate;

  Array<OneD, EquationSystemSharedPtr> equation_system;
  std::shared_ptr<WaveEquationPIC> maxwell_wave_pic;

  Array<OneD, NekDouble> ncd_phys_values;
  Array<OneD, NekDouble> ncd_coeff_values;

  Array<OneD, NekDouble> forcing_phys;
  Array<OneD, NekDouble> forcing_coeffs;
  Array<OneD, NekDouble> potential_phys;
  Array<OneD, NekDouble> potential_coeffs;

  double volume;
  int tot_points_phi;
  int tot_points_rho;
  int num_coeffs_phi;
  int num_coeffs_rho;

  inline void add_neutralising_field() {
    // Modifiable reference to coeffs
    auto coeffs = this->rho_function->UpdateCoeffs();
    const int num_coeffs_rho = this->rho_function->GetNcoeffs();

    // get the curent charge integral
    const double total_charge = this->rho_function->Integral();
    NESOASSERT(std::isfinite(total_charge),
               "Total charge is not finite (e.g. NaN or Inf/-Inf).");

    const double average_charge_density = total_charge / this->volume;
    NESOASSERT(std::isfinite(average_charge_density),
               "Average charge density is not finite (e.g. NaN or Inf/-Inf).");

    for (int cx = 0; cx < num_coeffs_rho; cx++) {
      NESOASSERT(std::isfinite(coeffs[cx]),
                 "A forcing coefficient is not finite (e.g. NaN or Inf/-Inf).");
      coeffs[cx] += this->ncd_coeff_values[cx] * average_charge_density;
    }

    // Modifiable reference to phys values
    auto phys_values = this->rho_function->UpdatePhys();
    const int num_phys_f = this->rho_function->GetTotPoints();
    for (int cx = 0; cx < num_phys_f; cx++) {
      NESOASSERT(std::isfinite(phys_values[cx]),
                 "A phys value is not finite (e.g. NaN or Inf/-Inf).");
      phys_values[cx] += this->ncd_phys_values[cx] * average_charge_density;
    }

    // integral should be approximately 0
    auto integral_ref_weight = 0.0;
    for (const auto & species : this->all_species) {
      integral_ref_weight += species->particle_weight;
    }
    const auto integral_forcing_func = this->rho_function->Integral() *
                                       integral_ref_weight;

    std::string error_msg =
        "RHS is not neutral, log10 error: " +
        std::to_string(std::log10(ABS(integral_forcing_func)));
    NESOASSERT(ABS(integral_forcing_func) < 1.0e-6, error_msg.c_str());
  }

  inline void solve_equation_system() {
    auto phys_f = this->rho_function->UpdatePhys();
    auto coeffs_rho = this->rho_function->UpdateCoeffs();
    // TODO: how to side step this for multiple species?
    const double scaling_factor = -this->all_species[0]->particle_weight;
    for (int cx = 0; cx < tot_points_rho; cx++) {
      phys_f[cx] *= scaling_factor;
    }
    for (int cx = 0; cx < num_coeffs_rho; cx++) {
      coeffs_rho[cx] = scaling_factor;
    }

    this->maxwell_wave_pic->DoSolve();
  }

public:
  /// The RHS of the maxwell_wave equation.
  std::shared_ptr<T> rho_function;
  std::shared_ptr<T> Jx_function;
  std::shared_ptr<T> Jy_function;
  std::shared_ptr<T> Jz_function;
  /// The solution function of the maxwell_wave equation.
  std::shared_ptr<T> phi_function;
  std::shared_ptr<T> Ax_function;
  std::shared_ptr<T> Ay_function;
  std::shared_ptr<T> Az_function;

  MaxwellWaveParticleCoupling(
      LibUtilities::SessionReaderSharedPtr session,
      SpatialDomains::MeshGraphSharedPtr graph, DriverSharedPtr driver,
      std::vector<std::shared_ptr<ChargedParticles> > all_species)
      : session(session), graph(graph), driver(driver),
        all_species(all_species) {

    this->equation_system = this->driver->GetEqu();
    this->maxwell_wave_pic =
        std::dynamic_pointer_cast<WaveEquationPIC>(this->equation_system[0]);
    auto fields = this->maxwell_wave_pic->UpdateFields();
    const int phi_index = this->maxwell_wave_pic->GetFieldIndex("phi");
    const int rho_index = this->maxwell_wave_pic->GetFieldIndex("rho");
    const int Ax_index = this->maxwell_wave_pic->GetFieldIndex("Ax");
    const int Ay_index = this->maxwell_wave_pic->GetFieldIndex("Ay");
    const int Az_index = this->maxwell_wave_pic->GetFieldIndex("Az");
    const int Jx_index = this->maxwell_wave_pic->GetFieldIndex("Jx");
    const int Jy_index = this->maxwell_wave_pic->GetFieldIndex("Jy");
    const int Jz_index = this->maxwell_wave_pic->GetFieldIndex("Jz");

    // extract the expansion for the potential function u
    this->phi_function = std::dynamic_pointer_cast<T>(fields[phi_index]);

    // Extract the expansion that corresponds to the RHS of the maxwell_wave
    // equation
    this->rho_function = std::dynamic_pointer_cast<T>(fields[rho_index]);

    this->tot_points_phi = this->phi_function->GetTotPoints();
    this->tot_points_rho = this->rho_function->GetTotPoints();
    this->num_coeffs_phi = this->phi_function->GetNcoeffs();
    this->num_coeffs_rho = this->rho_function->GetNcoeffs();

    auto potential_boundary_conditions =
        this->phi_function->GetBndConditions();
    for (auto &bx : potential_boundary_conditions) {
      auto bc = bx->GetBoundaryConditionType();
      NESOASSERT(bc == ePeriodic, "Boundary condition on u is not periodic");
    }

    // Create evaluation object to compute the gradient of the potential field
    this->phi_field_evaluate = std::make_shared<FieldEvaluate<T>>(
        this->phi_function, this->all_species[0]->particle_group, // TODO!
        this->all_species[0]->cell_id_translation, true); // TODO

    this->forcing_phys = Array<OneD, NekDouble>(tot_points_rho);
    this->forcing_coeffs = Array<OneD, NekDouble>(num_coeffs_rho);
    this->rho_function->SetPhysArray(this->forcing_phys);
    this->rho_function->SetCoeffsArray(this->forcing_coeffs);

    this->potential_phys = Array<OneD, NekDouble>(tot_points_phi);
    this->potential_coeffs = Array<OneD, NekDouble>(num_coeffs_phi);
    this->phi_function->SetPhysArray(this->potential_phys);
    this->phi_function->SetCoeffsArray(this->potential_coeffs);

    // Create a projection object for the RHS.
    this->rho_field_project = std::make_shared<FieldProject<T>>(
        this->rho_function, this->all_species[0]->particle_group, // TODO!
        this->all_species[0]->cell_id_translation); // TODO!

    auto forcing_boundary_conditions =
        this->rho_function->GetBndConditions();
    for (auto &bx : forcing_boundary_conditions) {
      auto bc = bx->GetBoundaryConditionType();
      NESOASSERT(bc == ePeriodic,
                 "Boundary condition on forcing function is not periodic");
    }

    // Compute the DOFs that correspond to a neutralising field of charge
    // density -1.0

    // First create the values at the quadrature points (uniform)
    this->ncd_phys_values = Array<OneD, NekDouble>(tot_points_rho);
    for (int pointx = 0; pointx < tot_points_rho; pointx++) {
      this->ncd_phys_values[pointx] = -1.0;
    }

    this->volume = -this->rho_function->Integral(this->ncd_phys_values);

    // Transform the quadrature point values into DOFs
    this->ncd_coeff_values = Array<OneD, NekDouble>(num_coeffs_rho);
    for (int cx = 0; cx < num_coeffs_rho; cx++) {
      this->ncd_coeff_values[cx] = 0.0;
    }

    this->rho_function->FwdTrans(this->ncd_phys_values,
                                     this->ncd_coeff_values);

    for (int cx = 0; cx < num_coeffs_rho; cx++) {
      NESOASSERT(std::isfinite(this->ncd_coeff_values[cx]),
                 "Neutralising coeff is not finite (e.g. NaN or Inf/-Inf).");
    }
    for (int cx = 0; cx < tot_points_rho; cx++) {
      this->ncd_phys_values[cx] = 0.0;
    }
    // Backward transform to ensure the quadrature point values are correct
    this->rho_function->BwdTrans(this->ncd_coeff_values,
                                     this->ncd_phys_values);

    auto tmp_phys = this->rho_function->UpdatePhys();
    for (int cx = 0; cx < tot_points_rho; cx++) {
      tmp_phys[cx] = -1.0;
    }

    const double l2_error =
        this->rho_function->L2(tmp_phys, this->ncd_phys_values) /
        this->volume;

    std::string l2_error_msg =
        "This L2 error != 0 indicates a mesh/function space issue. Error: " +
        std::to_string(l2_error);
    NESOASSERT(l2_error < 1.0e-6, l2_error_msg.c_str());

    for (int cx = 0; cx < tot_points_rho; cx++) {
      NESOASSERT(
          std::isfinite(this->ncd_phys_values[cx]),
          "Neutralising phys value is not finite (e.g. NaN or Inf/-Inf)..");
    }

    auto phys_u = this->phi_function->UpdatePhys();
    auto phys_f = this->rho_function->UpdatePhys();
    for (int cx = 0; cx < tot_points_phi; cx++) {
      phys_u[cx] = 0.0;
    }
    for (int cx = 0; cx < tot_points_rho; cx++) {
      phys_f[cx] = 0.0;
    }
    auto coeffs_phi = this->phi_function->UpdateCoeffs();
    auto coeffs_rho = this->rho_function->UpdateCoeffs();
    for (int cx = 0; cx < num_coeffs_phi; cx++) {
      coeffs_phi[cx] = 0.0;
    }
    for (int cx = 0; cx < num_coeffs_rho; cx++) {
      coeffs_rho[cx] = 0.0;
    }
  }

  inline void compute_field() {

    auto phys_u = this->phi_function->UpdatePhys();
    auto phys_f = this->rho_function->UpdatePhys();
    auto coeffs_phi = this->phi_function->UpdateCoeffs();
    auto coeffs_rho = this->rho_function->UpdateCoeffs();

    for (int cx = 0; cx < tot_points_phi; cx++) {
      phys_u[cx] = 0.0;
    }
    for (int cx = 0; cx < tot_points_rho; cx++) {
      phys_f[cx] = 0.0;
    }
    for (int cx = 0; cx < num_coeffs_phi; cx++) {
      coeffs_phi[cx] = 0.0;
    }
    for (int cx = 0; cx < num_coeffs_rho; cx++) {
      coeffs_rho[cx] = 0.0;
    }

    // Project density field
    for (auto & species : this->all_species) {
      this->rho_field_project->project(species->get_charge_sym());
    }

    // Add background density to neutralise the overall system.
    this->add_neutralising_field();

    // MaxwellWave solve
    this->solve_equation_system();

    // Evaluate the derivative of the potential at the particle locations.
    for (auto & species : this->all_species) {
      this->phi_field_evaluate->evaluate(
          species->get_potential_gradient_sym());
    }
  }

  inline void write_forcing(const int step) {
    const int rank =
        this->all_species[0]->sycl_target->comm_pair.rank_parent;
    std::string name =
        "forcing_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->rho_function, name, "Forcing");
  }

  inline void write_potential(const int step) {
    // TODO: is all_species[0] all ok
    const int rank =
        this->all_species[0]->sycl_target->comm_pair.rank_parent;
    std::string name = "potential_" + std::to_string(rank) + "_" +
                       std::to_string(step) + ".vtu";
    write_vtu(this->phi_function, name, "u");
  }

  /**
   *  Get the volume of the simulation domain.
   *
   *  @returns Volume of domain.
   */
  inline double get_volume() { return this->volume; }
};

#endif
