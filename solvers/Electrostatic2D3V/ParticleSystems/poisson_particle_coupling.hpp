#ifndef __POISSON_PARTICLE_COUPLING_H_
#define __POISSON_PARTICLE_COUPLING_H_

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

#include "../EquationSystems/PoissonPIC.h"
#include "charged_particles.hpp"

using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace NESO::Particles;

template <typename T> class PoissonParticleCoupling {
private:
  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  DriverSharedPtr driver;
  std::shared_ptr<ChargedParticles> charged_particles;

  std::shared_ptr<FieldProject<T>> field_project;
  std::shared_ptr<FieldEvaluate<T>> field_evaluate;

  Array<OneD, EquationSystemSharedPtr> equation_system;
  std::shared_ptr<PoissonPIC> poisson_pic;

  Array<OneD, NekDouble> ncd_phys_values;
  Array<OneD, NekDouble> ncd_coeff_values;

  Array<OneD, NekDouble> forcing_phys;
  Array<OneD, NekDouble> forcing_coeffs;
  Array<OneD, NekDouble> potential_phys;
  Array<OneD, NekDouble> potential_coeffs;

  double volume;
  int tot_points_u;
  int tot_points_f;
  int num_coeffs_u;
  int num_coeffs_f;

  inline void add_neutralising_field() {
    // Modifiable reference to coeffs
    auto coeffs = this->forcing_function->UpdateCoeffs();
    const int num_coeffs_f = this->forcing_function->GetNcoeffs();

    // get the curent charge integral
    const double total_charge = this->forcing_function->Integral();
    NESOASSERT(std::isfinite(total_charge), "Total charge is nan.");

    const double average_charge_density = total_charge / this->volume;
    NESOASSERT(std::isfinite(average_charge_density),
               "Average charge density is nan.");

    for (int cx = 0; cx < num_coeffs_f; cx++) {
      NESOASSERT(std::isfinite(coeffs[cx]), "A forcing coefficient is nan.");
      coeffs[cx] += this->ncd_coeff_values[cx] * average_charge_density;
    }

    // Modifiable reference to phys values
    auto phys_values = this->forcing_function->UpdatePhys();
    const int num_phys_f = this->forcing_function->GetTotPoints();
    for (int cx = 0; cx < num_phys_f; cx++) {
      NESOASSERT(std::isfinite(phys_values[cx]), "A phys value is nan.");
      phys_values[cx] += this->ncd_phys_values[cx] * average_charge_density;
    }

    // integral should be approximately 0
    const auto integral_forcing_func = this->forcing_function->Integral() *
                                       this->charged_particles->particle_weight;

    // std::cout << "log err: " << std::log10(ABS(integral_forcing_func)) <<
    // std::endl;
    std::string error_msg =
        "RHS is not neutral, log10 error: " +
        std::to_string(std::log10(ABS(integral_forcing_func)));
    NESOASSERT(ABS(integral_forcing_func) < 1.0e-6, error_msg.c_str());
  }

  inline void solve_equation_system() {
    auto phys_f = this->forcing_function->UpdatePhys();
    auto coeffs_f = this->forcing_function->UpdateCoeffs();

    const double scaling_factor =
        -1.0 * this->charged_particles->particle_weight;
    for (int cx = 0; cx < tot_points_f; cx++) {
      phys_f[cx] *= scaling_factor;
    }
    for (int cx = 0; cx < num_coeffs_f; cx++) {
      coeffs_f[cx] = scaling_factor;
    }

    this->poisson_pic->DoSolve();
  }

public:
  /// The RHS of the poisson equation.
  std::shared_ptr<T> forcing_function;
  /// The solution function of the poisson equation.
  std::shared_ptr<T> potential_function;

  PoissonParticleCoupling(LibUtilities::SessionReaderSharedPtr session,
                          SpatialDomains::MeshGraphSharedPtr graph,
                          DriverSharedPtr driver,
                          std::shared_ptr<ChargedParticles> charged_particles)
      : session(session), graph(graph), driver(driver),
        charged_particles(charged_particles) {

    this->equation_system = this->driver->GetEqu();
    this->poisson_pic =
        std::dynamic_pointer_cast<PoissonPIC>(this->equation_system[0]);
    auto fields = this->poisson_pic->UpdateFields();
    const int u_index = this->poisson_pic->GetFieldIndex("u");
    const int rho_index = this->poisson_pic->GetFieldIndex("rho");

    // extract the expansion for the potential function u
    this->potential_function = std::dynamic_pointer_cast<T>(fields[u_index]);

    // Extract the expansion that corresponds to the RHS of the poisson equation
    this->forcing_function = std::dynamic_pointer_cast<T>(fields[rho_index]);

    this->tot_points_u = this->potential_function->GetTotPoints();
    this->tot_points_f = this->forcing_function->GetTotPoints();
    this->num_coeffs_u = this->potential_function->GetNcoeffs();
    this->num_coeffs_f = this->forcing_function->GetNcoeffs();

    auto potential_boundary_conditions =
        this->potential_function->GetBndConditions();
    for (auto &bx : potential_boundary_conditions) {
      auto bc = bx->GetBoundaryConditionType();
      NESOASSERT(bc == ePeriodic, "Boundary condition on u is not periodic");
    }

    // Create evaluation object to compute the gradient of the potential field
    this->field_evaluate = std::make_shared<FieldEvaluate<T>>(
        this->potential_function, this->charged_particles->particle_group,
        this->charged_particles->cell_id_translation, true);

    this->forcing_phys = Array<OneD, NekDouble>(tot_points_f);
    this->forcing_coeffs = Array<OneD, NekDouble>(num_coeffs_f);
    this->forcing_function->SetPhysArray(this->forcing_phys);
    this->forcing_function->SetCoeffsArray(this->forcing_coeffs);

    this->potential_phys = Array<OneD, NekDouble>(tot_points_u);
    this->potential_coeffs = Array<OneD, NekDouble>(num_coeffs_u);
    this->potential_function->SetPhysArray(this->potential_phys);
    this->potential_function->SetCoeffsArray(this->potential_coeffs);

    // Create a projection object for the RHS.
    this->field_project = std::make_shared<FieldProject<T>>(
        this->forcing_function, this->charged_particles->particle_group,
        this->charged_particles->cell_id_translation);

    auto forcing_boundary_conditions =
        this->forcing_function->GetBndConditions();
    for (auto &bx : forcing_boundary_conditions) {
      auto bc = bx->GetBoundaryConditionType();
      NESOASSERT(bc == ePeriodic,
                 "Boundary condition on forcing function is not periodic");
    }

    // Compute the DOFs that correspond to a neutralising field of charge
    // density -1.0

    // First create the values at the quadrature points (uniform)
    this->ncd_phys_values = Array<OneD, NekDouble>(tot_points_f);
    for (int pointx = 0; pointx < tot_points_f; pointx++) {
      this->ncd_phys_values[pointx] = -1.0;
    }

    this->volume =
        this->forcing_function->Integral(this->ncd_phys_values) * -1.0;

    // Transform the quadrature point values into DOFs
    this->ncd_coeff_values = Array<OneD, NekDouble>(num_coeffs_f);
    for (int cx = 0; cx < num_coeffs_f; cx++) {
      this->ncd_coeff_values[cx] = 0.0;
    }

    this->forcing_function->FwdTrans(this->ncd_phys_values,
                                     this->ncd_coeff_values);

    for (int cx = 0; cx < num_coeffs_f; cx++) {
      NESOASSERT(std::isfinite(this->ncd_coeff_values[cx]),
                 "Neutralising coeff is nan.");
    }
    for (int cx = 0; cx < tot_points_f; cx++) {
      this->ncd_phys_values[cx] = 0.0;
    }
    // Backward transform to ensure the quadrature point values are correct
    this->forcing_function->BwdTrans(this->ncd_coeff_values,
                                     this->ncd_phys_values);

    auto tmp_phys = this->forcing_function->UpdatePhys();
    for (int cx = 0; cx < tot_points_f; cx++) {
      tmp_phys[cx] = -1.0;
    }

    const double l2_error =
        this->forcing_function->L2(tmp_phys, this->ncd_phys_values) /
        this->volume;

    std::string l2_error_msg =
        "This L2 error != 0 indicates a mesh/function space issue. Error: " +
        std::to_string(l2_error);
    NESOASSERT(l2_error < 1.0e-6, l2_error_msg.c_str());

    for (int cx = 0; cx < tot_points_f; cx++) {
      NESOASSERT(std::isfinite(this->ncd_phys_values[cx]),
                 "Neutralising phys value is nan.");
    }

    auto phys_u = this->potential_function->UpdatePhys();
    auto phys_f = this->forcing_function->UpdatePhys();
    for (int cx = 0; cx < tot_points_u; cx++) {
      phys_u[cx] = 0.0;
    }
    for (int cx = 0; cx < tot_points_f; cx++) {
      phys_f[cx] = 0.0;
    }
    auto coeffs_u = this->potential_function->UpdateCoeffs();
    auto coeffs_f = this->forcing_function->UpdateCoeffs();
    for (int cx = 0; cx < num_coeffs_u; cx++) {
      coeffs_u[cx] = 0.0;
    }
    for (int cx = 0; cx < num_coeffs_f; cx++) {
      coeffs_f[cx] = 0.0;
    }

    // auto lambda_sin = [&](const NekDouble x, const NekDouble y) {
    //   //const double eval = std::sin(2.0 * 3.14159265359 * x) + 1.0;
    //   const double eval = ((x - 1.0) * x) + 1.0;
    //   return eval;
    // };
    //
    // interpolate_onto_nektar_field_2d(lambda_sin, this->potential_function);
  }

  inline void compute_field() {

    auto phys_u = this->potential_function->UpdatePhys();
    auto phys_f = this->forcing_function->UpdatePhys();
    auto coeffs_u = this->potential_function->UpdateCoeffs();
    auto coeffs_f = this->forcing_function->UpdateCoeffs();

    for (int cx = 0; cx < tot_points_u; cx++) {
      phys_u[cx] = 0.0;
    }
    for (int cx = 0; cx < tot_points_f; cx++) {
      phys_f[cx] = 0.0;
    }
    for (int cx = 0; cx < num_coeffs_u; cx++) {
      coeffs_u[cx] = 0.0;
    }
    for (int cx = 0; cx < num_coeffs_f; cx++) {
      coeffs_f[cx] = 0.0;
    }

    // Project density field
    this->field_project->project(this->charged_particles->get_charge_sym());

    // Add background density to neutralise the overall system.
    this->add_neutralising_field();

    // Poisson solve
    this->solve_equation_system();

    // Evaluate the derivative of the potential at the particle locations.
    this->field_evaluate->evaluate(
        this->charged_particles->get_potential_gradient_sym());
  }

  inline void write_forcing(const int step) {
    const int rank =
        this->charged_particles->sycl_target->comm_pair.rank_parent;
    std::string name =
        "forcing_" + std::to_string(rank) + "_" + std::to_string(step) + ".vtu";
    write_vtu(this->forcing_function, name, "Forcing");
  }

  inline void write_potential(const int step) {
    const int rank =
        this->charged_particles->sycl_target->comm_pair.rank_parent;
    std::string name = "potential_" + std::to_string(rank) + "_" +
                       std::to_string(step) + ".vtu";
    write_vtu(this->potential_function, name, "u");
  }

  /**
   *  Get the volume of the simulation domain.
   *
   *  @returns Volume of domain.
   */
  inline double get_volume() { return this->volume; }
};

#endif
