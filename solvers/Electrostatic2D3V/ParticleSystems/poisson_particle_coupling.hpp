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

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mpi.h>
#include <random>
#include <string>

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

  SessionFunctionSharedPtr forcing_session_function;
  std::shared_ptr<T> forcing_function;
  std::shared_ptr<T> potential_function;
  Array<OneD, EquationSystemSharedPtr> equation_system;

  Array<OneD, NekDouble> ncd_phys_values;
  Array<OneD, NekDouble> ncd_coeff_values;

  inline void add_neutralising_field() {
    // Modifiable reference to coeffs
    auto coeffs = this->forcing_function->UpdateCoeffs();
    const int num_coeffs = this->forcing_function->GetNcoeffs();
    for (int cx = 0; cx < num_coeffs; cx++) {
      coeffs[cx] += this->ncd_coeff_values[cx];
    }

    // Modifiable reference to phys values
    auto phys_values = this->forcing_function->UpdatePhys();
    const int num_phys = this->forcing_function->GetTotPoints();
    for (int cx = 0; cx < num_phys; cx++) {
      phys_values[cx] += this->ncd_phys_values[cx];
    }

    // integral should be approximately 0
    const auto integral_forcing_func = this->forcing_function->Integral();
    nprint("Integral", integral_forcing_func);
    NESOASSERT(ABS(integral_forcing_func) < 1.0e-8, "RHS is not neutral.");
  }

  inline void solve_equation_system() {
    // this->driver->Execute();
    this->equation_system[0]->DoSolve();
  }

public:
  PoissonParticleCoupling(LibUtilities::SessionReaderSharedPtr session,
                          SpatialDomains::MeshGraphSharedPtr graph,
                          DriverSharedPtr driver,
                          std::shared_ptr<ChargedParticles> charged_particles)
      : session(session), graph(graph), driver(driver),
        charged_particles(charged_particles) {

    // Extract the expansion that corresponds to the RHS of the poisson equation
    equation_system = this->driver->GetEqu();
    this->forcing_session_function =
        this->equation_system[0]->GetFunction("Forcing");
    auto forcing_expansion_explist =
        this->forcing_session_function->GetExpansion();
    this->forcing_function =
        std::dynamic_pointer_cast<T>(forcing_expansion_explist);

    // Create a projection object for the RHS
    this->field_project = std::make_shared<FieldProject<T>>(
        this->forcing_function, this->charged_particles->particle_group,
        this->charged_particles->cell_id_translation);

    // Compute the DOFs that correspond to a neutralising field
    const double neutralising_charge_density =
        -1.0 * this->charged_particles->get_charge_density();

    // First create the values at the quadrature points (uniform)
    const int tot_points = this->forcing_function->GetTotPoints();
    this->ncd_phys_values = Array<OneD, NekDouble>(tot_points);
    for (int pointx = 0; pointx < tot_points; pointx++) {
      this->ncd_phys_values[pointx] = neutralising_charge_density;
    }

    // Transform the quadrature point values into DOFs
    this->ncd_coeff_values =
        Array<OneD, NekDouble>((unsigned)this->forcing_function->GetNcoeffs());
    this->forcing_function->FwdTrans(this->ncd_phys_values,
                                     this->ncd_coeff_values);

    // extract the expansion for the potential function u
    auto fields = this->equation_system[0]->UpdateFields();
    this->potential_function = std::dynamic_pointer_cast<T>(fields[0]);

    // Create evaluation object to compute the gradient of the potential field
    this->field_evaluate = std::make_shared<FieldEvaluate<T>>(
        this->potential_function, this->charged_particles->particle_group,
        this->charged_particles->cell_id_translation, true);
  }

  inline void compute_field() {

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
    std::string name = "forcing_" + std::to_string(step) + ".vtu";
    write_vtu(this->forcing_function, name, "Forcing");
  }

  inline void write_potential(const int step) {
    std::string name = "potential_" + std::to_string(step) + ".vtu";
    write_vtu(this->forcing_function, name, "u");
  }
};

#endif
