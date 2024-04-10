#include <boost/algorithm/string/predicate.hpp>

#include "SOLWithParticlesSystem.hpp"

namespace NESO::Solvers {
string SOLWithParticlesSystem::class_name =
    SU::GetEquationSystemFactory().RegisterCreatorFunction(
        "SOLWithParticles", SOLWithParticlesSystem::create,
        "SOL equations with particle source terms.");

SOLWithParticlesSystem::SOLWithParticlesSystem(
    const LU::SessionReaderSharedPtr &session,
    const SD::MeshGraphSharedPtr &graph)
    : SOLSystem(session, graph), m_field_to_index(session->GetVariables()) {

  m_particle_sys = std::make_shared<NeutralParticleSystem>(session, graph);
  m_required_flds.push_back("E_src");
  m_required_flds.push_back("rho_src");
  m_required_flds.push_back("rhou_src");
  m_required_flds.push_back("rhov_src");
  m_required_flds.push_back("T");

  // mass recording diagnostic creation
  m_diag_mass_recording_enabled =
      session->DefinesParameter("mass_recording_step");
}

/**
 * @brief Compute temperature from energy in advance of projection onto
 * particles
 */
void SOLWithParticlesSystem::update_temperature() {
  // Need values of rho,rhou,[rhov],[rhow],E
  int num_fields_for_T_calc = m_field_to_index.get_idx("E") + 1;
  Array<OneD, Array<OneD, NekDouble>> phys_vals(num_fields_for_T_calc);
  for (int i = 0; i < num_fields_for_T_calc; ++i) {
    phys_vals[i] = m_fields[i]->GetPhys();
  }
  auto temperature = m_fields[m_field_to_index.get_idx("T")];
  m_var_converter->GetTemperature(phys_vals, temperature->UpdatePhys());
  // Update coeffs - may not be needed?
  temperature->FwdTrans(temperature->GetPhys(), temperature->UpdateCoeffs());
}

void SOLWithParticlesSystem::v_InitObject(bool DeclareField) {
  SOLSystem::v_InitObject(DeclareField);

  // Set particle timestep from params
  m_session->LoadParameter("num_particle_steps_per_fluid_step",
                           m_num_part_substeps, 1);
  m_session->LoadParameter("particle_num_write_particle_steps",
                           m_num_write_particle_steps, 0);
  m_part_timestep = m_timestep / m_num_part_substeps;

  // Store DisContFieldSharedPtr casts of fields in a map, indexed by name, for
  // use in particle project,evaluate operations
  int idx = 0;
  for (auto &field_name : m_session->GetVariables()) {
    m_discont_fields[field_name] =
        std::dynamic_pointer_cast<MR::DisContField>(m_fields[idx]);
    idx++;
  }

  m_particle_sys->setup_project(
      m_discont_fields["rho_src"], m_discont_fields["rhou_src"],
      m_discont_fields["rhov_src"], m_discont_fields["E_src"]);

  m_particle_sys->setup_evaluate_n(m_discont_fields["rho"]);
  m_particle_sys->setup_evaluate_T(m_discont_fields["T"]);

  m_diag_mass_recording = std::make_shared<MassRecording<MR::DisContField>>(
      m_session, m_particle_sys, m_discont_fields["rho"]);
}

/**
 * @brief Destructor for SOLWithParticlesSystem class.
 */
SOLWithParticlesSystem::~SOLWithParticlesSystem() { m_particle_sys->free(); }

bool SOLWithParticlesSystem::v_PostIntegrate(int step) {
  // Writes a step of the particle trajectory.
  if (m_num_write_particle_steps > 0) {
    if ((step % m_num_write_particle_steps) == 0) {
      m_particle_sys->write(step);
      m_particle_sys->write_source_fields();
    }
  }

  if (m_diag_mass_recording_enabled) {
    m_diag_mass_recording->compute(step);
  }

  m_solver_callback_handler.call_post_integrate(this);
  return SOLSystem::v_PostIntegrate(step);
}

bool SOLWithParticlesSystem::v_PreIntegrate(int step) {
  m_solver_callback_handler.call_pre_integrate(this);

  if (m_diag_mass_recording_enabled) {
    m_diag_mass_recording->compute_initial_fluid_mass();
  }
  // Update Temperature field
  update_temperature();
  // Integrate the particle system to the requested time.
  m_particle_sys->integrate(m_time + m_timestep, m_part_timestep);
  // Project onto the source fields
  m_particle_sys->project_source_terms();

  return SOLSystem::v_PreIntegrate(step);
}

} // namespace NESO::Solvers
