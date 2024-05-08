#ifndef __SIMPLESOL_SOLWITHPARTICLESSYSTEM_H_
#define __SIMPLESOL_SOLWITHPARTICLESSYSTEM_H_

#include "../Diagnostics/mass_conservation.hpp"
#include "../ParticleSystems/neutral_particles.hpp"
#include "SOLSystem.hpp"
#include <solvers/solver_callback_handler.hpp>
#include <string>

namespace NESO::Solvers {
/**
 *
 */
class SOLWithParticlesSystem : public SOLSystem {
public:
  friend class MemoryManager<SOLWithParticlesSystem>;

  /// Name of class.
  static std::string class_name;

  /// Callback handler to call user defined callbacks.
  SolverCallbackHandler<SOLWithParticlesSystem> solver_callback_handler;

  // Object that allows optional recording of stats related to mass conservation
  std::shared_ptr<MassRecording<MR::DisContField>> m_diag_mass_recording;

  /// Creates an instance of this class.
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &session,
         const SD::MeshGraphSharedPtr &graph) {
    SU::EquationSystemSharedPtr equation_sys =
        MemoryManager<SOLWithParticlesSystem>::AllocateSharedPtr(session,
                                                                 graph);
    equation_sys->InitObject();
    return equation_sys;
  }

  SOLWithParticlesSystem(const LU::SessionReaderSharedPtr &session,
                         const SD::MeshGraphSharedPtr &graph);

protected:
  // Flag to toggle mass conservation checking
  bool mass_recording_enabled;
  // Number of particle timesteps per fluid timestep.
  int m_num_part_substeps;
  // Number of time steps between particle trajectory step writes.
  int m_num_write_particle_steps;
  // Particle timestep size.
  double m_part_timestep;

  /*
  Source fields cast to DisContFieldSharedPtr, indexed by name, for use in
  particle evaluation/projection methods
 */
  std::map<std::string, MR::DisContFieldSharedPtr> m_discont_fields;

  void update_temperature();

  virtual void v_InitObject(bool DeclareField) override;
  virtual bool v_PostIntegrate(int step) override;
  virtual bool v_PreIntegrate(int step) override;
};

} // namespace NESO::Solvers
#endif // SOLWITHPARTICLESSYSTEM_H
