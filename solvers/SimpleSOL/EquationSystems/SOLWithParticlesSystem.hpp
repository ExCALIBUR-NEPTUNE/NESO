#ifndef SOLWITHPARTICLESSYSTEM_H
#define SOLWITHPARTICLESSYSTEM_H

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
  static std::string className;

  /// Callback handler to call user defined callbacks.
  SolverCallbackHandler<SOLWithParticlesSystem> m_solver_callback_handler;

  // Object that allows optional recording of stats related to mass conservation
  std::shared_ptr<MassRecording<MR::DisContField>> m_diag_mass_recording;

  /// Creates an instance of this class.
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &pSession,
         const SD::MeshGraphSharedPtr &pGraph) {
    SU::EquationSystemSharedPtr p =
        MemoryManager<SOLWithParticlesSystem>::AllocateSharedPtr(pSession,
                                                                 pGraph);
    p->InitObject();
    return p;
  }

  SOLWithParticlesSystem(const LU::SessionReaderSharedPtr &pSession,
                         const SD::MeshGraphSharedPtr &pGraph);

  virtual ~SOLWithParticlesSystem();


protected:
  // Flag to toggle mass conservation checking
  bool m_diag_mass_recording_enabled;
  // Map of field name to field index
  NESO::NektarFieldIndexMap m_field_to_index;
  // Particles system object
  std::shared_ptr<NeutralParticleSystem> m_particle_sys;
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

  void UpdateTemperature();
  virtual void v_InitObject(bool DeclareField) override;
  virtual bool v_PostIntegrate(int step) override;
  virtual bool v_PreIntegrate(int step) override;
};

} // namespace NESO::Solvers
#endif // SOLWITHPARTICLESSYSTEM_H
