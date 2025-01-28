#ifndef __SIMPLESOL_SOLWITHPARTICLESSYSTEM_H_
#define __SIMPLESOL_SOLWITHPARTICLESSYSTEM_H_

#include "../Diagnostics/mass_conservation.hpp"
#include "../ParticleSystems/neutral_particles.hpp"
#include "SOLSystem.hpp"
#include <solvers/solver_callback_handler.hpp>
#include <string>

namespace NESO::Solvers::SimpleSOL {
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

  /// Object that allows optional recording of stats related to mass
  /// conservation
  std::shared_ptr<MassRecording<MR::DisContField>> diag_mass_recording;

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
  /*
  Source fields cast to DisContFieldSharedPtr, indexed by name, for use in
  particle evaluation/projection methods
 */
  std::map<std::string, MR::DisContFieldSharedPtr> discont_fields;
  // Flag to toggle mass conservation checking
  bool mass_recording_enabled;
  // Number of particle timesteps per fluid timestep.
  int num_part_substeps;
  // Particle timestep size.
  double part_timestep;

  void update_temperature();

  virtual void v_InitObject(bool DeclareField) override;
  virtual bool v_PostIntegrate(int step) override;
  virtual bool v_PreIntegrate(int step) override;
};

} // namespace NESO::Solvers::SimpleSOL
#endif // SOLWITHPARTICLESSYSTEM_H
