///////////////////////////////////////////////////////////////////////////////
//
// File SOLWithParticlesSystem.h
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description: Adds particles to SOLSystem
//
///////////////////////////////////////////////////////////////////////////////

#ifndef SOLWITHPARTICLESSYSTEM_H
#define SOLWITHPARTICLESSYSTEM_H

#include "../Diagnostics/mass_conservation.hpp"
#include "../ParticleSystems/neutral_particles.hpp"
#include "SOLSystem.h"
#include <string>
#include <solvers/solver_callback_handler.hpp>

namespace Nektar {
/**
 *
 */
class SOLWithParticlesSystem : public SOLSystem,
                               virtual public SolverUtils::AdvectionSystem,
                               virtual public SolverUtils::FluidInterface {
public:
  friend class MemoryManager<SOLWithParticlesSystem>;

  /// Name of class.
  static std::string className;

  /// Callback handler to call user defined callbacks.
  SolverCallbackHandler<SOLWithParticlesSystem> m_solver_callback_handler;

  // Object that allows optional recording of stats related to mass conservation
  std::shared_ptr<MassRecording<MultiRegions::DisContField>>
      m_diag_mass_recording;

  /// Creates an instance of this class.
  static SolverUtils::EquationSystemSharedPtr
  create(const LibUtilities::SessionReaderSharedPtr &pSession,
         const SpatialDomains::MeshGraphSharedPtr &pGraph) {
    SolverUtils::EquationSystemSharedPtr p =
        MemoryManager<SOLWithParticlesSystem>::AllocateSharedPtr(pSession,
                                                                 pGraph);
    p->InitObject();
    return p;
  }

  SOLWithParticlesSystem(const LibUtilities::SessionReaderSharedPtr &pSession,
                         const SpatialDomains::MeshGraphSharedPtr &pGraph);

  virtual ~SOLWithParticlesSystem();

  /**
   *  Get a field in the equation system by specifiying the field name.
   *
   *  @param field_name Name of field to extract.
   *  @returns Requested field if it exists otherwise nullptr
   */
  ExpListSharedPtr GetField(const std::string field_name);

  /**
   *  Get a shared pointer to the neutral particle system.
   *
   *  @returns Pointer to neutral particle system.
   */
  std::shared_ptr<NeutralParticleSystem> GetNeutralParticleSystem();


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
  std::map<std::string, MultiRegions::DisContFieldSharedPtr> m_discont_fields;

  void UpdateTemperature();
  virtual void v_InitObject(bool DeclareField) override;
  virtual bool v_PostIntegrate(int step) override;
  virtual bool v_PreIntegrate(int step) override;
};

} // namespace Nektar
#endif // SOLWITHPARTICLESSYSTEM_H
