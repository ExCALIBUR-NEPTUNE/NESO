#ifndef __SIMPLESOL_MASS_CONSERVATION_H_
#define __SIMPLESOL_MASS_CONSERVATION_H_

#include <memory>
#include <mpi.h>
#include <neso_particles.hpp>
using namespace NESO;
using namespace NESO::Particles;

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
using namespace Nektar;

#include "../ParticleSystems/neutral_particles.hpp"

template <typename T> class MassRecording {
protected:
    const LibUtilities::SessionReaderSharedPtr session;
    std::shared_ptr<NeutralParticleSystem> particle_sys;
    std::shared_ptr<T> rho;
public:
  MassRecording(
    const LibUtilities::SessionReaderSharedPtr session,
    std::shared_ptr<NeutralParticleSystem> particle_sys,
    std::shared_ptr<T> rho
  ) : session(session), particle_sys(particle_sys), rho(rho) {

  };

};

#endif
