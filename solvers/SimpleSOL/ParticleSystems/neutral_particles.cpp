#include "neutral_particles.hpp"

void NeutralParticleSystem::InitSpec() {
  particle_spec = {ParticleProp(Sym<REAL>("POSITION"), 2, true),
                   ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                   ParticleProp(Sym<INT>("PARTICLE_ID"), 2),
                   ParticleProp(Sym<REAL>("COMPUTATIONAL_WEIGHT"), 1),
                   ParticleProp(Sym<REAL>("SOURCE_DENSITY"), 1),
                   ParticleProp(Sym<REAL>("SOURCE_ENERGY"), 1),
                   ParticleProp(Sym<REAL>("SOURCE_MOMENTUM"), 2),
                   ParticleProp(Sym<REAL>("ELECTRON_DENSITY"), 1),
                   ParticleProp(Sym<REAL>("ELECTRON_TEMPERATURE"), 1),
                   ParticleProp(Sym<REAL>("MASS"), 1),
                   ParticleProp(Sym<REAL>("VELOCITY"), 3)};
}

std::string NeutralParticleSystem::className =
    GetParticleSystemFactory().RegisterCreatorFunction(
        "NeutralParticleSystem", NeutralParticleSystem::create,
        "Neutral Particle System");