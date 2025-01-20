#include "NeutralParticleSystem.hpp"
namespace NESO::Solvers::H3LAPD
{
void NeutralParticleSystem::InitSpec() {
  this->particle_spec = {ParticleProp(Sym<REAL>("POSITION"), 3, true),
                         ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                         ParticleProp(Sym<INT>("PARTICLE_ID"), 1),
                         ParticleProp(Sym<REAL>("COMPUTATIONAL_WEIGHT"), 1),
                         ParticleProp(Sym<REAL>("SOURCE_DENSITY"), 1),
                         ParticleProp(Sym<REAL>("ELECTRON_DENSITY"), 1),
                         ParticleProp(Sym<REAL>("MASS"), 1),
                         ParticleProp(Sym<REAL>("VELOCITY"), 3)};
}

std::string NeutralParticleSystem::className =
    GetParticleSystemFactory().RegisterCreatorFunction(
        "H3LAPDParticleSystem", NeutralParticleSystem::create,
        "Neutral Particle System");
}