#include "NeutralParticleSystem.hpp"
namespace NESO::Solvers::SimpleSOL {
void NeutralParticleSystem::init_spec() {
  this->particle_spec = {ParticleProp(Sym<REAL>("POSITION"), 2, true),
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

std::string NeutralParticleSystem::class_name =
    GetParticleSystemFactory().RegisterCreatorFunction(
        "SOLParticleSystem", NeutralParticleSystem::create,
        "Neutral Particle System");
} // namespace NESO::Solvers::SimpleSOL