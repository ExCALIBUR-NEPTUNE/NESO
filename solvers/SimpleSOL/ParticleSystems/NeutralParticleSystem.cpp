#include "NeutralParticleSystem.hpp"
namespace NESO::Solvers::SimpleSOL {
void NeutralParticleSystem::init_spec() {
  this->particle_spec = {
      NP::ParticleProp(NP::Sym<NP::REAL>("POSITION"), 2, true),
      NP::ParticleProp(NP::Sym<NP::INT>("CELL_ID"), 1, true),
      NP::ParticleProp(NP::Sym<NP::INT>("PARTICLE_ID"), 2),
      NP::ParticleProp(NP::Sym<NP::REAL>("COMPUTATIONAL_WEIGHT"), 1),
      NP::ParticleProp(NP::Sym<NP::REAL>("SOURCE_DENSITY"), 1),
      NP::ParticleProp(NP::Sym<NP::REAL>("SOURCE_ENERGY"), 1),
      NP::ParticleProp(NP::Sym<NP::REAL>("SOURCE_MOMENTUM"), 2),
      NP::ParticleProp(NP::Sym<NP::REAL>("ELECTRON_DENSITY"), 1),
      NP::ParticleProp(NP::Sym<NP::REAL>("ELECTRON_TEMPERATURE"), 1),
      NP::ParticleProp(NP::Sym<NP::REAL>("MASS"), 1),
      NP::ParticleProp(NP::Sym<NP::REAL>("VELOCITY"), 3)};
}

std::string NeutralParticleSystem::class_name =
    GetParticleSystemFactory().RegisterCreatorFunction(
        "SOLParticleSystem", NeutralParticleSystem::create,
        "Neutral Particle System");
} // namespace NESO::Solvers::SimpleSOL