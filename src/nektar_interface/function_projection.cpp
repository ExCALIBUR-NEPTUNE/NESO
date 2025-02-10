#include <nektar_interface/function_projection.hpp>

namespace NESO {
template void FieldProject<MultiRegions::DisContField>::project(
    std::shared_ptr<ParticleGroup> particle_sub_group,
    std::vector<Sym<REAL>> syms, std::vector<int> components);
template void FieldProject<MultiRegions::ContField>::project(
    std::shared_ptr<ParticleGroup> particle_sub_group,
    std::vector<Sym<REAL>> syms, std::vector<int> components);
template void FieldProject<MultiRegions::DisContField>::project(
    std::shared_ptr<ParticleSubGroup> particle_sub_group,
    std::vector<Sym<REAL>> syms, std::vector<int> components);
template void FieldProject<MultiRegions::ContField>::project(
    std::shared_ptr<ParticleSubGroup> particle_sub_group,
    std::vector<Sym<REAL>> syms, std::vector<int> components);
} // namespace NESO
