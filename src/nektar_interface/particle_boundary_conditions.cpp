#include <nektar_interface/particle_boundary_conditions.hpp>

namespace NESO {

NektarCartesianPeriodic::NektarCartesianPeriodic(
    SYCLTargetSharedPtr sycl_target,
    Nektar::SpatialDomains::MeshGraphSharedPtr graph,
    ParticleDatSharedPtr<REAL> position_dat)
    : sycl_target(sycl_target), ndim(graph->GetMeshDimension()),
      position_dat(position_dat), d_extents(sycl_target, 3),
      d_origin(sycl_target, 3) {

  NESOASSERT(this->ndim <= 3, "bad mesh ndim");

  auto verticies = graph->GetAllPointGeoms();

  double origin[3];
  double extent[3];
  for (int dimx = 0; dimx < 3; dimx++) {
    origin[dimx] = std::numeric_limits<double>::max();
    extent[dimx] = std::numeric_limits<double>::min();
  }

  for (auto &vx : verticies) {
    Nektar::NekDouble x, y, z;
    vx.second->GetCoords(x, y, z);
    origin[0] = std::min(origin[0], x);
    origin[1] = std::min(origin[1], y);
    origin[2] = std::min(origin[2], z);
    extent[0] = std::max(extent[0], x);
    extent[1] = std::max(extent[1], y);
    extent[2] = std::max(extent[2], z);
  }

  MPICHK(MPI_Allreduce(origin, this->global_origin, 3, MPI_DOUBLE, MPI_MIN,
                       sycl_target->comm_pair.comm_parent));
  MPICHK(MPI_Allreduce(extent, this->global_extent, 3, MPI_DOUBLE, MPI_MAX,
                       sycl_target->comm_pair.comm_parent));

  for (int dimx = 0; dimx < 3; dimx++) {
    this->global_extent[dimx] -= this->global_origin[dimx];
  }

  sycl_target->queue
      .memcpy(this->d_extents.ptr, this->global_extent,
              this->ndim * sizeof(double))
      .wait_and_throw();

  sycl_target->queue
      .memcpy(this->d_origin.ptr, this->global_origin,
              this->ndim * sizeof(double))
      .wait_and_throw();

  const int k_ndim = this->ndim;
  NESOASSERT(((k_ndim > 0) && (k_ndim < 4)), "Bad number of dimensions");
  const auto k_origin = this->d_origin.ptr;
  const auto k_extents = this->d_extents.ptr;
  this->loop = particle_loop(
      "NektarCartesianPeriodic", this->position_dat,
      [=](auto k_positions_dat) {
        for (int dimx = 0; dimx < k_ndim; dimx++) {
          const double pos = k_positions_dat.at(dimx) - k_origin[dimx];
          // offset the position in the current dimension to be
          // positive by adding a value times the extent
          const double n_extent_offset_real = ABS(pos);
          const double tmp_extent = k_extents[dimx];
          const INT n_extent_offset_int = n_extent_offset_real + 2.0;
          const double pos_fmod =
              sycl::fmod(pos + n_extent_offset_int * tmp_extent, tmp_extent);
          k_positions_dat.at(dimx) = pos_fmod + k_origin[dimx];
        }
      },
      Access::write(this->position_dat));
}

void NektarCartesianPeriodic::execute() { this->loop->execute(); }

NektarCompositeTruncatedReflection::NektarCompositeTruncatedReflection(
    Sym<REAL> velocity_sym, Sym<REAL> time_step_prop_sym,
    SYCLTargetSharedPtr sycl_target,
    std::shared_ptr<ParticleMeshInterface> mesh,
    std::vector<int> &composite_indices, ParameterStoreSharedPtr config)
    : velocity_sym(velocity_sym), time_step_prop_sym(time_step_prop_sym),
      sycl_target(sycl_target), composite_indices(composite_indices),
      ndim(mesh->get_ndim()) {

  std::map<int, std::vector<int>> boundary_groups = {
      {1, this->composite_indices}};
  this->composite_intersection =
      std::make_shared<CompositeInteraction::CompositeIntersection>(
          this->sycl_target, mesh, boundary_groups);

  this->reset_distance = config->get<REAL>(
      "NektarCompositeTruncatedReflection/reset_distance", 1.0e-7);

  this->boundary_reflection =
      std::make_shared<BoundaryReflection>(this->ndim, this->reset_distance);
}

void NektarCompositeTruncatedReflection::pre_advection(
    ParticleSubGroupSharedPtr particle_sub_group) {
  this->composite_intersection->pre_integration(particle_sub_group);
}

void NektarCompositeTruncatedReflection::execute(
    ParticleSubGroupSharedPtr particle_sub_group) {
  NESOASSERT(this->ndim == 3 || this->ndim == 2,
             "Unexpected number of dimensions.");

  auto groups =
      this->composite_intersection->get_intersections(particle_sub_group);

  for (auto &groupx : groups) {
    this->boundary_reflection->execute(
        groupx.second,
        get_particle_group(particle_sub_group)->position_dat->sym,
        this->velocity_sym, this->time_step_prop_sym,
        this->composite_intersection->previous_position_sym);
  }
}

} // namespace NESO
