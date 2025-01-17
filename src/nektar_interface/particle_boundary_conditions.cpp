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
};

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
  this->ep = std::make_unique<ErrorPropagate>(this->sycl_target);

  this->reset_distance = config->get<REAL>(
      "NektarCompositeTruncatedReflection/reset_distance", 1.0e-7);
}

void NektarCompositeTruncatedReflection::pre_advection(
    ParticleSubGroupSharedPtr particle_sub_group) {
  this->composite_intersection->pre_integration(particle_sub_group);
}

void NektarCompositeTruncatedReflection::execute_2d(
    ParticleSubGroupSharedPtr particle_sub_group) {
  auto particle_groups =
      this->composite_intersection->get_intersections(particle_sub_group);

  auto k_ep = this->ep->device_ptr();
  const REAL k_reset_distance = this->reset_distance;

  const auto k_normal_device_mapper =
      this->composite_intersection->composite_collections
          ->get_device_normal_mapper();

  if (particle_groups.count(1)) {
    auto pg = particle_groups.at(1);
    particle_loop(
        "NektarCompositeTruncatedReflection2D", pg,
        [=](auto V, auto P, auto PP, auto IC, auto IP, auto TSP) {
          REAL *normal;
          const bool normal_exists =
              k_normal_device_mapper.get(IC.at(2), &normal);
          NESO_KERNEL_ASSERT(normal_exists, k_ep);
          if (normal_exists) {
            // Normal vector
            const REAL n0 = normal[0];
            const REAL n1 = normal[1];
            const REAL p0 = P.at(0);
            const REAL p1 = P.at(1);
            const REAL v0 = V.at(0);
            const REAL v1 = V.at(1);
            // We don't know if the normal is inwards pointing or outwards
            // pointing.
            const REAL in_dot_product = KERNEL_DOT_PRODUCT_2D(n0, n1, v0, v1);

            // compute new velocity from reflection
            V.at(0) = v0 - 2.0 * in_dot_product * n0;
            V.at(1) = v1 - 2.0 * in_dot_product * n1;

            // Try and compute a sane new position
            // vector from intersection point back towards previous position
            const REAL oo0 = PP.at(0) - IP.at(0);
            const REAL oo1 = PP.at(1) - IP.at(1);
            REAL o0 = oo0;
            REAL o1 = oo1;

            const REAL o_norm2 = KERNEL_DOT_PRODUCT_2D(oo0, oo1, oo0, oo1);
            const REAL o_norm = Kernel::sqrt(o_norm2);
            const bool small_move = o_norm < (k_reset_distance * 0.1);

            const REAL o_inorm =
                small_move ? k_reset_distance : k_reset_distance / o_norm;
            o0 *= o_inorm;
            o1 *= o_inorm;
            // If the move is tiny place the particle back on the previous
            // position
            REAL np0 = small_move ? PP.at(0) : IP.at(0) + o0;
            REAL np1 = small_move ? PP.at(1) : IP.at(1) + o1;
            // Detect if we moved the particle back past the previous position
            // Both PP - np and PP - IP should have the same sign
            const bool moved_past_pp =
                ((PP.at(0) - np0) * o0 < 0.0) || ((PP.at(1) - np1) * o1 < 0.0);

            np0 = moved_past_pp ? PP.at(0) : np0;
            np1 = moved_past_pp ? PP.at(1) : np1;
            P.at(0) = np0;
            P.at(1) = np1;

            // Timestepping adjustment
            const REAL dist_trunc_step = o_norm2;

            const REAL f0 = p0 - PP.at(0);
            const REAL f1 = p1 - PP.at(1);
            const REAL dist_full_step = KERNEL_DOT_PRODUCT_2D(f0, f1, f0, f1);

            REAL tmp_prop_achieved = dist_full_step > 1.0e-16
                                         ? dist_trunc_step / dist_full_step
                                         : 1.0;
            tmp_prop_achieved =
                tmp_prop_achieved < 0.0 ? 0.0 : tmp_prop_achieved;
            tmp_prop_achieved =
                tmp_prop_achieved > 1.0 ? 1.0 : tmp_prop_achieved;

            // proportion along the full step that we truncated at
            const REAL proportion_achieved = Kernel::sqrt(tmp_prop_achieved);
            const REAL last_dt = TSP.at(1);
            const REAL correct_last_dt = TSP.at(1) * proportion_achieved;
            TSP.at(0) = TSP.at(0) - last_dt + correct_last_dt;
            TSP.at(1) = correct_last_dt;
          }
        },
        Access::write(this->velocity_sym),
        Access::write(pg->get_particle_group()->position_dat),
        Access::read(Sym<REAL>("NESO_COMP_INT_PREV_POS")),
        Access::read(Sym<INT>("NESO_COMP_INT_OUTPUT_COMP")),
        Access::read(Sym<REAL>("NESO_COMP_INT_OUTPUT_POS")),
        Access::write(this->time_step_prop_sym))
        ->execute();
  }

  this->ep->check_and_throw(
      "Failed to reflect particle off geometry composite.");
}

void NektarCompositeTruncatedReflection::execute_3d(
    ParticleSubGroupSharedPtr particle_sub_group) {
  auto particle_groups =
      this->composite_intersection->get_intersections(particle_sub_group);

  auto k_ep = this->ep->device_ptr();
  const REAL k_reset_distance = this->reset_distance;

  const auto k_normal_device_mapper =
      this->composite_intersection->composite_collections
          ->get_device_normal_mapper();

  if (particle_groups.count(1)) {
    auto pg = particle_groups.at(1);
    particle_loop(
        "NektarCompositeTruncatedReflection3D", pg,
        [=](auto V, auto P, auto PP, auto IC, auto IP, auto TSP) {
          const INT geom_id = static_cast<INT>(IC.at(2));
          REAL *normal;
          const bool exists = k_normal_device_mapper.get(geom_id, &normal);
          NESO_KERNEL_ASSERT(exists, k_ep);

          // Normal vector
          const REAL n0 = normal[0];
          const REAL n1 = normal[1];
          const REAL n2 = normal[2];
          const REAL p0 = P.at(0);
          const REAL p1 = P.at(1);
          const REAL p2 = P.at(2);
          const REAL v0 = V.at(0);
          const REAL v1 = V.at(1);
          const REAL v2 = V.at(2);
          const REAL in_dot_product =
              MAPPING_DOT_PRODUCT_3D(n0, n1, n2, v0, v1, v2);

          // compute new velocity from reflection
          V.at(0) = v0 - 2.0 * in_dot_product * n0;
          V.at(1) = v1 - 2.0 * in_dot_product * n1;
          V.at(2) = v2 - 2.0 * in_dot_product * n2;

          // Compute a sane new position

          // vector from intersection point back towards previous position
          const REAL oo0 = PP.at(0) - IP.at(0);
          const REAL oo1 = PP.at(1) - IP.at(1);
          const REAL oo2 = PP.at(2) - IP.at(2);
          REAL o0 = oo0;
          REAL o1 = oo1;
          REAL o2 = oo2;

          const REAL o_norm2 =
              MAPPING_DOT_PRODUCT_3D(oo0, oo1, oo2, oo0, oo1, oo2);
          const REAL o_norm = sqrt(o_norm2);
          const bool small_move = o_norm < (k_reset_distance * 0.1);
          const REAL o_inorm =
              small_move ? k_reset_distance : k_reset_distance / o_norm;
          o0 *= o_inorm;
          o1 *= o_inorm;
          o2 *= o_inorm;
          // If the move is tiny place the particle back on the previous
          // position
          REAL np0 = small_move ? PP.at(0) : IP.at(0) + o0;
          REAL np1 = small_move ? PP.at(1) : IP.at(1) + o1;
          REAL np2 = small_move ? PP.at(2) : IP.at(2) + o2;
          // Detect if we moved the particle back past the previous position
          // Both PP - np and PP - IP should have the same sign
          const bool moved_past_pp = ((PP.at(0) - np0) * o0 < 0.0) ||
                                     ((PP.at(1) - np1) * o1 < 0.0) ||
                                     ((PP.at(2) - np2) * o2 < 0.0);

          np0 = moved_past_pp ? PP.at(0) : np0;
          np1 = moved_past_pp ? PP.at(1) : np1;
          np2 = moved_past_pp ? PP.at(2) : np2;

          P.at(0) = np0;
          P.at(1) = np1;
          P.at(2) = np2;

          // Timestepping adjustment
          const REAL dist_trunc_step = o_norm2;

          const REAL f0 = p0 - PP.at(0);
          const REAL f1 = p1 - PP.at(1);
          const REAL f2 = p2 - PP.at(2);
          const REAL dist_full_step =
              MAPPING_DOT_PRODUCT_3D(f0, f1, f2, f0, f1, f2);

          REAL tmp_prop_achieved =
              dist_full_step > 1.0e-16 ? dist_trunc_step / dist_full_step : 1.0;
          tmp_prop_achieved = tmp_prop_achieved < 0.0 ? 0.0 : tmp_prop_achieved;
          tmp_prop_achieved = tmp_prop_achieved > 1.0 ? 1.0 : tmp_prop_achieved;

          // proportion along the full step that we truncated at
          const REAL proportion_achieved = sqrt(tmp_prop_achieved);
          const REAL last_dt = TSP.at(1);
          const REAL correct_last_dt = TSP.at(1) * proportion_achieved;
          TSP.at(0) = TSP.at(0) - last_dt + correct_last_dt;
          TSP.at(1) = correct_last_dt;
        },
        Access::write(this->velocity_sym),
        Access::write(pg->get_particle_group()->position_dat),
        Access::read(Sym<REAL>("NESO_COMP_INT_PREV_POS")),
        Access::read(Sym<INT>("NESO_COMP_INT_OUTPUT_COMP")),
        Access::read(Sym<REAL>("NESO_COMP_INT_OUTPUT_POS")),
        Access::write(this->time_step_prop_sym))
        ->execute();
  }

  this->ep->check_and_throw(
      "Failed to reflect particle off geometry composite.");
}

void NektarCompositeTruncatedReflection::execute(
    ParticleSubGroupSharedPtr particle_sub_group) {
  NESOASSERT(this->ndim == 3 || this->ndim == 2,
             "Unexpected number of dimensions.");
  if (this->ndim == 3) {
    this->execute_3d(particle_sub_group);
  } else {
    this->execute_2d(particle_sub_group);
  }
}

} // namespace NESO
