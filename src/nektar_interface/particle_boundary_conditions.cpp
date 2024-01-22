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
              fmod(pos + n_extent_offset_int * tmp_extent, tmp_extent);
          k_positions_dat.at(dimx) = pos_fmod + k_origin[dimx];
        }
      },
      Access::write(this->position_dat));
};

void NektarCartesianPeriodic::execute() { this->loop->execute(); }

void NektarCompositeReflection::collect() {
  // Add newly added geoms to the device map.
  auto map_composites_to_geoms =
      this->composite_collections->map_composites_to_geoms;
  for (const auto cx : composite_indices) {
    for (const auto pair_id_geom : map_composites_to_geoms[cx]) {
      if (!this->collected_geoms[cx].count(pair_id_geom.first)) {
        const int geom_id = pair_id_geom.first;
        auto geom = pair_id_geom.second;

        const int num_verts = geom->GetNumVerts();
        auto v0 = geom->GetVertex(0);
        auto v1 = geom->GetVertex(1);
        auto v2 = geom->GetVertex(num_verts - 1);

        Array<OneD, NekDouble> c0(3);
        NekDouble v00, v01, v02;
        NekDouble v10, v11, v12;
        NekDouble v20, v21, v22;
        v0->GetCoords(v00, v01, v02);
        v1->GetCoords(v10, v11, v12);
        v2->GetCoords(v20, v21, v22);
        const REAL d00 = v10 - v00;
        const REAL d01 = v11 - v01;
        const REAL d02 = v12 - v02;
        const REAL d10 = v20 - v00;
        const REAL d11 = v21 - v01;
        const REAL d12 = v22 - v02;

        NormalType nt;
        MAPPING_CROSS_PRODUCT_3D(d00, d01, d02, d10, d11, d12, nt.x, nt.y,
                                 nt.z);
        const REAL n_inorm = 1.0 / std::sqrt(MAPPING_DOT_PRODUCT_3D(
                                       nt.x, nt.y, nt.z, nt.x, nt.y, nt.z));

        nt.x = nt.x * n_inorm;
        nt.y = nt.y * n_inorm;
        nt.z = nt.z * n_inorm;

        this->map_geoms_normals->add(geom_id, nt);
        this->collected_geoms[cx].insert(pair_id_geom.first);
      }
    }
  }
  std::vector<BlockedBinaryNode<INT, NormalType, 8> *> h_root = {
      this->map_geoms_normals->root};
  this->la_root->set(h_root);
}

NektarCompositeReflection::NektarCompositeReflection(
    Sym<REAL> velocity_sym, SYCLTargetSharedPtr sycl_target,
    std::shared_ptr<CompositeInteraction::CompositeCollections>
        composite_collections,
    std::vector<int> &composite_indices)
    : velocity_sym(velocity_sym), sycl_target(sycl_target),
      composite_collections(composite_collections),
      composite_indices(composite_indices) {
  this->map_geoms_normals =
      std::make_unique<BlockedBinaryTree<INT, NormalType, 8>>(
          this->sycl_target);
  this->la_root =
      std::make_shared<LocalArray<BlockedBinaryNode<INT, NormalType, 8> *>>(
          this->sycl_target, 1);
  this->ep = std::make_unique<ErrorPropagate>(this->sycl_target);
  this->collect();
}

void NektarCompositeReflection::execute(
    std::map<int, ParticleSubGroupSharedPtr> &particle_groups) {
  this->collect();

  std::stack<ParticleLoopSharedPtr> loops;
  auto k_ep = this->ep->device_ptr();

  for (auto cx : this->composite_indices) {
    if (particle_groups.count(cx)) {
      auto pg = particle_groups.at(cx);
      auto loop = particle_loop(
          "NektarCompositeReflection", pg,
          [=](auto V, auto P, auto IC, auto IP, auto LA_ROOT) {
            const auto ROOT = LA_ROOT.at(0);
            const INT geom_id = static_cast<INT>(IC.at(2));
            NormalType *normal_location;
            bool *leaf_set;
            const bool exists =
                ROOT->get_location(geom_id, &leaf_set, &normal_location);
            if (exists && (*leaf_set)) {
              // Normal vector
              const REAL n0 = normal_location->x;
              const REAL n1 = normal_location->y;
              const REAL n2 = normal_location->z;

              const REAL v0 = V.at(0);
              const REAL v1 = V.at(1);
              const REAL v2 = V.at(2);
              const REAL in_dot_product =
                  MAPPING_DOT_PRODUCT_3D(n0, n1, n2, v0, v1, v2);

              // Reset the position to be on the composite
              P.at(0) = IP.at(0);
              P.at(1) = IP.at(1);
              P.at(2) = IP.at(2);

              // compute new velocity from reflection
              V.at(0) = v0 - 2.0 * in_dot_product * n0;
              V.at(1) = v1 - 2.0 * in_dot_product * n1;
              V.at(2) = v2 - 2.0 * in_dot_product * n2;

            } else {
              NESO_KERNEL_ASSERT(false, k_ep);
            }
          },
          Access::write(this->velocity_sym),
          Access::write(pg->get_particle_group()->position_dat),
          Access::read(Sym<INT>("NESO_COMP_INT_OUTPUT_COMP")),
          Access::read(Sym<REAL>("NESO_COMP_INT_OUTPUT_POS")),
          Access::read(this->la_root));
      loop->submit();
      loops.push(loop);
    }
  }

  while (!loops.empty()) {
    loops.top()->wait();
    loops.pop();
  }

  this->ep->check_and_throw(
      "Failed to reflect particle off geometry composite.");
}

} // namespace NESO
