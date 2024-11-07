#ifndef __FUNCTION_BARY_EVALUATION_H_
#define __FUNCTION_BARY_EVALUATION_H_
#include "coordinate_mapping.hpp"
#include "particle_interface.hpp"
#include <map>
#include <memory>
#include <neso_particles.hpp>

#include <LocalRegions/QuadExp.h>
#include <LocalRegions/TriExp.h>
#include <StdRegions/StdExpansion2D.h>

#include "bary_interpolation/bary_evaluation.hpp"
#include "expansion_looping/geom_to_expansion_builder.hpp"
#include "geometry_transport/shape_mapping.hpp"
#include "utility_sycl.hpp"

using namespace NESO::Particles;
using namespace Nektar::LocalRegions;
using namespace Nektar::StdRegions;

namespace NESO {

/**
 *  Evaluate 2D expansions at particle locations using Bary Interpolation.
 * Reimplements the algorithm in Nektar++.
 */
template <typename T> class BaryEvaluateBase : GeomToExpansionBuilder {
protected:
  int ndim;
  std::shared_ptr<T> field;
  ParticleMeshInterfaceSharedPtr mesh;
  SYCLTargetSharedPtr sycl_target;
  BufferDeviceHost<NekDouble> dh_global_physvals;

  struct CellInfo {
    int shape_type_int;
    std::size_t num_phys[3];
    REAL const *d_z[3];
    REAL const *d_bw[3];
    std::size_t phys_offset;
  };

  using MapKey = std::tuple<int, std::array<std::size_t, 3>>;

  std::map<MapKey, CellInfo> map_cells_to_info;

  std::shared_ptr<BufferDevice<CellInfo>> d_cell_info;
  std::stack<std::shared_ptr<BufferDevice<REAL>>> stack_ptrs;

  /// Stride between z and weight values in each dimension for all expansion
  /// types.
  std::size_t max_num_phys;

public:
  /// Disable (implicit) copies.
  BaryEvaluateBase(const BaryEvaluateBase &st) = delete;
  /// Disable (implicit) copies.
  BaryEvaluateBase &operator=(BaryEvaluateBase const &a) = delete;

  /**
   *  Create instance to evaluate a passed field.
   *
   *  @param field 2D Nektar field instance.
   *  @param mesh ParticleMeshInterface containing the same MeshGraph the field
   * is defined on.
   *  @param cell_id_translation CellIDTranslation instance in use with the
   * ParticleMeshInterface.
   */
  BaryEvaluateBase(std::shared_ptr<T> field,
                   ParticleMeshInterfaceSharedPtr mesh,
                   CellIDTranslationSharedPtr cell_id_translation)
      : ndim(mesh->get_ndim()), field(field), mesh(mesh),
        sycl_target(cell_id_translation->sycl_target),
        dh_global_physvals(sycl_target, 1) {

    // build the map from geometry ids to expansion ids
    std::map<int, int> geom_to_exp;
    build_geom_to_expansion_map(this->field, geom_to_exp);
    const int neso_cell_count = mesh->get_cell_count();

    std::vector<CellInfo> v_cell_info;
    v_cell_info.reserve(neso_cell_count);
    this->max_num_phys = 0;

    for (int neso_cellx = 0; neso_cellx < neso_cell_count; neso_cellx++) {
      const int nektar_geom_id = cell_id_translation->map_to_nektar[neso_cellx];
      const int expansion_id = geom_to_exp[nektar_geom_id];
      // get the nektar expansion
      auto expansion = this->field->GetExp(expansion_id);

      MapKey key;
      std::get<0>(key) =
          shape_type_to_int(expansion->GetGeom()->GetShapeType());
      for (int dimx = 0; dimx < 3; dimx++) {
        std::get<1>(key)[dimx] = 0;
      }
      for (int dimx = 0; dimx < this->ndim; dimx++) {
        std::get<1>(key)[dimx] =
            static_cast<std::size_t>(expansion->GetBase()[dimx]->GetZ().size());
      }
      if (this->map_cells_to_info.count(key) == 0) {
        CellInfo cell_info;
        cell_info.shape_type_int = std::get<0>(key);
        for (int dx = 0; dx < 3; dx++) {
          cell_info.num_phys[dx] = std::get<1>(key)[dx];
        }

        // Get the z values and bw values for this geom type with this number
        // of physvals
        for (int dimx = 0; dimx < this->ndim; dimx++) {
          auto base = expansion->GetBase();
          const auto &z = base[dimx]->GetZ();
          const auto &bw = base[dimx]->GetBaryWeights();
          NESOASSERT(z.size() == bw.size(),
                     "Expected these two sizes to match.");
          const auto size = z.size();
          this->max_num_phys = std::max(this->max_num_phys, size);
          std::vector<REAL> tmp_reals(2 * size);
          for (int cx = 0; cx < size; cx++) {
            tmp_reals.at(cx) = z[cx];
          }
          for (int cx = 0; cx < size; cx++) {
            tmp_reals.at(cx + size) = bw[cx];
          }
          auto ptr = std::make_shared<BufferDevice<REAL>>(this->sycl_target,
                                                          tmp_reals);
          auto d_ptr = ptr->ptr;
          this->stack_ptrs.push(ptr);

          cell_info.d_z[dimx] = d_ptr;
          cell_info.d_bw[dimx] = d_ptr + size;
        }

        this->map_cells_to_info[key] = cell_info;
      }

      // Get the generic info for this cell type and number of modes
      auto cell_info = this->map_cells_to_info.at(key);
      // push on the offset for this particular cell
      cell_info.phys_offset =
          static_cast<std::size_t>(this->field->GetPhys_Offset(expansion_id));

      v_cell_info.push_back(cell_info);
    }

    this->d_cell_info = std::make_shared<BufferDevice<CellInfo>>(
        this->sycl_target, v_cell_info);
  }

  /**
   *  Evaluate a Nektar++ field at particle locations using the provided
   *  quadrature point values and Bary Interpolation.
   *
   *  @param particle_group ParticleGroup containing the particles.
   *  @param sym Symbol that corresponds to a ParticleDat in the ParticleGroup.
   *  @param component Component in the ParticleDat in which to place the
   * output.
   *  @param global_physvals Field values at quadrature points to evaluate.
   */
  template <typename U, typename V>
  inline void evaluate(ParticleGroupSharedPtr particle_group, Sym<U> sym,
                       const int component, const V &global_physvals) {

    // copy the quadrature point values over to the device
    const int num_global_physvals = global_physvals.size();
    const int device_phys_vals_size = pad_to_vector_length(num_global_physvals);
    this->dh_global_physvals.realloc_no_copy(device_phys_vals_size);
    for (int px = 0; px < num_global_physvals; px++) {
      this->dh_global_physvals.h_buffer.ptr[px] = global_physvals[px];
    }
    for (int px = num_global_physvals; px < device_phys_vals_size; px++) {
      this->dh_global_physvals.h_buffer.ptr[px] = 0.0;
    }
    this->dh_global_physvals.host_to_device();
    const auto k_global_physvals = this->dh_global_physvals.d_buffer.ptr;

    // output and particle position dats
    auto k_output = (*particle_group)[sym]->cell_dat.device_ptr();
    const auto k_ref_positions =
        (*particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]
            ->cell_dat.device_ptr();
    const int k_component = component;

    // values required to evaluate the field
    auto k_cell_info = this->d_cell_info->ptr;

    // iteration set specification for the particle loop
    auto mpi_rank_dat = particle_group->mpi_rank_dat;
    ParticleLoopImplementation::ParticleLoopBlockIterationSet ish{mpi_rank_dat};
    const std::size_t local_size =
        this->sycl_target->parameters
            ->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
            ->value;
    const std::size_t local_num_reals =
        static_cast<std::size_t>(this->ndim * this->max_num_phys);
    const std::size_t num_bytes_local = local_num_reals * sizeof(REAL);

    EventStack es;
    auto is = ish.get_all_cells(local_size, num_bytes_local);

    ProfileRegion pr("BaryEvaluateBase", "evaluate_2d");
    for (auto &blockx : is) {
      const auto block_device = blockx.block_device;
      const std::size_t local_size = blockx.local_size;
      es.push(sycl_target->queue.submit([&](sycl::handler &cgh) {
        // Allocate local memory to compute the divides.
        sycl::local_accessor<REAL, 1> local_mem(
            sycl::range<1>(local_num_reals * local_size), cgh);

        auto lambda_inner = [=](auto idx_local, auto cell, auto layer) {
          // offset by the local index for the striding to work
          REAL *div_space0 = &local_mem[idx_local];
          const auto cell_info = k_cell_info[cell];

          const REAL xi0 = k_ref_positions[cell][0][layer];
          const REAL xi1 = k_ref_positions[cell][1][layer];
          // If this cell is a triangle then we need to map to the
          // collapsed coordinates.
          REAL coord0, coord1;

          GeometryInterface::loc_coord_to_loc_collapsed_2d(
              cell_info.shape_type_int, xi0, xi1, &coord0, &coord1);

          const auto num_phys0 = cell_info.num_phys[0];
          const auto num_phys1 = cell_info.num_phys[1];
          const auto z0 = cell_info.d_z[0];
          const auto z1 = cell_info.d_z[1];
          const auto bw0 = cell_info.d_bw[0];
          const auto bw1 = cell_info.d_bw[1];

          // Get pointer to the start of the quadrature point values for
          // this cell
          const auto physvals = &k_global_physvals[cell_info.phys_offset];

          const REAL evaluation =
              Bary::evaluate_2d(coord0, coord1, num_phys0, num_phys1, physvals,
                                div_space0, z0, z1, bw0, bw1, local_size);

          k_output[cell][k_component][layer] = evaluation;
        };

        if (blockx.layer_bounds_check_required) {
          cgh.parallel_for<>(
              blockx.loop_iteration_set, [=](sycl::nd_item<2> idx) {
                const int idx_local = idx.get_local_id(1);
                std::size_t cell;
                std::size_t layer;
                block_device.get_cell_layer(idx, &cell, &layer);
                if (block_device.work_item_required(cell, layer)) {
                  lambda_inner(idx_local, cell, layer);
                }
              });
        } else {
          cgh.parallel_for<>(blockx.loop_iteration_set,
                             [=](sycl::nd_item<2> idx) {
                               const int idx_local = idx.get_local_id(1);
                               std::size_t cell;
                               std::size_t layer;
                               block_device.get_cell_layer(idx, &cell, &layer);
                               lambda_inner(idx_local, cell, layer);
                             });
        }
      }));
    }
    const auto nphys = this->max_num_phys;
    const auto npart = particle_group->get_npart_local();
    const auto nflop_prepare = this->ndim * nphys * 5;
    const auto nflop_loop = nphys * nphys * 3;
    pr.num_flops = (nflop_loop + nflop_prepare) * npart;
    es.wait();
    pr.end();
    this->sycl_target->profile_map.add_region(pr);
  }
};

} // namespace NESO

#endif
