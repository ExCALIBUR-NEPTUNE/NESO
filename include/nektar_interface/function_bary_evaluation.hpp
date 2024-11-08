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
  BufferDevice<NekDouble> d_global_physvals;

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

  template <typename U>
  static inline void
  dispatch_2d(SYCLTargetSharedPtr sycl_target, EventStack &es,
              const std::size_t num_functions, const int k_max_num_phys,
              const NekDouble *const RESTRICT k_global_physvals_interlaced,
              const CellInfo *const RESTRICT k_cell_info,
              ParticleDatSharedPtr<INT> mpi_rank_dat,
              ParticleDatSharedPtr<REAL> ref_positions_dat, U ****k_syms_ptrs,
              int *k_components) {
    constexpr int ndim = 2;
    ParticleLoopImplementation::ParticleLoopBlockIterationSet ish{mpi_rank_dat};
    const std::size_t local_size =
        sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
            ->value;
    const std::size_t local_num_reals =
        static_cast<std::size_t>(ndim * k_max_num_phys) + num_functions;
    const std::size_t num_bytes_local = local_num_reals * sizeof(REAL);
    auto is = ish.get_all_cells(local_size, num_bytes_local);
    const auto k_ref_positions = ref_positions_dat->cell_dat.device_ptr();

    for (auto &blockx : is) {
      const auto block_device = blockx.block_device;
      const std::size_t local_size = blockx.local_size;
      es.push(sycl_target->queue.submit([&](sycl::handler &cgh) {
        // Allocate local memory to compute the divides.
        sycl::local_accessor<REAL, 1> local_mem(
            sycl::range<1>(local_num_reals * local_size), cgh);

        cgh.parallel_for<>(
            blockx.loop_iteration_set, [=](sycl::nd_item<2> idx) {
              const int idx_local = idx.get_local_id(1);
              std::size_t cell;
              std::size_t layer;
              block_device.get_cell_layer(idx, &cell, &layer);
              if (block_device.work_item_required(cell, layer)) {
                // offset by the local index for the striding to work
                REAL *evaluations = &local_mem[0] + idx_local * num_functions;
                REAL *div_start = &local_mem[local_size * num_functions];
                REAL *div_space0 = div_start + idx_local;
                REAL *div_space1 = div_space0 + k_max_num_phys * local_size;

                const auto cell_info = k_cell_info[cell];

                const REAL xi0 = k_ref_positions[cell][0][layer];
                const REAL xi1 = k_ref_positions[cell][1][layer];
                // If this cell is a triangle then we need to map to the
                // collapsed coordinates.
                REAL eta0, eta1;

                GeometryInterface::loc_coord_to_loc_collapsed_2d(
                    cell_info.shape_type_int, xi0, xi1, &eta0, &eta1);

                const auto num_phys0 = cell_info.num_phys[0];
                const auto num_phys1 = cell_info.num_phys[1];
                const auto z0 = cell_info.d_z[0];
                const auto z1 = cell_info.d_z[1];
                const auto bw0 = cell_info.d_bw[0];
                const auto bw1 = cell_info.d_bw[1];

                Bary::preprocess_weights(num_phys0, eta0, z0, bw0, div_space0,
                                         local_size);
                Bary::preprocess_weights(num_phys1, eta1, z1, bw1, div_space1,
                                         local_size);

                // Get pointer to the start of the quadrature point values for
                // this cell
                const auto physvals =
                    &k_global_physvals_interlaced[cell_info.phys_offset *
                                                  num_functions];

                Bary::compute_dir_10_interlaced(
                    num_functions, num_phys0, num_phys1, physvals, div_space0,
                    div_space1, evaluations, local_size);

                for (std::size_t fx = 0; fx < num_functions; fx++) {
                  auto ptr = k_syms_ptrs[fx];
                  auto component = k_components[fx];
                  ptr[cell][component][layer] = evaluations[fx];
                }
              }
            });
      }));
    }
  }

  template <typename U>
  static inline void
  dispatch_3d(SYCLTargetSharedPtr sycl_target, EventStack &es,
              const std::size_t num_functions, const int k_max_num_phys,
              const NekDouble *const RESTRICT k_global_physvals_interlaced,
              const CellInfo *const RESTRICT k_cell_info,
              ParticleDatSharedPtr<INT> mpi_rank_dat,
              ParticleDatSharedPtr<REAL> ref_positions_dat, U ****k_syms_ptrs,
              int *k_components) {
    constexpr int ndim = 3;
    ParticleLoopImplementation::ParticleLoopBlockIterationSet ish{mpi_rank_dat};
    const std::size_t local_size =
        sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
            ->value;
    const std::size_t local_num_reals =
        static_cast<std::size_t>(ndim * k_max_num_phys) + num_functions;
    const std::size_t num_bytes_local = local_num_reals * sizeof(REAL);
    auto is = ish.get_all_cells(local_size, num_bytes_local);
    const auto k_ref_positions = ref_positions_dat->cell_dat.device_ptr();

    for (auto &blockx : is) {
      const auto block_device = blockx.block_device;
      const std::size_t local_size = blockx.local_size;
      es.push(sycl_target->queue.submit([&](sycl::handler &cgh) {
        // Allocate local memory to compute the divides.
        sycl::local_accessor<REAL, 1> local_mem(
            sycl::range<1>(local_num_reals * local_size), cgh);

        cgh.parallel_for<>(
            blockx.loop_iteration_set, [=](sycl::nd_item<2> idx) {
              const int idx_local = idx.get_local_id(1);
              std::size_t cell;
              std::size_t layer;
              block_device.get_cell_layer(idx, &cell, &layer);
              if (block_device.work_item_required(cell, layer)) {
                // offset by the local index for the striding to work
                REAL *evaluations = &local_mem[0] + idx_local * num_functions;
                REAL *div_start = &local_mem[local_size * num_functions];
                REAL *div_space0 = div_start + idx_local;
                REAL *div_space1 = div_space0 + k_max_num_phys * local_size;
                REAL *div_space2 = div_space1 + k_max_num_phys * local_size;

                const auto cell_info = k_cell_info[cell];

                const REAL xi0 = k_ref_positions[cell][0][layer];
                const REAL xi1 = k_ref_positions[cell][1][layer];
                const REAL xi2 = k_ref_positions[cell][2][layer];
                REAL eta0, eta1, eta2;

                GeometryInterface::loc_coord_to_loc_collapsed_3d(
                    cell_info.shape_type_int, xi0, xi1, xi2, &eta0, &eta1,
                    &eta2);

                const auto num_phys0 = cell_info.num_phys[0];
                const auto num_phys1 = cell_info.num_phys[1];
                const auto num_phys2 = cell_info.num_phys[2];
                const auto z0 = cell_info.d_z[0];
                const auto z1 = cell_info.d_z[1];
                const auto z2 = cell_info.d_z[2];
                const auto bw0 = cell_info.d_bw[0];
                const auto bw1 = cell_info.d_bw[1];
                const auto bw2 = cell_info.d_bw[2];

                Bary::preprocess_weights(num_phys0, eta0, z0, bw0, div_space0,
                                         local_size);
                Bary::preprocess_weights(num_phys1, eta1, z1, bw1, div_space1,
                                         local_size);
                Bary::preprocess_weights(num_phys2, eta2, z2, bw2, div_space2,
                                         local_size);

                // Get pointer to the start of the quadrature point values for
                // this cell
                const auto physvals =
                    &k_global_physvals_interlaced[cell_info.phys_offset *
                                                  num_functions];

                Bary::compute_dir_210_interlaced(
                    num_functions, num_phys0, num_phys1, num_phys2, physvals,
                    div_space0, div_space1, div_space2, evaluations,
                    local_size);

                for (std::size_t fx = 0; fx < num_functions; fx++) {
                  auto ptr = k_syms_ptrs[fx];
                  auto component = k_components[fx];
                  ptr[cell][component][layer] = evaluations[fx];
                }
              }
            });
      }));
    }
  }

  template <typename U>
  inline void
  evaluate_inner(ParticleGroupSharedPtr particle_group,
                 std::vector<Sym<U>> syms, const std::vector<int> components,
                 std::vector<Array<OneD, NekDouble> *> &global_physvals) {

    EventStack es;
    NESOASSERT(syms.size() == components.size(), "Input size missmatch");
    NESOASSERT(global_physvals.size() == components.size(),
               "Input size missmatch");
    const std::size_t num_functions = static_cast<int>(syms.size());
    const std::size_t num_physvals_per_function = global_physvals.at(0)->size();

    // copy the quadrature point values over to the device
    const std::size_t num_global_physvals =
        num_functions * num_physvals_per_function;

    const std::size_t factor = (num_functions > 1) ? 2 : 1;
    this->d_global_physvals.realloc_no_copy(factor * num_global_physvals);
    NekDouble *k_global_physvals = this->d_global_physvals.ptr;
    NekDouble *k_global_physvals_interlaced =
        (num_functions > 1) ? k_global_physvals + num_global_physvals
                            : k_global_physvals;

    static_assert(
        std::is_same<NekDouble, REAL>::value == true,
        "This implementation assumes that NekDouble and REAL are the same.");
    for (std::size_t fx = 0; fx < num_functions; fx++) {
      NESOASSERT(num_physvals_per_function == global_physvals.at(fx)->size(),
                 "Missmatch in number of physvals between functions.");

      es.push(this->sycl_target->queue.memcpy(
          k_global_physvals + fx * num_physvals_per_function,
          global_physvals.at(fx)->data(),
          num_physvals_per_function * sizeof(NekDouble)));
    }
    // wait for the copies
    es.wait();

    // interlaced the values for the bary evaluation function
    if (num_functions > 1) {
      matrix_transpose(this->sycl_target, num_functions,
                       num_physvals_per_function, k_global_physvals,
                       k_global_physvals_interlaced)
          .wait_and_throw();
    }

    std::vector<U ***> h_sym_ptrs(num_functions);
    for (std::size_t fx = 0; fx < num_functions; fx++) {
      h_sym_ptrs.at(fx) =
          particle_group->get_dat(syms.at(fx))->cell_dat.device_ptr();
    }
    BufferDevice<U ***> d_syms_ptrs(this->sycl_target, h_sym_ptrs);
    BufferDevice<int> d_components(this->sycl_target, components);

    ProfileRegion pr("BaryEvaluateBase", "evaluate_" +
                                             std::to_string(this->ndim) + "d_" +
                                             std::to_string(num_functions));

    if (this->ndim == 2) {
      this->dispatch_2d(
          this->sycl_target, es, num_functions, this->max_num_phys,
          k_global_physvals_interlaced, this->d_cell_info->ptr,
          particle_group->mpi_rank_dat,
          particle_group->get_dat(Sym<REAL>("NESO_REFERENCE_POSITIONS")),
          d_syms_ptrs.ptr, d_components.ptr);
    } else {
      this->dispatch_3d(
          this->sycl_target, es, num_functions, this->max_num_phys,
          k_global_physvals_interlaced, this->d_cell_info->ptr,
          particle_group->mpi_rank_dat,
          particle_group->get_dat(Sym<REAL>("NESO_REFERENCE_POSITIONS")),
          d_syms_ptrs.ptr, d_components.ptr);
    }

    const auto nphys = this->max_num_phys;
    const auto npart = particle_group->get_npart_local();
    const auto nflop_prepare = this->ndim * nphys * 5;
    const auto nflop_loop = (this->ndim == 2)
                                ? nphys * nphys * 3
                                : nphys * nphys + nphys * nphys * nphys * 3;
    pr.num_flops = (nflop_loop + nflop_prepare) * npart;
    pr.num_bytes = sizeof(REAL) * (npart * ((this->ndim + num_functions)) +
                                   num_global_physvals);
    // wait for the loop to complete
    es.wait();
    pr.end();
    this->sycl_target->profile_map.add_region(pr);
  }

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
        d_global_physvals(sycl_target, 1) {

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
   *  Evaluate Nektar++ fields at particle locations using the provided
   *  quadrature point values and Bary Interpolation.
   *
   *  @param particle_group ParticleGroup containing the particles.
   *  @param syms Vector of Syms in which to place evaluations.
   *  @param components Vector of components in which to place evaluations.
   *  @param global_physvals Phys values for each function to evaluate.
   */
  template <typename U>
  inline void evaluate(ParticleGroupSharedPtr particle_group,
                       std::vector<Sym<U>> syms,
                       const std::vector<int> components,
                       std::vector<Array<OneD, NekDouble> *> &global_physvals) {
    if ((this->ndim == 2) || (this->ndim == 3)) {
      return this->evaluate_inner(particle_group, syms, components,
                                  global_physvals);
    } else {
      NESOASSERT(false, "Not implemented in this number of dimensions.");
    }
  }
};

} // namespace NESO

#endif
