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

  template <typename GROUP_TYPE, typename U>
  static inline void
  dispatch_2d(std::shared_ptr<GROUP_TYPE> particle_sub_group,
              const std::size_t num_functions, const int k_max_num_phys,
              const NekDouble *const RESTRICT k_global_physvals_interlaced,
              const CellInfo *const RESTRICT k_cell_info,
              ParticleDatImplGetConstT<REAL> k_ref_positions,
              ParticleDatImplGetT<U> *k_syms_ptrs, int *k_components) {
    constexpr int ndim = 2;
    const std::size_t local_num_reals =
        static_cast<std::size_t>(ndim * k_max_num_phys) + num_functions;

    particle_loop(
        "BaryEvaluateBase::dispatch_2d", particle_sub_group,
        [=](auto INDEX, auto LOCAL_MEMORY) {
          const auto cell = INDEX.cell;
          const auto layer = INDEX.layer;
          const std::size_t stride = LOCAL_MEMORY.stride;
          REAL *evaluations = LOCAL_MEMORY.data();
          REAL *div_space0 = &LOCAL_MEMORY.at(num_functions);
          REAL *div_space1 = &LOCAL_MEMORY.at(num_functions + k_max_num_phys);

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
                                   stride);
          Bary::preprocess_weights(num_phys1, eta1, z1, bw1, div_space1,
                                   stride);

          // Get pointer to the start of the quadrature point values for
          // this cell
          const auto physvals =
              &k_global_physvals_interlaced[cell_info.phys_offset *
                                            num_functions];

          Bary::compute_dir_10_interlaced(num_functions, num_phys0, num_phys1,
                                          physvals, div_space0, div_space1,
                                          evaluations, stride, stride);

          for (std::size_t fx = 0; fx < num_functions; fx++) {
            auto ptr = k_syms_ptrs[fx];
            auto component = k_components[fx];
            ptr[cell][component][layer] = LOCAL_MEMORY.at(fx);
          }
        },
        Access::read(ParticleLoopIndex{}),
        Access::write(
            std::make_shared<LocalMemoryInterlaced<REAL>>(local_num_reals)))
        ->execute();
  }

  template <typename GROUP_TYPE, typename U>
  static inline void
  dispatch_3d(std::shared_ptr<GROUP_TYPE> particle_sub_group,
              const std::size_t num_functions, const int k_max_num_phys,
              const NekDouble *const RESTRICT k_global_physvals_interlaced,
              const CellInfo *const RESTRICT k_cell_info,
              ParticleDatImplGetConstT<REAL> k_ref_positions,
              ParticleDatImplGetT<U> *k_syms_ptrs, int *k_components) {
    constexpr int ndim = 3;
    const std::size_t local_num_reals =
        static_cast<std::size_t>(ndim * k_max_num_phys) + num_functions;

    particle_loop(
        "BaryEvaluateBase::dispatch_3d", particle_sub_group,
        [=](auto INDEX, auto LOCAL_MEMORY) {
          const auto cell = INDEX.cell;
          const auto layer = INDEX.layer;
          const std::size_t stride = LOCAL_MEMORY.stride;
          REAL *evaluations = LOCAL_MEMORY.data();
          REAL *div_space0 = &LOCAL_MEMORY.at(num_functions);
          REAL *div_space1 = &LOCAL_MEMORY.at(num_functions + k_max_num_phys);
          REAL *div_space2 =
              &LOCAL_MEMORY.at(num_functions + 2 * k_max_num_phys);

          const auto cell_info = k_cell_info[cell];

          const REAL xi0 = k_ref_positions[cell][0][layer];
          const REAL xi1 = k_ref_positions[cell][1][layer];
          const REAL xi2 = k_ref_positions[cell][2][layer];
          REAL eta0, eta1, eta2;

          GeometryInterface::loc_coord_to_loc_collapsed_3d(
              cell_info.shape_type_int, xi0, xi1, xi2, &eta0, &eta1, &eta2);

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
                                   stride);
          Bary::preprocess_weights(num_phys1, eta1, z1, bw1, div_space1,
                                   stride);
          Bary::preprocess_weights(num_phys2, eta2, z2, bw2, div_space2,
                                   stride);

          // Get pointer to the start of the quadrature point values for
          // this cell
          const auto physvals =
              &k_global_physvals_interlaced[cell_info.phys_offset *
                                            num_functions];

          Bary::compute_dir_210_interlaced(
              num_functions, num_phys0, num_phys1, num_phys2, physvals,
              div_space0, div_space1, div_space2, evaluations, stride, stride);

          for (std::size_t fx = 0; fx < num_functions; fx++) {
            auto ptr = k_syms_ptrs[fx];
            auto component = k_components[fx];
            ptr[cell][component][layer] = LOCAL_MEMORY.at(fx);
          }
        },
        Access::read(ParticleLoopIndex{}),
        Access::write(
            std::make_shared<LocalMemoryInterlaced<REAL>>(local_num_reals)))
        ->execute();
  }

  template <typename U>
  static inline void
  dispatch_3d_cpu(SYCLTargetSharedPtr sycl_target, EventStack &es,
                  const std::size_t num_functions, const int k_max_num_phys,
                  const NekDouble *const RESTRICT k_global_physvals_interlaced,
                  const CellInfo *const RESTRICT k_cell_info,
                  ParticleDatSharedPtr<INT> mpi_rank_dat,
                  ParticleDatImplGetConstT<REAL> k_ref_positions,
                  ParticleDatImplGetT<U> *k_syms_ptrs, int *k_components) {
    constexpr int ndim = 3;
    ParticleLoopImplementation::ParticleLoopBlockIterationSet ish{mpi_rank_dat};
    const std::size_t local_size =
        sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
            ->value;
    const std::size_t nbin =
        sycl_target->parameters->template get<SizeTParameter>("LOOP_NBIN")
            ->value;
    const std::size_t local_num_reals =
        static_cast<std::size_t>(ndim * k_max_num_phys) + num_functions;
    const std::size_t num_bytes_local = local_num_reals * sizeof(REAL);
    auto is = ish.get_all_cells(nbin, local_size, num_bytes_local,
                                NESO_VECTOR_BLOCK_SIZE);
    for (auto &blockx : is) {
      const auto block_device = blockx.block_device;
      const std::size_t local_size = blockx.local_size;
      es.push(sycl_target->queue.submit([&](sycl::handler &cgh) {
        const std::size_t local_mem_stride =
            local_num_reals * NESO_VECTOR_BLOCK_SIZE;
        // Allocate local memory to compute the divides.
        sycl::local_accessor<REAL, 1> local_mem(
            sycl::range<1>(local_size * local_mem_stride), cgh);

        cgh.parallel_for<>(
            blockx.loop_iteration_set, [=](sycl::nd_item<2> idx) {
              const int idx_local = idx.get_local_id(1);
              std::size_t cell;
              std::size_t block;
              block_device.stride_get_cell_block(idx, &cell, &block);
              if (block_device.stride_work_item_required(cell, block)) {
                const std::size_t particle_start =
                    block * NESO_VECTOR_BLOCK_SIZE;
                const std::size_t local_bound =
                    block_device.stride_local_index_bound(cell, block);

                REAL *evaluations =
                    &local_mem[0] + idx_local * local_mem_stride;
                const std::size_t div_space_per_work_item =
                    NESO_VECTOR_BLOCK_SIZE * k_max_num_phys;
                REAL *div_space0 =
                    evaluations + NESO_VECTOR_BLOCK_SIZE * num_functions;
                REAL *div_space1 = div_space0 + div_space_per_work_item;
                REAL *div_space2 = div_space1 + div_space_per_work_item;

                const auto cell_info = k_cell_info[cell];
                const auto num_phys0 = cell_info.num_phys[0];
                const auto num_phys1 = cell_info.num_phys[1];
                const auto num_phys2 = cell_info.num_phys[2];
                const auto z0 = cell_info.d_z[0];
                const auto z1 = cell_info.d_z[1];
                const auto z2 = cell_info.d_z[2];
                const auto bw0 = cell_info.d_bw[0];
                const auto bw1 = cell_info.d_bw[1];
                const auto bw2 = cell_info.d_bw[2];
                // Get pointer to the start of the quadrature point values for
                // this cell
                const auto physvals =
                    &k_global_physvals_interlaced[cell_info.phys_offset *
                                                  num_functions];

                REAL xi0[NESO_VECTOR_BLOCK_SIZE];
                REAL xi1[NESO_VECTOR_BLOCK_SIZE];
                REAL xi2[NESO_VECTOR_BLOCK_SIZE];
                REAL eta0[NESO_VECTOR_BLOCK_SIZE];
                REAL eta1[NESO_VECTOR_BLOCK_SIZE];
                REAL eta2[NESO_VECTOR_BLOCK_SIZE];

                for (std::size_t blockx = 0; blockx < local_bound; blockx++) {
                  const std::size_t px = particle_start + blockx;
                  xi0[blockx] = k_ref_positions[cell][0][px];
                  xi1[blockx] = k_ref_positions[cell][1][px];
                  xi2[blockx] = k_ref_positions[cell][2][px];
                }

                for (std::size_t blockx = 0; blockx < NESO_VECTOR_BLOCK_SIZE;
                     blockx++) {
                  GeometryInterface::loc_coord_to_loc_collapsed_3d(
                      cell_info.shape_type_int, xi0[blockx], xi1[blockx],
                      xi2[blockx], eta0 + blockx, eta1 + blockx, eta2 + blockx);
                }

                Bary::preprocess_weights_block<NESO_VECTOR_BLOCK_SIZE>(
                    num_phys0, eta0, z0, bw0, div_space0);
                Bary::preprocess_weights_block<NESO_VECTOR_BLOCK_SIZE>(
                    num_phys1, eta1, z1, bw1, div_space1);
                Bary::preprocess_weights_block<NESO_VECTOR_BLOCK_SIZE>(
                    num_phys2, eta2, z2, bw2, div_space2);

                Bary::compute_dir_210_interlaced_block<NESO_VECTOR_BLOCK_SIZE>(
                    num_functions, num_phys0, num_phys1, num_phys2, physvals,
                    div_space0, div_space1, div_space2, evaluations);

                for (std::size_t fx = 0; fx < num_functions; fx++) {
                  for (std::size_t blockx = 0; blockx < local_bound; blockx++) {
                    const std::size_t px = particle_start + blockx;
                    auto ptr = k_syms_ptrs[fx];
                    auto component = k_components[fx];
                    ptr[cell][component][px] =
                        evaluations[fx * NESO_VECTOR_BLOCK_SIZE + blockx];
                  }
                }
              }
            });
      }));
    }
  }

  template <typename GROUP_TYPE, typename U>
  inline void
  evaluate_inner(std::shared_ptr<GROUP_TYPE> particle_sub_group,
                 std::vector<Sym<U>> syms, const std::vector<int> components,
                 std::vector<Array<OneD, NekDouble> *> &global_physvals) {

    auto particle_group = get_particle_group(particle_sub_group);

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

    std::vector<ParticleDatImplGetT<U>> h_sym_ptrs(num_functions);
    for (std::size_t fx = 0; fx < num_functions; fx++) {
      h_sym_ptrs.at(fx) = Access::direct_get(
          Access::write(particle_group->get_dat(syms.at(fx))));
    }
    BufferDevice<ParticleDatImplGetT<U>> d_syms_ptrs(this->sycl_target,
                                                     h_sym_ptrs);
    BufferDevice<int> d_components(this->sycl_target, components);

    auto k_ref_positions = Access::direct_get(Access::read(
        particle_group->get_dat(Sym<REAL>("NESO_REFERENCE_POSITIONS"))));

    ProfileRegion pr("BaryEvaluateBase", "evaluate_" +
                                             std::to_string(this->ndim) + "d_" +
                                             std::to_string(num_functions));
    if (this->ndim == 2) {
      this->dispatch_2d(particle_sub_group, num_functions, this->max_num_phys,
                        k_global_physvals_interlaced, this->d_cell_info->ptr,
                        k_ref_positions, d_syms_ptrs.ptr, d_components.ptr);
    } else {
      if (this->sycl_target->device.is_gpu() ||
          is_particle_sub_group(particle_sub_group)) {
        this->dispatch_3d(particle_sub_group, num_functions, this->max_num_phys,
                          k_global_physvals_interlaced, this->d_cell_info->ptr,
                          k_ref_positions, d_syms_ptrs.ptr, d_components.ptr);
      } else {
        this->dispatch_3d_cpu(this->sycl_target, es, num_functions,
                              this->max_num_phys, k_global_physvals_interlaced,
                              this->d_cell_info->ptr,
                              particle_group->mpi_rank_dat, k_ref_positions,
                              d_syms_ptrs.ptr, d_components.ptr);
      }
    }

    const auto nphys = this->max_num_phys;
    const auto npart = particle_sub_group->get_npart_local();
    const auto nflop_prepare = this->ndim * nphys * 5;
    const auto nflop_loop =
        (this->ndim == 2)
            ? nphys * nphys * (1 + num_functions * 2)
            : nphys * nphys + nphys * nphys * nphys * (1 + num_functions * 2);
    pr.num_flops = (nflop_loop + nflop_prepare) * npart;
    pr.num_bytes = sizeof(REAL) * (npart * ((this->ndim + num_functions)) +
                                   num_global_physvals);
    // wait for the loop to complete
    es.wait();
    for (std::size_t fx = 0; fx < num_functions; fx++) {
      Access::direct_restore(
          Access::write(particle_group->get_dat(syms.at(fx))),
          h_sym_ptrs.at(fx));
    }
    Access::direct_restore(Access::read(particle_group->get_dat(
                               Sym<REAL>("NESO_REFERENCE_POSITIONS"))),
                           k_ref_positions);
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

    // Temporary map for testing values are consistent between cells.
    std::map<MapKey, std::vector<REAL>> map_test_coeffs;

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
      std::size_t size_sum = 0;
      for (int dimx = 0; dimx < this->ndim; dimx++) {
        const std::size_t tmp_size =
            static_cast<std::size_t>(expansion->GetBase()[dimx]->GetZ().size());
        std::get<1>(key)[dimx] = tmp_size;
        size_sum += tmp_size;
      }
      if (this->map_cells_to_info.count(key) == 0) {
        map_test_coeffs[key].reserve(2 * size_sum);

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
            map_test_coeffs.at(key).push_back(z[cx]);
          }
          for (int cx = 0; cx < size; cx++) {
            tmp_reals.at(cx + size) = bw[cx];
            map_test_coeffs.at(key).push_back(bw[cx]);
          }
          auto ptr = std::make_shared<BufferDevice<REAL>>(this->sycl_target,
                                                          tmp_reals);
          auto d_ptr = ptr->ptr;
          this->stack_ptrs.push(ptr);

          cell_info.d_z[dimx] = d_ptr;
          cell_info.d_bw[dimx] = d_ptr + size;
        }

        this->map_cells_to_info[key] = cell_info;
      } else {
        // Test that the held values that we reuse are actually the same.
        std::vector<REAL> to_test;
        to_test.reserve(2 * size_sum);
        for (int dimx = 0; dimx < this->ndim; dimx++) {
          auto base = expansion->GetBase();
          const auto &z = base[dimx]->GetZ();
          const auto &bw = base[dimx]->GetBaryWeights();
          NESOASSERT(z.size() == bw.size(),
                     "Expected these two sizes to match.");
          const auto size = z.size();
          for (int cx = 0; cx < size; cx++) {
            to_test.push_back(z[cx]);
          }
          for (int cx = 0; cx < size; cx++) {
            to_test.push_back(bw[cx]);
          }
        }
        NESOASSERT(to_test.size() == map_test_coeffs.at(key).size(),
                   "Size missmatch in coeff checking.");
        const std::size_t size = to_test.size();
        auto lambda_rel_err = [](const REAL a, const REAL b) {
          const REAL err_abs = std::abs(a - b);
          const REAL abs_a = std::abs(a);
          const REAL err_rel = abs_a > 0.0 ? err_abs / abs_a : err_abs;
          return std::min(err_rel, err_abs);
        };

        for (std::size_t ix = 0; ix < size; ix++) {
          NESOASSERT(lambda_rel_err(to_test.at(ix),
                                    map_test_coeffs.at(key).at(ix)) < 1.0e-10,
                     "Big missmatch in coefficients detected. Please raise an "
                     "issue on the git repository.");
        }
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
   *  @param particle_sub_group ParticleGroup or ParticleSubGroup containing the
   * particles.
   *  @param syms Vector of Syms in which to place evaluations.
   *  @param components Vector of components in which to place evaluations.
   *  @param global_physvals Phys values for each function to evaluate.
   */
  template <typename GROUP_TYPE, typename U>
  inline void evaluate(std::shared_ptr<GROUP_TYPE> particle_sub_group,
                       std::vector<Sym<U>> syms,
                       const std::vector<int> components,
                       std::vector<Array<OneD, NekDouble> *> &global_physvals) {
    if ((this->ndim == 2) || (this->ndim == 3)) {
      return this->evaluate_inner(particle_sub_group, syms, components,
                                  global_physvals);
    } else {
      NESOASSERT(false, "Not implemented in this number of dimensions.");
    }
  }
};

} // namespace NESO

#endif
