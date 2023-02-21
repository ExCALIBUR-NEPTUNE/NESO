#ifndef __FUNCTION_COUPLING_BASE_H_
#define __FUNCTION_COUPLING_BASE_H_
#include "particle_interface.hpp"
#include <map>
#include <memory>
#include <neso_particles.hpp>

#include <LocalRegions/QuadExp.h>
#include <LocalRegions/TriExp.h>
#include <StdRegions/StdExpansion2D.h>

using namespace NESO::Particles;
using namespace Nektar::LocalRegions;
using namespace Nektar::StdRegions;

namespace NESO {

/**
 *  Class to provide a common method that builds the map from geometry ids to
 *  expansion ids.
 */
class GeomToExpansionBuilder {

protected:
  /**
   *  Build the map from geometry ids to expansion ids for the expansion.
   *  Nektar++ Expansions hold a reference to the geometry element they are
   *  defined over. This function assumes that map is injective and builds the
   *  inverse map from geometry id to expansion id.
   *
   *  @param field Nektar++ Expansion, e.g. ContField or DisContField.
   *  @param geom_to_exp Output map to build.
   */
  template <typename T>
  static inline void
  build_geom_to_expansion_map(std::shared_ptr<T> field,
                              std::map<int, int> &geom_to_exp) {
    // build the map from geometry ids to expansion ids
    auto expansions = field->GetExp();
    const int num_expansions = (*expansions).size();
    for (int ex = 0; ex < num_expansions; ex++) {
      auto exp = (*expansions)[ex];
      // The indexing in Nektar++ source suggests that ex is the important
      // index if these do not match in future.
      NESOASSERT(ex == exp->GetElmtId(),
                 "expected expansion id to match element id?");
      int geom_gid = exp->GetGeom()->GetGlobalID();
      geom_to_exp[geom_gid] = ex;
    }
  };
};

template <typename T> class BaryEvaluateBase : GeomToExpansionBuilder {
protected:
  std::shared_ptr<T> field;
  ParticleMeshInterfaceSharedPtr mesh;
  CellIDTranslationSharedPtr cell_id_translation;
  SYCLTargetSharedPtr sycl_target;

  BufferDeviceHost<int> dh_phys_offsets;
  BufferDeviceHost<int> dh_phys_num0;
  BufferDeviceHost<int> dh_phys_num1;
  BufferDeviceHost<NekDouble> dh_global_physvals;

  inline void assemble_data(const int offset,
                            StdExpansion2DSharedPtr expansion) {
    auto base = expansion->GetBase();

    double *z_ptr = this->dh_z.h_buffer.ptr + offset * stride_expansion_type;
    double *bw_ptr = this->dh_bw.h_buffer.ptr + offset * stride_expansion_type;

    for (int dimx = 0; dimx < 2; dimx++) {
      const auto &z = base[dimx]->GetZ();
      const auto &bw = base[dimx]->GetBaryWeights();
      NESOASSERT(z.size() == bw.size(), "Expected these two sizes to match.");
      const int size = z.size();
      for (int cx = 0; cx < size; cx++) {
        z_ptr[dimx * stride_base + cx] = z[cx];
        bw_ptr[dimx * stride_base + cx] = bw[cx];
      }
    }
  }

public:
  /// Disable (implicit) copies.
  BaryEvaluateBase(const BaryEvaluateBase &st) = delete;
  /// Disable (implicit) copies.
  BaryEvaluateBase &operator=(BaryEvaluateBase const &a) = delete;

  /// Stride between z and weight values in each dimension for all expansion
  /// types.
  int stride_base;
  /// Stride between expansion types for z and weight values.
  int stride_expansion_type;

  /// The GetZ values for different expansion types.
  BufferDeviceHost<double> dh_z;

  /// The GetBaryWeights for different expansion types.
  BufferDeviceHost<double> dh_bw;

  BaryEvaluateBase(std::shared_ptr<T> field,
                   ParticleMeshInterfaceSharedPtr mesh,
                   CellIDTranslationSharedPtr cell_id_translation)
      : field(field), mesh(mesh), cell_id_translation(cell_id_translation),
        sycl_target(cell_id_translation->sycl_target), dh_z(sycl_target, 1),
        dh_bw(sycl_target, 1), dh_phys_offsets(sycl_target, 1),
        dh_phys_num0(sycl_target, 1), dh_phys_num1(sycl_target, 1),
        dh_global_physvals(sycl_target, 1) {

    // build the map from geometry ids to expansion ids
    std::map<int, int> geom_to_exp;
    build_geom_to_expansion_map(this->field, geom_to_exp);

    TriExpSharedPtr tri_exp{nullptr};
    QuadExpSharedPtr quad_exp{nullptr};
    auto geom_type_lookup =
        this->cell_id_translation->dh_map_to_geom_type.h_buffer.ptr;

    const int index_tri_geom = this->cell_id_translation->index_tri_geom;
    const int index_quad_geom = this->cell_id_translation->index_quad_geom;

    const int neso_cell_count = mesh->get_cell_count();
    this->dh_phys_offsets.realloc_no_copy(neso_cell_count);
    this->dh_phys_num0.realloc_no_copy(neso_cell_count);
    this->dh_phys_num1.realloc_no_copy(neso_cell_count);

    for (int neso_cellx = 0; neso_cellx < neso_cell_count; neso_cellx++) {

      const int nektar_geom_id =
          this->cell_id_translation->map_to_nektar[neso_cellx];
      const int expansion_id = geom_to_exp[nektar_geom_id];
      // get the nektar expansion
      auto expansion = this->field->GetExp(expansion_id);

      // is this a tri expansion?
      if ((geom_type_lookup[neso_cellx] == index_tri_geom) &&
          (tri_exp == nullptr)) {
        tri_exp = std::dynamic_pointer_cast<TriExp>(expansion);
      }
      // is this a quad expansion?
      if ((geom_type_lookup[neso_cellx] == index_quad_geom) &&
          (quad_exp == nullptr)) {
        quad_exp = std::dynamic_pointer_cast<QuadExp>(expansion);
      }

      // record offsets and number of coefficients
      this->dh_phys_offsets.h_buffer.ptr[neso_cellx] =
          this->field->GetPhys_Offset(expansion_id);

      this->dh_phys_num0.h_buffer.ptr[neso_cellx] =
          (expansion->GetBase()[0])->GetZ().size();
      this->dh_phys_num1.h_buffer.ptr[neso_cellx] =
          (expansion->GetBase()[1])->GetZ().size();
    }

    // stride between basis values accross all expansion types
    stride_base = 0;
    for (int dimx = 0; dimx < 2; dimx++) {
      if (tri_exp != nullptr) {
        auto base = tri_exp->GetBase();
        stride_base = std::max(stride_base, (int)(base[dimx]->GetZ().size()));
      }
      if (quad_exp != nullptr) {
        auto base = quad_exp->GetBase();
        stride_base = std::max(stride_base, (int)(base[dimx]->GetZ().size()));
      }
    }
    // stride between expansion types.
    stride_expansion_type = 2 * stride_base;

    // malloc space for arrays
    const int num_coeffs_all_types = 2 * stride_expansion_type;
    this->dh_z.realloc_no_copy(num_coeffs_all_types);
    this->dh_bw.realloc_no_copy(num_coeffs_all_types);

    // TriExp has expansion type 0 - is the first set of data
    if (tri_exp != nullptr) {
      this->assemble_data(index_tri_geom,
                          std::static_pointer_cast<StdExpansion2D>(tri_exp));
    }
    // QuadExp has expansion type 1 - is the first set of data
    if (quad_exp != nullptr) {
      this->assemble_data(index_quad_geom,
                          std::static_pointer_cast<StdExpansion2D>(quad_exp));
    }

    this->dh_phys_offsets.host_to_device();
    this->dh_phys_num0.host_to_device();
    this->dh_phys_num1.host_to_device();
    this->dh_z.host_to_device();
    this->dh_bw.host_to_device();
  }

  template <typename U>
  inline void evaluate(ParticleGroupSharedPtr particle_group, Sym<U> sym) {

    auto global_physvals = this->field->GetPhys();
    const int num_global_physvals = global_physvals.size();
    this->dh_global_physvals.realloc_no_copy(num_global_physvals);
    for (int px = 0; px < num_global_physvals; px++) {
      this->dh_global_physvals.h_buffer.ptr[px] = global_physvals[px];
    }
    this->dh_global_physvals.host_to_device();

    auto mpi_rank_dat = particle_group->mpi_rank_dat;
    const auto pl_iter_range = mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride = mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell = mpi_rank_dat->get_particle_loop_npart_cell();

    auto k_output = (*particle_group)[sym]->cell_dat.device_ptr();
    const auto k_ref_positions =
        (*particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]
            ->cell_dat.device_ptr();

    const auto k_global_physvals = this->dh_global_physvals.d_buffer.ptr;
    const auto k_phys_offsets = this->dh_phys_offsets.d_buffer.ptr;
    const auto k_phys_num0 = this->dh_phys_num0.d_buffer.ptr;
    const auto k_phys_num1 = this->dh_phys_num1.d_buffer.ptr;
    const auto k_z = this->dh_z.d_buffer.ptr;
    const auto k_bw = this->dh_bw.d_buffer.ptr;
    const auto k_stride_base = this->stride_base;
    const auto k_stride_expansion_type = this->stride_expansion_type;
    const auto k_map_to_geom_type =
        this->cell_id_translation->dh_map_to_geom_type.d_buffer.ptr;

    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                const int expansion_type = k_map_to_geom_type[cellx];
                const int expansion_type_offset =
                    expansion_type * k_stride_expansion_type;

                const auto z0 = &k_z[expansion_type_offset];
                const auto z1 = &k_z[expansion_type_offset + k_stride_base];
                const auto bw0 = &k_bw[expansion_type_offset];
                const auto bw1 = &k_bw[expansion_type_offset + k_stride_base];

                const REAL coord0 = k_ref_positions[cellx][0][layerx];
                const REAL coord1 = k_ref_positions[cellx][1][layerx];

                // TODO COLLAPSE COORDINATES TODO

                const int num_phys0 = k_phys_num0[cellx];
                const int num_phys1 = k_phys_num1[cellx];

                const auto physvals = &k_global_physvals[k_phys_offsets[cellx]];


                REAL numer1 = 0.0;
                REAL denom1 = 0.0;
                bool mask1 = false;
                REAL eval1 = 0.0;
                for (int i1 = 0; i1 < num_phys1; i1++) {
                  const REAL xdiff1 = z1[i1] - coord1;

                  REAL numer0 = 0.0;
                  REAL denom0 = 0.0;
                  bool mask0 = false;
                  REAL eval0 = 0.0;
                  for (int i0 = 0; i0 < num_phys0; i0++) {

                    REAL xdiff0 = z0[i0] - coord0;
                    REAL pval0 = physvals[i1 * num_phys0 + i0];

                    const bool mask0_inner = (xdiff0 == 0.0);
                    eval0 = mask0_inner ? pval0 : eval0;
                    mask0 = mask0_inner || mask0;

                    const REAL tmp0 = mask0 ? 0.0 : bw0[i0] / xdiff0;
                    numer0 += tmp0 * pval0;
                    denom0 += tmp0;
                  }

                  eval0 = mask0 ? eval0 : numer0 / denom0;
                  REAL pval1 = eval0;

                  const bool mask1_inner = (xdiff1 == 0.0);
                  eval1 = mask1_inner ? pval1 : eval1;
                  mask1 = (mask1_inner || mask1);

                  const REAL tmp1 = mask1 ? 0.0 : bw1[i1] / xdiff1;
                  numer1 += tmp1 * pval1;
                  denom1 += tmp1;
                }

                const REAL evaluation = mask1 ? eval1 : numer1 / denom1;

                k_output[cellx][0][layerx] = evaluation;

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
  }
};

} // namespace NESO

#endif
