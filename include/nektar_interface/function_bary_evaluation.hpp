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

#include "expansion_looping/geom_to_expansion_builder.hpp"
#include "geometry_transport/shape_mapping.hpp"
#include "utility_sycl.hpp"

using namespace NESO::Particles;
using namespace Nektar::LocalRegions;
using namespace Nektar::StdRegions;

namespace NESO {

namespace Bary {

/**
 *  The purpose of these methods is to precompute as much of the Bary
 *  interpolation algorithm for each coordinate dimension as possible (where
 *  the cost is O(p)) to minimise the computational work in the O(p^2) double
 *  loop.
 */
struct Evaluate {

  /**
   * For quadrature point r_i with weight bw_i compute bw_i / (r - r_i).
   *
   * @param[in] stride_base Length of output vector used for temporary storage,
   * i.e. the maximum number of quadrature points across all elements plus any
   * padding.
   * @param num_phys The number of quadrature points for the element and
   * dimension for which this computation is performed.
   * @param[in] coord The evauation point in the dimension of interest.
   * @param[in] z_values A length num_phys array containing the quadrature
   * points.
   * @param[in] z_values A length num_phys array containing the quadrature
   * weights.
   * @param[in, out] exact_index If the input coordinate lie exactly on a
   * quadrature point then this pointer will be set to the index of that
   * quadrature point. Otherwise this memory is untouched.
   * @param[in, out] div_values Array of length stride_base which will be
   * populated with the bw_i/(r - r_i) values. Entries in the range num_phys to
   * stride_base-1 will be zeroed.
   */
  static inline void preprocess_weights(const int stride_base,
                                        const int num_phys, const REAL coord,
                                        const REAL *const z_values,
                                        const REAL *const bw_values,
                                        int *exact_index, REAL *div_values) {
    const sycl::vec<REAL, NESO_VECTOR_LENGTH> coord_vec{coord};

    sycl::global_ptr<const REAL> z_ptr{z_values};
    sycl::global_ptr<const REAL> bw_ptr{bw_values};
    sycl::local_ptr<REAL> div_ptr{div_values};

    for (int ix = 0; ix * NESO_VECTOR_LENGTH < num_phys; ix++) {
      sycl::vec<REAL, NESO_VECTOR_LENGTH> z_vec{};
      z_vec.load(ix, z_ptr);
      sycl::vec<REAL, NESO_VECTOR_LENGTH> bw_vec{};
      bw_vec.load(ix, bw_ptr);
      const auto xdiff_vec = z_vec - coord_vec;
      const auto bw_over_diff = bw_vec / xdiff_vec;
      bw_over_diff.store(ix, div_ptr);

      const int max_index = (((ix + 1) * NESO_VECTOR_LENGTH) <= num_phys)
                                ? ((ix + 1) * NESO_VECTOR_LENGTH)
                                : num_phys;

      for (int jx = ix * NESO_VECTOR_LENGTH; jx < max_index; jx++) {
        if (xdiff_vec[jx % NESO_VECTOR_LENGTH] == 0.0) {
          *exact_index = jx;
        }
      }
    }

    // zero the extra padding values so they do not contribute later
    // If they contributed a NaN all results would be NaN.
    for (int cx = num_phys; cx < stride_base; cx++) {
      div_values[cx] = 0.0;
    }
  };

  /**
   * In each dimension of the Bary interpolation the sum of the weights over
   * distances can be precomputed.
   *
   * @param num_phys Number of quadrature points.
   * @param div_space Values to sum.
   * @returns Sum of the first num_phys values of div_space.
   */
  static inline REAL preprocess_denominator(const int num_phys,
                                            const REAL *const div_space) {
    REAL denom = 0.0;
    for (int ix = 0; ix < num_phys; ix++) {
      const REAL tmp = div_space[ix];
      denom += tmp;
    }
    return denom;
  };

  /**
   * Perform Bary interpolation in the first dimension. This function is
   * intended to be called from a function that performs Bary interpolation
   * over the second dimension and first dimension.
   *
   * @param num_phys Number of quadrature points.
   * @param physvals Vector of length num_phys plus padding to multiple of the
   * vector length which contains the quadrature point values. Padding should
   * contain finite values.
   * @param exact_i If exact_i is non-negative then exact_i then it is assumed
   * that the evaluation point is exactly the quadrature point exact_i.
   * @param denom Sum over Bary weights divided by differences, see
   * preprocess_denominator.
   * @returns Contribution to Bary interpolation from a dimension 0 evaluation.
   */
  static inline REAL compute_dir_0(const int num_phys,
                                   const REAL *const physvals,
                                   const REAL *const div_space,
                                   const int exact_i, const REAL denom) {
    if ((exact_i > -1) && (exact_i < num_phys)) {
      const REAL exact_quadrature_val = physvals[exact_i];
      return exact_quadrature_val;
    } else {
      REAL numer = 0.0;

      for (int ix = 0; ix < num_phys; ix++) {
        const REAL pval = physvals[ix];
        const REAL tmp = div_space[ix];
        numer += tmp * pval;
      }

      const REAL eval0 = numer / denom;
      return eval0;
    }
  };

  /**
   * Computes Bary interpolation over two dimensions. The inner dimension is
   * computed with calls to compute_dir_0.
   *
   * @param num_phys0 Number of quadrature points in dimension 0.
   * @param num_phys1 Number of quadrature points in dimension 1.
   * @param physvals Array of function values at quadrature points.
   * @param div_space0 The output of preprocess_weights applied to dimension 0.
   * @param div_space1 The output of preprocess_weights applied to dimension 1.
   * @param exact_i0 Non-negative value indicates that the coordinate lies on
   * quadrature point exact_i0 in dimension 0.
   * @param exact_i1 Non-negative value indicates that the coordinate lies on
   * quadrature point exact_i1 in dimension 1.
   * @returns Bary evaluation of a function at a coordinate.
   */
  static inline REAL compute_dir_10(const int num_phys0, const int num_phys1,
                                    const REAL *const physvals,
                                    const REAL *const div_space0,
                                    const REAL *const div_space1,
                                    const int exact_i0, const int exact_i1) {
    const REAL denom0 =
        Bary::Evaluate::preprocess_denominator(num_phys0, div_space0);
    if ((exact_i1 > -1) && (exact_i1 < num_phys1)) {
      const REAL bary_eval_0 = Bary::Evaluate::compute_dir_0(
          num_phys0, &physvals[exact_i1 * num_phys0], div_space0, exact_i0,
          denom0);
      return bary_eval_0;
    } else {
      REAL numer = 0.0;
      REAL denom = 0.0;
      for (int ix = 0; ix < num_phys1; ix++) {
        const REAL pval = Bary::Evaluate::compute_dir_0(
            num_phys0, &physvals[ix * num_phys0], div_space0, exact_i0, denom0);
        const REAL tmp = div_space1[ix];
        numer += tmp * pval;
        denom += tmp;
      }
      const REAL eval1 = numer / denom;
      return eval1;
    }
  };
};
} // namespace Bary

/**
 *  Evaluate 2D expansions at particle locations using Bary Interpolation.
 * Reimplements the algorithm in Nektar++.
 */
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

  /**
   *  Helper function to assemble the data required on the device for an
   *  expansion type.
   */
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

    const int index_tri_geom =
        shape_type_to_int(LibUtilities::ShapeType::eTriangle);
    const int index_quad_geom =
        shape_type_to_int(LibUtilities::ShapeType::eQuadrilateral);

    const int neso_cell_count = mesh->get_cell_count();
    this->dh_phys_offsets.realloc_no_copy(neso_cell_count);
    this->dh_phys_num0.realloc_no_copy(neso_cell_count);
    this->dh_phys_num1.realloc_no_copy(neso_cell_count);

    // Assume all TriGeoms and QuadGeoms are the same (TODO generalise for
    // varying p). Get the offsets to the coefficients for each cell.
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

      // Get the number of quadrature points in both dimensions.
      this->dh_phys_num0.h_buffer.ptr[neso_cellx] =
          (expansion->GetBase()[0])->GetZ().size();
      this->dh_phys_num1.h_buffer.ptr[neso_cellx] =
          (expansion->GetBase()[1])->GetZ().size();
    }

    // stride between basis values across all expansion types
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
    stride_base = pad_to_vector_length(stride_base);

    // stride between expansion types.
    stride_expansion_type = 2 * stride_base;

    // malloc space for arrays
    const int num_coeffs_all_types = 2 * stride_expansion_type;
    this->dh_z.realloc_no_copy(num_coeffs_all_types);
    this->dh_bw.realloc_no_copy(num_coeffs_all_types);
    for (int cx = 0; cx < num_coeffs_all_types; cx++) {
      this->dh_z.h_buffer.ptr[cx] = 0.0;
      this->dh_bw.h_buffer.ptr[cx] = 0.0;
    }

    // TriExp has expansion type 0 - is the first set of data
    if (tri_exp != nullptr) {
      this->assemble_data(0, std::static_pointer_cast<StdExpansion2D>(tri_exp));
    }
    // QuadExp has expansion type 1 - is the first set of data
    if (quad_exp != nullptr) {
      this->assemble_data(1,
                          std::static_pointer_cast<StdExpansion2D>(quad_exp));
    }

    this->dh_phys_offsets.host_to_device();
    this->dh_phys_num0.host_to_device();
    this->dh_phys_num1.host_to_device();
    this->dh_z.host_to_device();
    this->dh_bw.host_to_device();
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

    // iteration set specification for the particle loop
    auto mpi_rank_dat = particle_group->mpi_rank_dat;
    const auto pl_stride = mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell = mpi_rank_dat->get_particle_loop_npart_cell();

    const std::size_t local_num_reals = 2 * this->stride_base;
    const std::size_t local_size = get_num_local_work_items(
        this->sycl_target, local_num_reals * sizeof(REAL), 32);
    const std::size_t cell_global_size =
        get_particle_loop_global_size(mpi_rank_dat, local_size);
    const std::size_t ncells = mpi_rank_dat->cell_dat.ncells;

    sycl::range<2> global_iter_set{ncells, cell_global_size};
    sycl::range<2> local_iter_set{1, local_size};
    sycl::nd_range<2> pl_iter_range{global_iter_set, local_iter_set};

    // output and particle position dats
    auto k_output = (*particle_group)[sym]->cell_dat.device_ptr();
    const auto k_ref_positions =
        (*particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]
            ->cell_dat.device_ptr();
    const int k_component = component;

    // values required to evaluate the field
    const int k_index_tri_geom =
        shape_type_to_int(LibUtilities::ShapeType::eTriangle);
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
          // Allocate local memory to compute the divides.
          sycl::accessor<REAL, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>
              local_mem(sycl::range<1>(local_num_reals * local_size), cgh);

          cgh.parallel_for<>(pl_iter_range, [=](sycl::nd_item<2> idx) {
            const INT cellx = idx.get_global_id(0);
            const int idx_local = idx.get_local_id(1);
            const INT layerx = idx.get_global_id(1);
            if (layerx < pl_npart_cell[cellx]) {

              REAL *div_space0 = &local_mem[idx_local * 2 * k_stride_base];
              REAL *div_space1 = div_space0 + k_stride_base;

              // query the map from cells to expansion type
              const int expansion_type = k_map_to_geom_type[cellx];
              // use the type key to index into the Bary weights and points
              const int expansion_type_offset =
                  (expansion_type == k_index_tri_geom)
                      ? 0
                      : k_stride_expansion_type;
              // get the z values and weights for this expansion type
              const auto z0 = &k_z[expansion_type_offset];
              const auto z1 = &k_z[expansion_type_offset + k_stride_base];
              const auto bw0 = &k_bw[expansion_type_offset];
              const auto bw1 = &k_bw[expansion_type_offset + k_stride_base];

              const REAL xi0 = k_ref_positions[cellx][0][layerx];
              const REAL xi1 = k_ref_positions[cellx][1][layerx];
              // If this cell is a triangle then we need to map to the
              // collapsed coordinates.
              REAL coord0, coord1;

              GeometryInterface::loc_coord_to_loc_collapsed_2d(
                  expansion_type, xi0, xi1, &coord0, &coord1);

              const int num_phys0 = k_phys_num0[cellx];
              const int num_phys1 = k_phys_num1[cellx];

              // Get pointer to the start of the quadrature point values for
              // this cell
              const auto physvals = &k_global_physvals[k_phys_offsets[cellx]];

              int exact_i0 = -1;
              int exact_i1 = -1;

              Bary::Evaluate::preprocess_weights(k_stride_base, num_phys0,
                                                 coord0, z0, bw0, &exact_i0,
                                                 div_space0);
              Bary::Evaluate::preprocess_weights(k_stride_base, num_phys1,
                                                 coord1, z1, bw1, &exact_i1,
                                                 div_space1);

              const REAL evaluation = Bary::Evaluate::compute_dir_10(
                  num_phys0, num_phys1, physvals, div_space0, div_space1,
                  exact_i0, exact_i1);

              k_output[cellx][k_component][layerx] = evaluation;
            }
          });
        })
        .wait_and_throw();
  }
};

} // namespace NESO

#endif
