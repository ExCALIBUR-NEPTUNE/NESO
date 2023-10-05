#ifndef __BASIS_EVALUATE_BASE_H_
#define __BASIS_EVALUATE_BASE_H_

#include "geom_to_expansion_builder.hpp"

namespace NESO {

/**
 * Base class for derived classes that evaluate Nektar++ basis functions to
 * evaluate and project onto fields.
 */
template <typename T> class BasisEvaluateBase : GeomToExpansionBuilder {
protected:
  std::shared_ptr<T> field;
  ParticleMeshInterfaceSharedPtr mesh;
  CellIDTranslationSharedPtr cell_id_translation;
  SYCLTargetSharedPtr sycl_target;

  BufferDeviceHost<int> dh_nummodes;

  std::map<ShapeType, int> map_shape_to_count;
  std::map<ShapeType, std::vector<int>> map_shape_to_cells;
  std::map<ShapeType, std::unique_ptr<BufferDeviceHost<int>>>
      map_shape_to_dh_cells;

  BufferDeviceHost<int> dh_coeffs_offsets;
  BufferDeviceHost<REAL> dh_global_coeffs;
  BufferDeviceHost<REAL> dh_coeffs_pnm10;
  BufferDeviceHost<REAL> dh_coeffs_pnm11;
  BufferDeviceHost<REAL> dh_coeffs_pnm2;
  int stride_n;
  std::map<ShapeType, std::array<int, 3>> map_total_nummodes;

public:
  /// Do all the elements have the same node count?
  bool common_nummodes;

  /// Disable (implicit) copies.
  BasisEvaluateBase(const BasisEvaluateBase &st) = delete;
  /// Disable (implicit) copies.
  BasisEvaluateBase &operator=(BasisEvaluateBase const &a) = delete;

  /**
   * Create new instance. Expected to be called by a derived class - not a user.
   *
   * @param field Example field this class will be used to evaluate basis
   * functions for.
   * @param mesh Interface between NESO-Particles and Nektar++ meshes.
   * @param cell_id_translation Map between NESO-Particles cells and Nektar++
   * cells.
   */
  BasisEvaluateBase(std::shared_ptr<T> field,
                    ParticleMeshInterfaceSharedPtr mesh,
                    CellIDTranslationSharedPtr cell_id_translation)
      : field(field), mesh(mesh), cell_id_translation(cell_id_translation),
        sycl_target(cell_id_translation->sycl_target),
        dh_nummodes(sycl_target, 1), dh_global_coeffs(sycl_target, 1),
        dh_coeffs_offsets(sycl_target, 1), dh_coeffs_pnm10(sycl_target, 1),
        dh_coeffs_pnm11(sycl_target, 1), dh_coeffs_pnm2(sycl_target, 1) {

    // build the map from geometry ids to expansion ids
    std::map<int, int> geom_to_exp;
    build_geom_to_expansion_map(this->field, geom_to_exp);

    auto geom_type_lookup =
        this->cell_id_translation->dh_map_to_geom_type.h_buffer.ptr;

    const int index_tri_geom =
        shape_type_to_int(LibUtilities::ShapeType::eTriangle);
    const int index_quad_geom =
        shape_type_to_int(LibUtilities::ShapeType::eQuadrilateral);

    const int neso_cell_count = mesh->get_cell_count();

    this->dh_nummodes.realloc_no_copy(neso_cell_count);
    this->dh_coeffs_offsets.realloc_no_copy(neso_cell_count);

    int max_n = 1;
    int max_alpha = 1;

    std::array<ShapeType, 6> shapes = {eTriangle, eQuadrilateral, eHexahedron,
                                       ePrism,    ePyramid,       eTetrahedron};
    for (auto shape : shapes) {
      this->map_shape_to_count[shape] = 0;
      this->map_shape_to_count[shape] = 0;
      for (int dimx = 0; dimx < 3; dimx++) {
        this->map_total_nummodes[shape][dimx] = 0;
      }
      this->map_shape_to_dh_cells[shape] =
          std::make_unique<BufferDeviceHost<int>>(this->sycl_target, 1);
    }

    for (int neso_cellx = 0; neso_cellx < neso_cell_count; neso_cellx++) {

      const int nektar_geom_id =
          this->cell_id_translation->map_to_nektar[neso_cellx];
      const int expansion_id = geom_to_exp[nektar_geom_id];
      // get the nektar expansion
      auto expansion = this->field->GetExp(expansion_id);
      auto basis = expansion->GetBase();
      const int expansion_ndim = basis.size();

      // build the map from shape types to neso cells
      auto shape_type = expansion->DetShapeType();
      this->map_shape_to_cells[shape_type].push_back(neso_cellx);

      // do all the elements have the same number of modes
      int common_nummodes_tmp = -1;
      this->common_nummodes = true;

      for (int dimx = 0; dimx < expansion_ndim; dimx++) {
        const int basis_nummodes = basis[dimx]->GetNumModes();
        const int basis_total_nummodes = basis[dimx]->GetTotNumModes();
        max_n = std::max(max_n, basis_nummodes - 1);
        if (dimx == 0) {
          this->dh_nummodes.h_buffer.ptr[neso_cellx] = basis_nummodes;
        } else {
          NESOASSERT(this->dh_nummodes.h_buffer.ptr[neso_cellx] ==
                         basis_nummodes,
                     "Differing numbers of modes in coordinate directions.");
        }
        this->map_total_nummodes.at(shape_type).at(dimx) =
            std::max(this->map_total_nummodes.at(shape_type).at(dimx),
                     basis_total_nummodes);

        if (common_nummodes_tmp < 0) {
          common_nummodes_tmp = basis_nummodes;
        } else if (basis_nummodes != common_nummodes_tmp) {
          this->common_nummodes = false;
        }
      }

      // determine the maximum Jacobi order and alpha value required to
      // evaluate the basis functions for this expansion
      int alpha_tmp = 0;
      int n_tmp = 0;
      BasisReference::get_total_num_modes(
          shape_type, this->dh_nummodes.h_buffer.ptr[neso_cellx], &n_tmp,
          &alpha_tmp);
      max_alpha = std::max(max_alpha, alpha_tmp);
      max_n = std::max(max_n, n_tmp);

      // record offsets and number of coefficients
      this->dh_coeffs_offsets.h_buffer.ptr[neso_cellx] =
          this->field->GetCoeff_Offset(expansion_id);
    }

    int expansion_count = 0;
    // create the maps from shape types to NESO::Particles cells which
    // correpond to the shape type.

    for (auto &item : this->map_shape_to_cells) {
      expansion_count += item.second.size();
      auto shape_type = item.first;
      auto &cells = item.second;
      const int num_cells = cells.size();
      // allocate and build the map.
      this->map_shape_to_dh_cells[shape_type] =
          std::make_unique<BufferDeviceHost<int>>(this->sycl_target, num_cells);
      for (int cellx = 0; cellx < num_cells; cellx++) {
        const int cell = cells[cellx];
        this->map_shape_to_dh_cells[shape_type]->h_buffer.ptr[cellx] = cell;
      }
      this->map_shape_to_dh_cells[shape_type]->host_to_device();
      this->map_shape_to_count[shape_type] = num_cells;
    }

    NESOASSERT(expansion_count == neso_cell_count,
               "Missmatch in number of cells found and total number of cells.");

    JacobiCoeffModBasis jacobi_coeff(max_n, max_alpha);

    const int num_coeffs = jacobi_coeff.coeffs_pnm10.size();
    this->dh_coeffs_pnm10.realloc_no_copy(num_coeffs);
    this->dh_coeffs_pnm11.realloc_no_copy(num_coeffs);
    this->dh_coeffs_pnm2.realloc_no_copy(num_coeffs);
    for (int cx = 0; cx < num_coeffs; cx++) {
      this->dh_coeffs_pnm10.h_buffer.ptr[cx] = jacobi_coeff.coeffs_pnm10[cx];
      this->dh_coeffs_pnm11.h_buffer.ptr[cx] = jacobi_coeff.coeffs_pnm11[cx];
      this->dh_coeffs_pnm2.h_buffer.ptr[cx] = jacobi_coeff.coeffs_pnm2[cx];
    }
    this->stride_n = jacobi_coeff.stride_n;

    this->dh_coeffs_offsets.host_to_device();
    this->dh_nummodes.host_to_device();
    this->dh_coeffs_pnm10.host_to_device();
    this->dh_coeffs_pnm11.host_to_device();
    this->dh_coeffs_pnm2.host_to_device();
  }
};

} // namespace NESO

#endif
