#ifndef __UTILITIES_H_
#define __UTILITIES_H_

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mpi.h>
#include <string>

#include <LibUtilities/Foundations/Basis.h>
#include <LibUtilities/Polylib/Polylib.h>
#include <MultiRegions/ContField.h>
#include <MultiRegions/DisContField.h>

#include "cell_id_translation.hpp"

using namespace Nektar;
using namespace Nektar::SpatialDomains;
using namespace Nektar::MultiRegions;
using namespace Nektar::LibUtilities;

#include <neso_particles.hpp>
using namespace NESO::Particles;

namespace NESO {

/**
 * Utility class to convert Nektar field names to indices.
 */
class NektarFieldIndexMap {
private:
  std::map<std::string, int> field_to_index;

public:
  /**
   *  Create map from field names to indices. It is assumed that the field
   *  index is the position in the input vector.
   *
   *  @param field_names Vector of field names.
   */
  NektarFieldIndexMap(std::vector<std::string> field_names) {
    int index = 0;
    for (auto field_name : field_names) {
      this->field_to_index[field_name] = index++;
    }
  }
  /**
   *  Get the index of a field by name.
   *
   *  @param field_name Name of field to get index for.
   *  @returns Non-negative integer if field exists -1 otherwise.
   */
  int get_idx(std::string field_name) {
    return (this->field_to_index.count(field_name) > 0)
               ? this->field_to_index[field_name]
               : -1;
  }

  /**
   * Identical to get_idx except this method mirrors the std library behaviour
   * and is fatal if the named field does not exist in the map.
   *
   * @param field_name Name of field to get index for.
   * @returns Non-negative integer if field exists.
   */
  int at(std::string field_name) { return this->field_to_index.at(field_name); }
};

/**
 *  Interpolate f(x,y) or f(x,y,z) onto a Nektar++ field.
 *
 *  @param func Function matching a signature like: double func(double x,
 *  double y) or func(double x, double y, double z).
 *  @param field Output Nektar++ field.
 */
template <typename T, typename U>
inline void interpolate_onto_nektar_field_3d(T &func,
                                             std::shared_ptr<U> field) {

  // space for quadrature points
  const int tot_points = field->GetTotPoints();
  Array<OneD, NekDouble> x(tot_points);
  Array<OneD, NekDouble> y(tot_points);
  Array<OneD, NekDouble> f(tot_points);
  Array<OneD, NekDouble> z(tot_points);

  // Evaluate function at quadrature points.
  field->GetCoords(x, y, z);
  for (int pointx = 0; pointx < tot_points; pointx++) {
    f[pointx] = func(x[pointx], y[pointx], z[pointx]);
  }

  const int num_coeffs = field->GetNcoeffs();
  Array<OneD, NekDouble> coeffs_f(num_coeffs);
  for (int cx = 0; cx < num_coeffs; cx++) {
    coeffs_f[cx] = 0.0;
  }

  // interpolate onto expansion
  field->FwdTrans(f, coeffs_f);
  for (int cx = 0; cx < num_coeffs; cx++) {
    field->SetCoeff(cx, coeffs_f[cx]);
  }

  // transform backwards onto phys
  field->BwdTrans(coeffs_f, f);
  field->SetPhys(f);
}

/**
 *  Interpolate f(x,y) onto a Nektar++ field.
 *
 *  @param func Function matching a signature like: double func(double x,
 *  double y);
 *  @parma field Output Nektar++ field.
 */
template <typename T, typename U>
inline void interpolate_onto_nektar_field_2d(T &func,
                                             std::shared_ptr<U> field) {

  // space for quadrature points
  const int tot_points = field->GetTotPoints();
  Array<OneD, NekDouble> x(tot_points);
  Array<OneD, NekDouble> y(tot_points);
  Array<OneD, NekDouble> f(tot_points);

  // Evaluate function at quadrature points.
  field->GetCoords(x, y);
  for (int pointx = 0; pointx < tot_points; pointx++) {
    f[pointx] = func(x[pointx], y[pointx]);
  }

  const int num_coeffs = field->GetNcoeffs();
  Array<OneD, NekDouble> coeffs_f(num_coeffs);
  for (int cx = 0; cx < num_coeffs; cx++) {
    coeffs_f[cx] = 0.0;
  }

  // interpolate onto expansion
  field->FwdTrans(f, coeffs_f);
  for (int cx = 0; cx < num_coeffs; cx++) {
    field->SetCoeff(cx, coeffs_f[cx]);
  }

  // transform backwards onto phys
  field->BwdTrans(coeffs_f, f);
  field->SetPhys(f);
}

/**
 *  Write a Nektar++ field to vtu for visualisation in Paraview.
 *
 *  @param field Nektar++ field to write.
 *  @param filename Output filename. Most likely should end in vtu.
 */
template <typename T>
inline void write_vtu(std::shared_ptr<T> field, std::string filename,
                      std::string var = "v") {

  std::filebuf file_buf;
  file_buf.open(filename, std::ios::out);
  std::ostream outfile(&file_buf);

  field->WriteVtkHeader(outfile);

  auto expansions = field->GetExp();
  const int num_expansions = (*expansions).size();
  for (int ex = 0; ex < num_expansions; ex++) {
    field->WriteVtkPieceHeader(outfile, ex);
    field->WriteVtkPieceData(outfile, ex, var);
    field->WriteVtkPieceFooter(outfile, ex);
  }

  field->WriteVtkFooter(outfile);

  file_buf.close();
}

/**
 * Evaluate a scalar valued Nektar++ function at a point. Avoids assertion
 * issue.
 *
 * @param field Nektar++ field.
 * @param x X coordinate.
 * @param y Y coordinate.
 * @returns Evaluation.
 */
template <typename T>
inline double evaluate_scalar_2d(std::shared_ptr<T> field, const double x,
                                 const double y) {
  Array<OneD, NekDouble> xi(2);
  Array<OneD, NekDouble> coords(2);

  coords[0] = x;
  coords[1] = y;
  int elmtIdx = field->GetExpIndex(coords, xi);
  auto elmtPhys = field->GetPhys() + field->GetPhys_Offset(elmtIdx);

  const double eval = field->GetExp(elmtIdx)->StdPhysEvaluate(xi, elmtPhys);
  return eval;
}

/**
 * Evaluate a scalar valued Nektar++ function at a point. Avoids assertion
 * issue.
 *
 * @param field Nektar++ field.
 * @param x X coordinate.
 * @param y Y coordinate.
 * @returns Evaluation.
 */
template <typename T>
inline double evaluate_scalar_3d(std::shared_ptr<T> field, const double x,
                                 const double y, const double z) {
  Array<OneD, NekDouble> xi(3);
  Array<OneD, NekDouble> coords(3);

  coords[0] = x;
  coords[1] = y;
  coords[2] = z;
  int elmtIdx = field->GetExpIndex(coords, xi);
  auto elmtPhys = field->GetPhys() + field->GetPhys_Offset(elmtIdx);

  const double eval = field->GetExp(elmtIdx)->StdPhysEvaluate(xi, elmtPhys);
  return eval;
}

/**
 * Evaluate the derivative of scalar valued Nektar++ function at a point.
 * Avoids assertion issue.
 *
 * @param field Nektar++ field.
 * @param x X coordinate.
 * @param y Y coordinate.
 * @param direction Direction for derivative.
 * @returns Output du/d(direction).
 */
template <typename T>
inline double evaluate_scalar_derivative_2d(std::shared_ptr<T> field,
                                            const double x, const double y,
                                            const int direction) {
  Array<OneD, NekDouble> xi(2);
  Array<OneD, NekDouble> coords(2);

  coords[0] = x;
  coords[1] = y;
  int elmtIdx = field->GetExpIndex(coords, xi);
  auto elmtPhys = field->GetPhys() + field->GetPhys_Offset(elmtIdx);
  auto expansion = field->GetExp(elmtIdx);

  const int num_quadrature_points = expansion->GetTotPoints();
  auto du = Array<OneD, NekDouble>(num_quadrature_points);

  expansion->PhysDeriv(direction, elmtPhys, du);

  const double eval = expansion->StdPhysEvaluate(xi, du);
  return eval;
}

/**
 * Globally find the nektar++ geometry object that owns a point.
 *
 * @param[in] graph MeshGraph to locate particle on.
 * @param[in] point Subscriptable description of point.
 * @param[out] owning_rank_output Rank that owns the point.
 * @param[out] coord_ref_output Reference coordinates (optional).
 * @param[out] num_geoms_output Number of geometry objects that claimed the
 * point (optional).
 * @param[in] comm MPI communicator to use (default MPI_COMM_WORLD).
 * @returns True if an owning geometry object was found.
 */
template <typename T>
inline bool find_owning_geom(Nektar::SpatialDomains::MeshGraphSharedPtr graph,
                             const T &point, int *owning_rank_output,
                             int *geom_id_output,
                             double *coord_ref_output = nullptr,
                             int *num_geoms_output = nullptr,
                             MPI_Comm comm = MPI_COMM_WORLD) {
  bool geom_found = false;

  const int ndim = point.size();

  NekDouble x = point[0];
  NekDouble y = (ndim > 1) ? point[1] : 0.0;
  NekDouble z = (ndim > 2) ? point[2] : 0.0;
  auto point_nektar = std::make_shared<PointGeom>(point.size(), 0, x, y, z);
  auto candidate_geom_ids = graph->GetElementsContainingPoint(point_nektar);

  Array<OneD, NekDouble> coord_phys{3};
  coord_phys[0] = x;
  coord_phys[1] = y;
  coord_phys[2] = z;
  Array<OneD, NekDouble> coord_ref{3};
  coord_ref[0] = 0.0;
  coord_ref[1] = 0.0;
  coord_ref[2] = 0.0;

  int rank, size;
  MPICHK(MPI_Comm_rank(comm, &rank));
  MPICHK(MPI_Comm_size(comm, &size));

  int num_geoms = 0;
  for (const int geom_id : candidate_geom_ids) {
    auto geom = graph->GetGeometry2D(geom_id);
    geom_found = geom->ContainsPoint(coord_phys, coord_ref, 0.0);
    if (geom_found) {
      if (coord_ref_output != nullptr) {
        for (int dimx = 0; dimx < ndim; dimx++) {
          coord_ref_output[dimx] = coord_ref[dimx];
        }
      }
      *geom_id_output = geom_id;
      num_geoms++;
    }
  }

  // reduce to get consensus in the case where multiple geoms claim they own
  // the point.
  const int send_value = geom_found ? rank : size;
  int recv_value;
  MPICHK(MPI_Allreduce(&send_value, &recv_value, 1, MPI_INT, MPI_MIN, comm));

  if (recv_value == size) {
    // case where no geom found the point
    geom_found = false;
    *owning_rank_output = -1;
  } else {
    // recv_value contains the rank that claims the point
    geom_found = true;
    *owning_rank_output = recv_value;
    MPICHK(MPI_Bcast(geom_id_output, 1, MPI_INT, recv_value, comm));
    if (coord_ref_output != nullptr) {
      std::vector<double> tmp_coords(ndim);
      for (int dimx = 0; dimx < ndim; dimx++) {
        tmp_coords[dimx] = coord_ref_output[dimx];
      }
      MPICHK(MPI_Bcast(tmp_coords.data(), ndim, MPI_DOUBLE, recv_value, comm));
      for (int dimx = 0; dimx < ndim; dimx++) {
        coord_ref_output[dimx] = tmp_coords[dimx];
      }
    }
  }

  // reduce the count of number of geometry objects that claimed the point.
  if (num_geoms_output != nullptr) {
    MPICHK(
        MPI_Allreduce(&num_geoms, num_geoms_output, 1, MPI_INT, MPI_SUM, comm));
  }

  return geom_found;
}

/**
 * Helper class to map directly from NESO-Particles cells to Nektar++
 * expansions.
 */
class NESOCellsToNektarExp {
protected:
  ExpListSharedPtr exp_list;
  std::map<int, int> map;

public:
  /**
   *  Create map for a given DisContField or ContField.
   *
   *  @param exp_list DistContField or ContField (ExpList deriviative)
   * containing expansions for each cell in the mesh.
   *  @param cell_id_translation CellIDTranslation instance for the MeshGraph
   * used by the expansion list.
   */
  template <typename T>
  NESOCellsToNektarExp(std::shared_ptr<T> exp_list,
                       CellIDTranslationSharedPtr cell_id_translation) {

    this->exp_list = std::dynamic_pointer_cast<ExpList>(exp_list);

    std::map<int, int> map_geom_to_exp;
    for (int ei = 0; ei < exp_list->GetNumElmts(); ei++) {
      auto ex = exp_list->GetExp(ei);
      auto geom = ex->GetGeom();
      const int gid = geom->GetGlobalID();
      map_geom_to_exp[gid] = ei;
    }

    const int num_cells = cell_id_translation->map_to_nektar.size();
    for (int neso_cell = 0; neso_cell < num_cells; neso_cell++) {
      const int gid = cell_id_translation->map_to_nektar[neso_cell];
      const int exp_id = map_geom_to_exp.at(gid);
      this->map[neso_cell] = exp_id;
    }
  }

  /**
   *  Get the expansion that corresponds to a input NESO::Particles cell.
   *
   *  @param neso_cell NESO::Particles cell to get expansion for.
   *  @returns Nektar++ expansion for requested cell.
   */
  inline LocalRegions::ExpansionSharedPtr get_exp(const int neso_cell) {
    return this->exp_list->GetExp(this->get_exp_id(neso_cell));
  }

  /**
   *  Get the expansion id that corresponds to a input NESO::Particles cell.
   *
   *  @param neso_cell NESO::Particles cell to get expansion for.
   *  @returns Nektar++ expansion id for requested cell.
   */
  inline int get_exp_id(const int neso_cell) { return this->map.at(neso_cell); }
};

} // namespace NESO

#endif
