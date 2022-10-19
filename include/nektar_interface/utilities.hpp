#ifndef __UTILITIES_H_
#define __UTILITIES_H_

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <MultiRegions/ContField.h>
#include <MultiRegions/DisContField.h>

using namespace Nektar;
using namespace Nektar::SpatialDomains;
using namespace Nektar::MultiRegions;

namespace NESO {

/**
 *  Interpolate f(x,y) onto a Nektar++ field.
 *
 *  @param func Function matching a signature like: double func(double x,
 *  double y);
 *  @parma field Output Nektar++ field.
 *
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

  field->SetPhys(f);

  Array<OneD, NekDouble> coeffs_f((unsigned)field->GetNcoeffs());

  // interpolate onto expansion
  field->FwdTrans(f, coeffs_f);
  field->SetCoeffsArray(coeffs_f);
}

/**
 *  Write a Nektar++ field to vtu for visualisation in Paraview.
 *
 *  @param field Nektar++ field to write.
 *  @param filename Output filename. Most likely should end in vtu.
 */
template <typename T>
inline void write_vtu(std::shared_ptr<T> field, std::string filename) {

  std::filebuf file_buf;
  file_buf.open(filename, std::ios::out);
  std::ostream outfile(&file_buf);

  field->WriteVtkHeader(outfile);

  auto expansions = field->GetExp();
  const int num_expansions = (*expansions).size();
  for (int ex = 0; ex < num_expansions; ex++) {
    field->WriteVtkPieceHeader(outfile, ex);
    field->WriteVtkPieceData(outfile, ex);
    field->WriteVtkPieceFooter(outfile, ex);
  }

  field->WriteVtkFooter(outfile);

  file_buf.close();
}

} // namespace NESO

#endif
