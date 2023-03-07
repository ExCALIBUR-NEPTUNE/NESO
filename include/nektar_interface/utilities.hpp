#ifndef __UTILITIES_H_
#define __UTILITIES_H_

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include <MultiRegions/ContField.h>
#include <MultiRegions/DisContField.h>
#include <LibUtilities/Foundations/Basis.h>
#include <LibUtilities/Polylib/Polylib.h>

using namespace Nektar;
using namespace Nektar::SpatialDomains;
using namespace Nektar::MultiRegions;

#include <neso_particles.hpp>

using namespace NESO::Particles;
using namespace Nektar::LibUtilities;



namespace NESO {


/*

   if (i==0, 0 <= j < P){
    phi^b_{ij}(z_k) = phi^a_j(z_k) 
   } else if (1 <= i < P, j == 0) {
    phi^b{ij}(z_k) = 0.5 * (1 - z_k)
   } else if (1 <= i < P, 1 <= j < P-i) {
    phi^b{ij} = 0.5 * (1 - z_k) * 0.5 * (1 + z_k) * P^{2i-1,1}_{j-1}(z_k)
   }


  m_bdata[n(i,j) + k*m_numpoints] =
  \f$ \phi^b_{ij}(z_k) = \left \{ \begin{array}{lll}
  \phi^a_j(z_k) & i = 0, &   0\leq j < P  \\
  \\
  \left ( \frac{1-z_k}{2}\right )^{i}  & 1 \leq i < P,&   j = 0 \\
  \\
  \left ( \frac{1-z_k}{2}\right )^{i} \left ( \frac{1+z_k}{2}\right )
  P^{2i-1,1}_{j-1}(z_k) & 1 \leq i < P,\ &  1\leq j < P-i\ \\
  \end{array}  \right . , \f$
 
  where \f$ n(i,j) \f$ is a consecutive ordering of the
  triangular indices \f$ 0 \leq i, i+j < P \f$ where \a j
  runs fastest.
*/

  inline int pochhammer(
    const int m,
    const int n
  ){
    int output = 1;
    for(int offset=0 ; offset<=(n-1) ; offset++){
      output *= (m + offset);
    }
    return output;
  };

  inline double jacobi(
    const int p, 
    const double z, 
    const int alpha, 
    const int beta
  ){
    if (p == 0) { return 1.0; }
    double pnm1 = 0.5 * (2 * (alpha + 1) + (alpha + beta + 2) * (z - 1.0));
    if (p == 1) {return pnm1;}
    double pn = 0.125 * (
        4 * (alpha + 1) * (alpha + 2) + 
        4 * (alpha + beta + 3) * (alpha + 2) * (z - 1.0) +
        (alpha + beta + 3) * (alpha + beta + 4) * (z - 1.0) * (z - 1.0)
      );
    if (p == 2) { return pn;}
    
    double pnp1;
    for (int px = 3 ; px<= p ; px++){
      const int n = (px - 1);
      const double coeff_pnp1 = 2 * (n + 1) * (n + alpha + beta + 1) * (2 * n + alpha + beta);
      const double coeff_pn = (2 * n + alpha + beta + 1) * (alpha * alpha - beta * beta) + 
        pochhammer(2 * n + alpha + beta, 3) * z;
      const double coeff_pnm1 = -2.0 * (n + alpha) * (n + beta) * (2*n+alpha+beta+2);

      pnp1 = (1.0 / coeff_pnp1) * (coeff_pn * pn + coeff_pnm1 * pnm1);

      pnm1 = pn;
      pn = pnp1;
    }

    return pnp1;
  };
  
  inline double eval_modA_i(const int p, const double z){
    const double b0 = 0.5 * (1.0 - z);
    const double b1 = 0.5 * (1.0 + z);
    if (p == 0){
      return b0;
    }
    if (p == 1){
      return b1;
    }
    return b0*b1 * jacobi(p-2, z, 1, 1);
  }

  inline void eval_modA(const int p, const double z, std::vector<double> &b) {
    b[0] = 0.5 * (1.0 - z);
    b[1] = 0.5 * (1.0 + z);
    const double b0b1 = b[0] * b[1];
    for(int px=0 ; px<p-2 ; px++){
      b[px] = b0b1 * jacobi(px, z, 1, 1); 
    }
  };

  inline double eval_modB_ij_book_85(
    const int i, 
    const int j,
    const int nummodes0,
    const int nummodes1,
    const double z) {

      double output;

      if ((i==0) && (0 <= j) && (j <= nummodes1)){
       // phi^b_{ij}(z_k) = phi^a_j(z_k)
       output = eval_modA_i(j, z);
      } else if ((i == nummodes0) && (0 <= j) && (j <= nummodes1)) {
       output = eval_modA_i(j, z);
      } else if ((1 <= i) && (i < nummodes0) && (j == 0)) {
       // phi^b{ij}(z_k) = 0.5 * (1 - z_k)
       output = std::pow(0.5 * (1.0 - z), i + 1);
      } else if ((1 <= i) && (i < nummodes0) && (1 <= j) && (j < (nummodes1-1))) {
       //phi^b{ij} = 0.5 * (1 - z_k) * 0.5 * (1 + z_k) * P^{2i-1,1}_{j-1}(z_k)
       output = std::pow(0.5 * (1.0 - z), i + 1) * 0.5 * (1.0 + z) * jacobi(j-1, z, 2*i+1,1);
      } else {
        NESOASSERT(false, "should be unreachable");
      }
    return output;
  };

  
  inline double eval_modB_ij(
      const int i,
      const int j,
      const double z){

      double output;

      if (i==0){
       output = eval_modA_i(j, z);
      } else if (j == 0) {
       output = std::pow(0.5 * (1.0 - z), (double) i);
      } else {
       output = std::pow(0.5 * (1.0 - z), (double) i) * 0.5 * (1.0 + z) * jacobi(j-1, z, 2*i-1,1);
      }
    return output;
  }





  inline void to_collapsed(Array<OneD, NekDouble> &xi, double * eta){
    const REAL xi0 = xi[0];
    const REAL xi1 = xi[1];

    const NekDouble d1_original = 1.0 - xi1;
    const bool mask_small_cond =
        (fabs(d1_original) < NekConstants::kNekZeroTol);
    NekDouble d1 = d1_original;

    d1 = (mask_small_cond && (d1 >= 0.0))
             ? NekConstants::kNekZeroTol
             : ((mask_small_cond && (d1 < 0.0))
                    ? -NekConstants::kNekZeroTol
                    : d1);
    eta[0] = 2. * (1. + xi0) / d1 - 1.0;
    eta[1] = xi1;
  };

/**
 *  TODO
 *
 */
template <typename T>
inline double evaluate_poly_scalar_2d(std::shared_ptr<T> field, const double x,
                                 const double y) {
  Array<OneD, NekDouble> xi(2);
  Array<OneD, NekDouble> eta_array(2);
  Array<OneD, NekDouble> coords(2);

  coords[0] = x;
  coords[1] = y;
  int elmtIdx = field->GetExpIndex(coords, xi);
  auto elmtPhys = field->GetPhys() + field->GetPhys_Offset(elmtIdx);

  //const double eval = field->GetExp(elmtIdx)->StdPhysEvaluate(xi, elmtPhys);
  auto expansion = field->GetExp(elmtIdx);
  
  auto global_coeffs = field->GetCoeffs();
  const int coeff_offset = field->GetCoeff_Offset(elmtIdx);
  
  auto coeffs = &global_coeffs[coeff_offset];
  
  const int num_modes = expansion->GetNcoeffs();
  
  auto basis0 = expansion->GetBasis(0);
  auto basis1 = expansion->GetBasis(1);

  const int nummodes0 = basis0->GetNumModes();
  const int nummodes1 = basis1->GetNumModes();

  nprint(
      basis0->GetBasisType() == eModified_A, 
      basis1->GetBasisType() == eModified_A,
      basis1->GetBasisType() == eModified_B
  );

  const bool quad = (basis0->GetBasisType() == eModified_A) && (basis1->GetBasisType() == eModified_A);

  std::cout << num_modes << " ---------------------------" << std::endl;

  if (quad){
  /*  
    std::vector<double> b0(nummodes0);
    std::vector<double> b1(nummodes1);

    eval_modA(nummodes0, xi[0], b0);
    eval_modA(nummodes1, xi[1], b1);
    
    double eval = 0.0;
    for(int px=0 ; px<nummodes0 ; px++){
      for(int py=0 ; py<nummodes1 ; py++){

        const double inner_coeff = coeffs[py * nummodes0 + px];
        const double basis_eval = b0[px] * b1[py];
        eval += inner_coeff * basis_eval;
      }
    }

    const double eval_correct = field->GetExp(elmtIdx)->StdPhysEvaluate(xi, elmtPhys);
    const double err = abs(eval_correct - eval);

    if (err > 1.0e-12){
      nprint("BAD EVAL:", err, eval_correct, eval);
    }

    auto bdata0 = basis0->GetBdata();
    auto bdata1 = basis1->GetBdata();
    auto Z0 = basis0->GetZ();
    auto Z1 = basis1->GetZ();
    
    nprint("Z0 num points", basis0->GetNumPoints());
    nprint("Z0 size:", Z0.size());
    nprint("Z1 size:", Z1.size());
    nprint("nummodes0", nummodes0, "nummodes1", nummodes1);
    nprint("bdata0 size:", bdata0.size());
    nprint("bdata1 size:", bdata1.size());
    
    const int numpoints0 = basis0->GetNumPoints();
    int tindex=0;
    for(int px=0 ; px<nummodes0 ; px++){
      for(int qx=0 ; qx<numpoints0 ; qx++){
        const double ztmp = Z0[qx];
        const double btmp = bdata0[tindex++];
        const double etmp = eval_modA_i(px, Z0[qx]);
        const double err = abs(btmp - etmp);
        if (err > 1.0e-12){
          nprint("BAD EVAL quad dir0 err:\t", err, "\t", btmp, "\t", etmp, "\t", ztmp);
        }
      }
    }
*/

  } else {
    double eta[2];
    to_collapsed(xi, eta);
    eta_array[0] = eta[0];
    eta_array[1] = eta[1];

    //eta[0] = xi[0];
    //eta[1] = xi[1];

    //nprint("~~~~~~~~~~~~~~~~");
    //nprint("(4)_5:", pochhammer(4,5));
    //nprint("P^(3,4)_5(0.3)", jacobi(5, 0.3, 3,4));
    //nprint("P^(7,11)_13(0.3)", jacobi(13, 0.3, 7,11));
    //nprint("P^(7,11)_13(-0.5)", jacobi(13, -0.5, 7,11));


    
    auto bdata0 = basis0->GetBdata();
    auto bdata1 = basis1->GetBdata();
    auto Z0 = basis0->GetZ();
    auto Z1 = basis1->GetZ();
    
    nprint("Z0 num points", basis0->GetNumPoints());
    nprint("Z1 num points", basis1->GetNumPoints());
    nprint("Z0 size:", Z0.size());
    nprint("Z1 size:", Z1.size());
    nprint("nummodes0", nummodes0, "nummodes1", nummodes1);
    nprint("bdata0 size:", bdata0.size());
    nprint("bdata1 size:", bdata1.size());


    for(int nx=0 ; nx<nummodes0 ; nx++){
      nprint("n0:", nx, Z0[nx]);
    }
    for(int nx=0 ; nx<nummodes1 ; nx++){
      nprint("n1:", nx, Z1[nx]);
    }
    

    const int numpoints1 = basis1->GetNumPoints();
    int tindex=0;
    for(int px=0 ; px<nummodes1 ; px++){
      for(int qx=0 ; qx<(nummodes1-px) ; qx++){
        for(int pointx=0 ; pointx<numpoints1 ; pointx++){
          const double ztmp = Z1[pointx];
          const double btmp = bdata1[tindex++];
          const double etmp = eval_modB_ij(px, qx, ztmp);
          const double err = abs(btmp - etmp);
          //if (err > 1.0e-12){
            nprint(px, qx, "BAD EVAL tqp dir1 err:\t", err, "\t", btmp, "\t", etmp, "\t", ztmp);
          //}
        }
      }
    }


    const int nummodes_total = expansion->GetNcoeffs();
    const double eval_correct = field->GetExp(elmtIdx)->StdPhysEvaluate(xi, elmtPhys);

    double eval_modes = 0.0;
    nprint("N coeffs", nummodes_total);
    for(int modex=0 ; modex<nummodes_total ; modex++){
      const double basis_eval = expansion->PhysEvaluateBasis(xi, modex);
      eval_modes += basis_eval * coeffs[modex];
    }

    const double err_modes = abs(eval_correct - eval_modes);
    nprint("TRI EVAL MODES:", err_modes, eval_correct, eval_modes);


    tindex=0;
    for(int px=0 ; px<nummodes1 ; px++){
      for(int qx=0 ; qx<(nummodes1-px) ; qx++){
        const int modex = tindex++;
        const double ztmp0 = eta[0];
        const double ztmp1 = eta[1];
        const double btmp = expansion->PhysEvaluateBasis(xi, modex);
        double etmp0 = eval_modA_i(px, ztmp0);
        double etmp1 = eval_modB_ij(px, qx, ztmp1);
        //if(modex==1){
        //  etmp1 *= 1.0 / eval_modA_i(px, ztmp0);
        //}
        // or
        if(modex==1){
          etmp0 = 1.0;
        }
        const double etmp = etmp0 * etmp1;
        const double err = abs(btmp - etmp);
        //if (err > 1.0e-12){
          nprint(px, qx, "BAD TB EVAL err:\t", err, "\t", btmp, "\t", etmp);
        //}
      }
    }

    nprint("nummodes total:", nummodes_total, "tindex", tindex);







  
    


  }
  


  return 0.0;
}


/**
 * Utility class to convert Nektar field names to indices.
 */
class NektarFieldIndexMap {
private:
  std::map<std::string, int> field_to_index;

public:
  NektarFieldIndexMap(std::vector<std::string> field_names) {
    int index = 0;
    for (auto field_name : field_names) {
      this->field_to_index[field_name] = index++;
    }
  }
  int get_idx(std::string field_name) {
    return (this->field_to_index.count(field_name) > 0)
               ? this->field_to_index[field_name]
               : -1;
  }
};

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

} // namespace NESO

#endif
