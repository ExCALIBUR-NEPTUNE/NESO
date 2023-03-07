#ifndef __FUNCTION_BASIS_EVALUATION_H_
#define __FUNCTION_BASIS_EVALUATION_H_
#include "particle_interface.hpp"
#include <map>
#include <memory>
#include <neso_particles.hpp>

#include <LocalRegions/QuadExp.h>
#include <LocalRegions/TriExp.h>
#include <StdRegions/StdExpansion2D.h>

#include "function_coupling_base.hpp"

using namespace NESO::Particles;
using namespace Nektar::LocalRegions;
using namespace Nektar::StdRegions;

namespace NESO {


class JacobiCoeffModBasis {

protected:




public:
  /// Disable (implicit) copies.
  JacobiCoeffModBasis(const JacobiCoeffModBasis &st) = delete;
  /// Disable (implicit) copies.
  JacobiCoeffModBasis &operator=(JacobiCoeffModBasis const &a) = delete;
  
  /**
   *  Coefficients such that
   *  P_^{alpha, 1}_{n} = 
   *      (coeffs_pnm10) * P_^{alpha, 1}_{n-1} * z
   *    + (coeffs_pnm11) * P_^{alpha, 1}_{n-1} 
   *    + (coeffs_pnm2) * P_^{alpha, 1}_{n-2}
   *
   *  Coefficients are stored in a matrix (row major) where each row gives the
   *  coefficients for a fixed alpha. i.e. the columns are the orders.
   */
  std::vector<double> coeffs_pnm10;
  std::vector<double> coeffs_pnm11;
  std::vector<double> coeffs_pnm2;

  const int max_n;
  const int max_alpha;
  
  JacobiCoeffModBasis(
    const int max_n,
    const int max_alpha
  ) : max_n(max_n), max_alpha(max_alpha) {

    const int beta = 1;
    this->coeffs_pnm10.reserve((max_n+1) * (max_alpha+1));
    this->coeffs_pnm11.reserve((max_n+1) * (max_alpha+1));
    this->coeffs_pnm2.reserve((max_n+1) * (max_alpha+1));

    for(int alphax=0 ; alphax<=max_alpha ; alphax++){
      for(int nx=0 ; nx<=max_n ; nx++){
        const double a = nx + alphax;
        const double b = nx + beta;
        const double c = a + b;
        const double n = nx;
        
        const double c_pn = 2.0 * n * (c - n) * (c - 2.0);
        const double c_pnm10 = (c - 1.0) * c * (c - 2);
        const double c_pnm11 = (c - 1.0) * (a - b) * (c - 2 * n);
        const double c_pnm2 = -2.0 * (a - 1.0) * (b - 1.0) * c;
        
        const double ic_pn = 1.0 / c_pn;

        this->coeffs_pnm10.push_back(ic_pn * c_pnm10);
        this->coeffs_pnm11.push_back(ic_pn * c_pnm11);
        this->coeffs_pnm2.push_back(ic_pn * c_pnm2);
      }
    }
  }

  inline double host_evaluate(
    const int n,
    const int alpha,
    const double z
  ){

    double pnm2 = 1.0;
    if (n == 0) { return pnm2; }
    const int beta = 1;
    double pnm1 = 0.5 * (2 * (alpha + 1) + (alpha + beta + 2) * (z - 1.0));
    if (n == 1) { return pnm1; }
    
    double pn;
    for(int nx=2 ; nx<=n ; nx++){
      const double c_pnm10 = this->coeffs_pnm10[(this->max_n+1) * alpha + nx];
      const double c_pnm11 = this->coeffs_pnm11[(this->max_n+1) * alpha + nx];
      const double c_pnm2 = this->coeffs_pnm2[(this->max_n+1) * alpha + nx];
      pn = c_pnm10 * pnm1 * z + c_pnm11 * pnm1 + c_pnm2 * pnm2;
      pnm2 = pnm1;
      pnm1 = pn;
    }
    
    return pn;
  }

};


/**
 *  Evaluate 2D expansions at particle locations using Bary Interpolation.
 * Reimplements the algorithm in Nektar++.
 */
template <typename T> class BasisEvaluateBase : GeomToExpansionBuilder {
protected:
  std::shared_ptr<T> field;
  ParticleMeshInterfaceSharedPtr mesh;
  CellIDTranslationSharedPtr cell_id_translation;
  SYCLTargetSharedPtr sycl_target;

  std::vector<int> cells_quads;
  std::vector<int> cells_tris;

  std::shared_ptr<JacobiCoeffModBasis> jacobi_coeff;


public:
  /// Disable (implicit) copies.
  BasisEvaluateBase(const BasisEvaluateBase &st) = delete;
  /// Disable (implicit) copies.
  BasisEvaluateBase &operator=(BasisEvaluateBase const &a) = delete;


  /**
   * TODO
   */
  BasisEvaluateBase(std::shared_ptr<T> field,
                   ParticleMeshInterfaceSharedPtr mesh,
                   CellIDTranslationSharedPtr cell_id_translation)
      : field(field), mesh(mesh), cell_id_translation(cell_id_translation),
        sycl_target(cell_id_translation->sycl_target) {

    // build the map from geometry ids to expansion ids
    std::map<int, int> geom_to_exp;
    build_geom_to_expansion_map(this->field, geom_to_exp);

    auto geom_type_lookup =
        this->cell_id_translation->dh_map_to_geom_type.h_buffer.ptr;

    const int index_tri_geom = this->cell_id_translation->index_tri_geom;
    const int index_quad_geom = this->cell_id_translation->index_quad_geom;

    const int neso_cell_count = mesh->get_cell_count();

    int max_n = 1;
    int max_alpha = 1;

    // Assume all TriGeoms and QuadGeoms are the same (TODO generalise for
    // varying p). Get the offsets to the coefficients for each cell.
    for (int neso_cellx = 0; neso_cellx < neso_cell_count; neso_cellx++) {

      const int nektar_geom_id =
          this->cell_id_translation->map_to_nektar[neso_cellx];
      const int expansion_id = geom_to_exp[nektar_geom_id];
      // get the nektar expansion
      auto expansion = this->field->GetExp(expansion_id);

      auto basis0 = expansion->GetBasis(0);
      auto basis1 = expansion->GetBasis(1);
      const int nummodes0 = basis0->GetNumModes();
      const int nummodes1 = basis1->GetNumModes();

      max_n = std::max(max_n, nummodes0 - 1);
      max_n = std::max(max_n, nummodes1 - 1);
      max_alpha = std::max(max_alpha, (nummodes0 - 1) * 2 - 1);

      // is this a tri expansion?
      if (geom_type_lookup[neso_cellx] == index_tri_geom) {
        this->cells_tris.push_back(neso_cellx);
      }
      // is this a quad expansion?
      if (geom_type_lookup[neso_cellx] == index_quad_geom){
        this->cells_quads.push_back(neso_cellx);
      }

    }


    this->jacobi_coeff = std::make_shared<JacobiCoeffModBasis>(max_n, max_alpha);





  }

  /**
   * TODO
   */
  template <typename U, typename V>
  inline void evaluate(ParticleGroupSharedPtr particle_group, Sym<U> sym,
                       const int component, V &global_physvals) {

  }
};

} // namespace NESO

#endif
