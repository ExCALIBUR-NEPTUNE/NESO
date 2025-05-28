#ifndef _NESO_PARTICLE_CELL_MAPPING_NEWTON_RELATIVE_EXIT_TOLERANCES_HPP_
#define _NESO_PARTICLE_CELL_MAPPING_NEWTON_RELATIVE_EXIT_TOLERANCES_HPP_
#include <SpatialDomains/MeshGraphIO.h>
#include <neso_particles.hpp>

namespace NESO {

/**
 * Device copyable type for computing relative errors.
 */
struct NewtonRelativeExitTolerances {
  Particles::REAL scaling_jacobian{1.0};
  Particles::REAL scaling_dir[3]{1.0, 1.0, 1.0};

  /**
   * @param f0 Residual in direction 0.
   * @param f1 Residual in direction 1.
   * @returns Maximum of inputs scaled to be relative errors in each direction.
   */
  inline Particles::REAL get_relative_error_2d(const Particles::REAL f0,
                                               const Particles::REAL f1) const {
    const Particles::REAL err0 = sycl::fabs(f0) * this->scaling_dir[0];
    const Particles::REAL err1 = sycl::fabs(f1) * this->scaling_dir[1];
    return sycl::fmax(err0, err1);
  }

  /**
   * @param f0 Residual in direction 0.
   * @param f1 Residual in direction 1.
   * @param f2 Residual in direction 2.
   * @returns Maximum of inputs scaled to be relative errors in each direction.
   */
  inline Particles::REAL get_relative_error_3d(const Particles::REAL f0,
                                               const Particles::REAL f1,
                                               const Particles::REAL f2) const {
    const Particles::REAL err0 = sycl::fabs(f0) * this->scaling_dir[0];
    const Particles::REAL err1 = sycl::fabs(f1) * this->scaling_dir[1];
    const Particles::REAL err2 = sycl::fabs(f2) * this->scaling_dir[2];
    return sycl::fmax(sycl::fmax(err0, err1), err2);
  }
};

/**
 * Initialise a NewtonRelativeExitTolerances instance from a geometry object.
 *
 * @param[in] geom Geometry object to create tolerances object from.
 * @param[in, out] Output device copyable tolerances object.
 */
inline void create_newton_relative_exit_tolerances(
    Nektar::SpatialDomains::GeometrySharedPtr geom,
    NewtonRelativeExitTolerances *newton_relative_exit_tolerances) {

  auto m_xmap = geom->GetXmap();
  auto m_geomFactors = geom->GetGeomFactors();
  Nektar::Array<Nektar::OneD, const Nektar::NekDouble> Jac =
      m_geomFactors->GetJac(m_xmap->GetPointsKeys());
  Nektar::NekDouble tol_scaling =
      Vmath::Vsum(Jac.size(), Jac, 1) / ((Nektar::NekDouble)Jac.size());
  newton_relative_exit_tolerances->scaling_jacobian =
      static_cast<Particles::REAL>(ABS(1.0 / tol_scaling));

  auto bounding_box = geom->GetBoundingBox();
  const auto ndim = geom->GetCoordim();
  for (int dx = 0; dx < ndim; dx++) {
    newton_relative_exit_tolerances->scaling_dir[dx] =
        1.0 /
        (std::max(std::abs(bounding_box[dx]), std::abs(bounding_box[dx + 3])) *
         tol_scaling);
  }
}

} // namespace NESO

#endif
