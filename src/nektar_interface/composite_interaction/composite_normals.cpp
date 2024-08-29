#include <nektar_interface/composite_interaction/composite_normals.hpp>

namespace NESO::CompositeInteraction {

void get_normal_vector(std::shared_ptr<SpatialDomains::Geometry1D> geom,
                       std::vector<REAL> &normal) {
  SpatialDomains::SegGeomSharedPtr curve;
  if ((curve = std::dynamic_pointer_cast<SpatialDomains::SegGeom>(geom)) !=
      nullptr) {
    NESOASSERT(curve->GetCurve() == nullptr,
               "Cannot compute a normal vector to a curved line.");
  }

  NESOASSERT(geom->GetNumVerts() == 2, "Expected 2 vertices.");
  auto a = geom->GetVertex(0);
  auto b = geom->GetVertex(1);

  NekDouble x0, y0, z0;
  a->GetCoords(x0, y0, z0);
  NekDouble x1, y1, z1;
  b->GetCoords(x1, y1, z1);

  // compute the normal to the facet
  const REAL dx = x1 - x0;
  const REAL dy = y1 - y0;
  const REAL n0t = -dy;
  const REAL n1t = dx;
  const REAL l = 1.0 / std::sqrt(n0t * n0t + n1t * n1t);
  const REAL n0 = n0t * l;
  const REAL n1 = n1t * l;

  normal.clear();
  normal.reserve(2);
  normal.push_back(n0);
  normal.push_back(n1);
}

void get_normal_vector(std::shared_ptr<SpatialDomains::Geometry2D> geom,
                       std::vector<REAL> &normal) {

  NESOASSERT(geom->GetCurve() == nullptr,
             "Cannot compute a constant normal to a curved surface.");

  const int num_verts = geom->GetNumVerts();
  auto v0 = geom->GetVertex(0);
  auto v1 = geom->GetVertex(1);
  auto v2 = geom->GetVertex(num_verts - 1);

  Array<OneD, NekDouble> c0(3);
  NekDouble v00, v01, v02;
  NekDouble v10, v11, v12;
  NekDouble v20, v21, v22;
  v0->GetCoords(v00, v01, v02);
  v1->GetCoords(v10, v11, v12);
  v2->GetCoords(v20, v21, v22);
  const REAL d00 = v10 - v00;
  const REAL d01 = v11 - v01;
  const REAL d02 = v12 - v02;
  const REAL d10 = v20 - v00;
  const REAL d11 = v21 - v01;
  const REAL d12 = v22 - v02;

  REAL ntx, nty, ntz;

  KERNEL_CROSS_PRODUCT_3D(d00, d01, d02, d10, d11, d12, ntx, nty, ntz);
  const REAL n_inorm =
      1.0 / std::sqrt(KERNEL_DOT_PRODUCT_3D(ntx, nty, ntz, ntx, nty, ntz));

  ntx = ntx * n_inorm;
  nty = nty * n_inorm;
  ntz = ntz * n_inorm;

  NESOASSERT(std::isfinite(ntx), "ntx is not finite.");
  NESOASSERT(std::isfinite(nty), "nty is not finite.");
  NESOASSERT(std::isfinite(ntz), "ntz is not finite.");
  NESOASSERT(std::isfinite(n_inorm), "n_inorm is not finite.");

  normal.clear();
  normal.reserve(3);
  normal.push_back(ntx);
  normal.push_back(nty);
  normal.push_back(ntz);
}

} // namespace NESO::CompositeInteraction
