#pragma once
#include <LibUtilities/BasicUtils/ShapeType.hpp>
namespace NekUtil = Nektar::LibUtilities;
namespace NESO::Project {
struct eQuad {
  static constexpr NekUtil::ShapeType shape_type = NekUtil::eQuadrilateral;
  static constexpr int dim = 2;
};

struct eTri {
  static constexpr NekUtil::ShapeType shape_type = NekUtil::eTriangle;
  static constexpr int dim = 2;
};
} // namespace NESO::Project
