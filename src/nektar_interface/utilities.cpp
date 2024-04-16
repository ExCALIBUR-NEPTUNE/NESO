#include <nektar_interface/utilities.hpp>

namespace NESO {

std::mt19937 uniform_within_elements(
    Nektar::SpatialDomains::MeshGraphSharedPtr graph, const int npart_per_cell,
    std::vector<std::vector<double>> &positions, std::vector<int> &cells,
    const REAL tol, std::optional<std::mt19937> rng_in) {

  std::mt19937 rng;
  if (!rng_in) {
    rng = std::mt19937(std::random_device{}());
  } else {
    rng = rng_in.value();
  }

  const int ndim = graph->GetMeshDimension();
  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>> geoms_2d;
  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>> geoms_3d;
  int npart_total;
  int nelements;

  if (ndim == 2) {
    get_all_elements_2d(graph, geoms_2d);
    nelements = geoms_2d.size();
  } else if (ndim == 3) {
    get_all_elements_3d(graph, geoms_3d);
    nelements = geoms_3d.size();
  }
  npart_total = nelements * npart_per_cell;

  positions.resize(ndim);
  cells.resize(npart_total);
  for (int dimx = 0; dimx < ndim; dimx++) {
    positions[dimx] = std::vector<double>(npart_total);
  }

  auto lambda_sample = [&](auto geom, Array<OneD, NekDouble> &coord) {
    auto bb = geom->GetBoundingBox();
    coord[0] = 0.0;
    coord[1] = 0.0;
    coord[2] = 0.0;

    auto lambda_sample_new = [&]() {
      for (int dx = 0; dx < ndim; dx++) {
        const REAL bound_lower = bb[dx];
        const REAL bound_upper = bb[dx + 3];
        std::uniform_real_distribution<double> dist(bound_lower, bound_upper);
        coord[dx] = dist(rng);
      }
    };

    lambda_sample_new();
    int trial_count = 0;
    while (!geom->ContainsPoint(coord, tol)) {
      lambda_sample_new();
      trial_count++;
      NESOASSERT(trial_count < 1000000, "Unable to sample point in geom.");
    }
  };

  auto lambda_dispatch = [&](auto container) {
    Array<OneD, NekDouble> coord(3);
    int ex = 0;
    int index = 0;
    for (auto id_element : container) {
      for (int px = 0; px < npart_per_cell; px++) {
        lambda_sample(id_element.second, coord);
        for (int dx = 0; dx < ndim; dx++) {
          positions.at(dx).at(index) = coord[dx];
        }
        cells.at(index) = ex;
        index++;
      }
      ex++;
    }
  };

  if (ndim == 2) {
    lambda_dispatch(geoms_2d);
  } else if (ndim == 3) {
    lambda_dispatch(geoms_3d);
  }

  return rng;
}

} // namespace NESO
