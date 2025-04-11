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
    Array<OneD, NekDouble> local_coord(3);
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
    auto lambda_contains_point = [&]() -> bool {
      geom->GetLocCoords(coord, local_coord);
      bool contained = true;
      for (int dx = 0; dx < ndim; dx++) {
        // Restrict inwards using the tolerance as we really do not want to
        // sample points outside the geom as then the position might be outside
        // the domain.
        bool dim_contained =
            ((-1.0 + tol) < local_coord[dx]) && (local_coord[dx] < (1.0 - tol));
        contained = contained && dim_contained;
      }
      return contained && geom->ContainsPoint(coord);
    };

    int trial_count = 0;
    while (!lambda_contains_point()) {
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

std::mt19937
dist_within_extents(Nektar::SpatialDomains::MeshGraphSharedPtr graph,
                    Nektar::LibUtilities::EquationSharedPtr eqn, const double t,
                    const int npart,
                    std::vector<std::vector<double>> &positions,
                    std::vector<int> &cells, const REAL tol,
                    std::optional<std::mt19937> rng_in) {

  std::mt19937 rng;
  if (!rng_in) {
    rng = std::mt19937(std::random_device{}());
  } else {
    rng = rng_in.value();
  }

  const int ndim = graph->GetMeshDimension();
  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>> geoms_2d;
  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>> geoms_3d;
  int nelements;

  if (ndim == 2) {
    get_all_elements_2d(graph, geoms_2d);
    nelements = geoms_2d.size();
  } else if (ndim == 3) {
    get_all_elements_3d(graph, geoms_3d);
    nelements = geoms_3d.size();
  }

  auto lambda_sample = [&](auto geom, Array<OneD, NekDouble> &coord) {
    Array<OneD, NekDouble> local_coord(3);
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

    auto lambda_contains_point = [&]() -> bool {
      geom->GetLocCoords(coord, local_coord);
      bool contained = true;
      for (int dx = 0; dx < ndim; dx++) {
        // Restrict inwards using the tolerance as we really do not want to
        // sample points outside the geom as then the position might be outside
        // the domain.
        bool dim_contained =
            ((-1.0 + tol) < local_coord[dx]) && (local_coord[dx] < (1.0 - tol));
        contained = contained && dim_contained;
      }
      return contained && geom->ContainsPoint(coord);
    };

    int trial_count = 0;
    while (!lambda_contains_point()) {
      lambda_sample_new();
      trial_count++;
      NESOASSERT(trial_count < 1000000, "Unable to sample point in geom.");
    }
  };

  auto lambda_dispatch = [&](auto &container) {
    Array<OneD, NekDouble> coord(3);
    std::vector<double> weight_per_cell(nelements, 0);
    std::vector<int> flat_idx(nelements);
    double local_weight = 0;
    auto lambda_preprocess = [&](auto &geoms) {
      double x, y, z;
      int ex = 0;
      for (const auto &[id, geom] : geoms) {
        int Nv = geom->GetNumVerts();
        for (int v = 0; v < Nv; ++v) {
          geom->GetVertex(v)->GetCoords(x, y, z);
          double P = eqn->Evaluate(x, y, z, t);
          NESOASSERT(P >= 0 && P <= 1, "Probability distribution must be "
                                       "between 0 and 1, but evaluates to " +
                                           std::to_string(P));
          weight_per_cell[ex] += P / Nv;
        }
        local_weight += weight_per_cell[ex];
        flat_idx[ex] = id;
        ex++;
      }
    };

    lambda_preprocess(container);

    double global_weight = 0;
    MPI_Allreduce(&local_weight, &global_weight, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    int npart_local = std::round(npart * local_weight / global_weight);

    if (npart_local > 0) {
      positions.resize(ndim);
      cells.resize(npart_local);
      for (int dimx = 0; dimx < ndim; dimx++) {
        positions[dimx] = std::vector<double>(npart_local);
      }
      std::discrete_distribution disc(weight_per_cell.begin(),
                                      weight_per_cell.end());

      int index = 0;
      while (index < npart_local) {
        auto cell = disc(rng);
        lambda_sample(container[flat_idx[cell]], coord);
        double P = eqn->Evaluate(coord[0], coord[1], coord[2], t);
        NESOASSERT(P >= 0 && P <= 1, "Probability distribution must be "
                                     "between 0 and 1, but evaluates to " +
                                         std::to_string(P));
        std::bernoulli_distribution bern(P);

        if (bern(rng)) {
          for (int dx = 0; dx < ndim; dx++) {
            positions.at(dx).at(index) = coord[dx];
          }
          cells.at(index) = cell;
          index++;
        }
      }
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
