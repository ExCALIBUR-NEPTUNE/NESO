#ifndef _NESO_PARTICLE_INITIALISATION_LINE
#define _NESO_PARTICLE_INITIALISATION_LINE

#include <mpi.h>
#include <neso_particles.hpp>
#include <random>
#include <vector>

#include "../nektar_interface/particle_interface.hpp"

using namespace NESO::Particles;

namespace NESO {

/**
 *  Helper class to initialise particles that line along a (discretised)
 *  straight line embedded in a Nektar++ mesh. The line is decomposed into N
 *  nodes which are possible creation locations for particles. These points are
 *  decomposed over the mesh such that the MPI rank that owns a point in space
 *  holds the point in this class. The reference positions and NESO-Particles
 *  cells are also precomputed and stored.
 */
class ParticleInitialisationLine {
protected:
public:
  /// The number of possible particle locations along the line.
  const int npoints_total;
  /// The number of points along the line owned by this MPI rank.
  int npoints_local;
  /// Physical location of points indexed by [dimension][point_index];
  std::vector<std::vector<double>> point_phys_positions;
  /// Reference location of points indexed by [dimension][point_index];
  std::vector<std::vector<double>> point_ref_positions;
  /// NESO cell index of points indexed by [point_index].
  std::vector<int> point_neso_cells;

  /**
   *  Create N points along a line by specifying the start and end of the line
   *  and the number of possible points. These N points are evenly distributed
   *  over the line.
   *
   *  @param domain A NESO-Particles domain instance. This domain must be
   * created with a NESO ParticleMeshInterface mesh object.
   *  @param sycl_target SYCLTarget to use.
   *  @param line_start Starting point of the line. Number of components must be
   * at least the spatial dimension of the passed domain.
   *  @param line_end End point of the line. Number of components must be at
   * least the spatial dimension of the passed domain.
   *  @param npoints_total Number of possible locations where particles can be
   * created along the line.
   */
  ParticleInitialisationLine(DomainSharedPtr domain,
                             SYCLTargetSharedPtr sycl_target,
                             std::vector<double> line_start,
                             std::vector<double> line_end,
                             const int npoints_total)
      : npoints_total(npoints_total) {

    const int ndim = domain->mesh->get_ndim();
    NESOASSERT(ndim <= line_start.size(),
               "line_start has fewer dimensions than particle domain.");
    NESOASSERT(ndim <= line_end.size(),
               "line_end has fewer dimensions than particle domain.");
    NESOASSERT(npoints_total > 0, "Bad npoints_total value.");

    ParticleSpec particle_spec{ParticleProp(Sym<REAL>("POSITION"), ndim, true),
                               ParticleProp(Sym<INT>("CELL_ID"), 1, true)};
    auto particle_group =
        std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

    long rstart, rend;
    const long size = sycl_target->comm_pair.size_parent;
    const long rank = sycl_target->comm_pair.rank_parent;
    get_decomp_1d(size, (long)this->npoints_total, rank, &rstart, &rend);
    const long N = rend - rstart;

    std::vector<double> r_step(ndim);
    const double ih = 1.0 / ((double)npoints_total);

    for (int dimx = 0; dimx < ndim; dimx++) {
      r_step[dimx] = (line_end[dimx] - line_start[dimx]) * ih;
    }

    if (N > 0) {
      ParticleSet initial_distribution(N, particle_group->get_particle_spec());

      for (int px = 0; px < N; px++) {
        const double offset_px = rstart + px;

        for (int dimx = 0; dimx < ndim; dimx++) {
          initial_distribution[Sym<REAL>("POSITION")][px][dimx] =
              line_start[dimx] + offset_px * r_step[dimx];
        }
      }

      particle_group->add_particles_local(initial_distribution);
    }
    NESO::Particles::parallel_advection_initialisation(particle_group);

    auto particle_mesh_interface =
        dynamic_pointer_cast<ParticleMeshInterface>(domain->mesh);
    // Setup map between cell indices
    auto cell_id_translation = std::make_shared<CellIDTranslation>(
        sycl_target, particle_group->cell_id_dat, particle_mesh_interface);
    cell_id_translation->execute();
    particle_group->cell_move();

    // the particles left on this rank correspond to the owned points along the
    // line
    this->npoints_local = particle_group->get_npart_local();

    this->point_phys_positions.resize(ndim);
    this->point_ref_positions.resize(ndim);
    this->point_neso_cells.resize(this->npoints_local);
    for (int dimx = 0; dimx < ndim; dimx++) {
      this->point_phys_positions[dimx].resize(this->npoints_local);
      this->point_ref_positions[dimx].resize(this->npoints_local);
    }

    // loop over local cells and collect points
    int particle_index = 0;

    // for each cell
    const int cell_count = domain->mesh->get_cell_count();
    for (int cellx = 0; cellx < cell_count; cellx++) {
      // for each particle in the cell

      auto cells =
          (*particle_group)[Sym<INT>("CELL_ID")]->cell_dat.get_cell(cellx);
      const int nrow = cells->nrow;
      // many cells will be empty so we check before issuing more copies
      if (nrow > 0) {
        auto phys_positions =
            (*particle_group)[Sym<REAL>("POSITION")]->cell_dat.get_cell(cellx);
        auto ref_positions =
            (*particle_group)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]
                ->cell_dat.get_cell(cellx);
        for (int rowx = 0; rowx < nrow; rowx++) {
          // copy the particle data into the store of points

          this->point_neso_cells[particle_index] = cellx;
          for (int dimx = 0; dimx < ndim; dimx++) {
            this->point_phys_positions[dimx][particle_index] =
                (*phys_positions)[dimx][rowx];
            this->point_ref_positions[dimx][particle_index] =
                (*ref_positions)[dimx][rowx];
          }
          particle_index++;
        }
      }
    }

    NESOASSERT(
        particle_index == this->npoints_local,
        "Miss-match in number of found points and expected number of points.");

    particle_group->free();
  };
};

/**
 *  Simple class to globally sample from a set of points. N samples are made on
 *  each MPI rank and those within the bounds of the point set are kept.
 */
template <typename T> class SimpleUniformPointSampler {
protected:
  SYCLTargetSharedPtr sycl_target;
  std::shared_ptr<T> initialisation_points;
  std::mt19937 rng;
  std::uniform_int_distribution<int> point_distribution;

  int index_start;
  int index_end;

public:
  /// The seed used by the RNG used to sample points.
  int rng_seed;

  /**
   *  Create instance from a set of points.
   *
   *  @param sycl_target SYCLTarget instance to use.
   *  @param initialisation_points Set of points, e.g.
   * ParticleInitialisationLine instance.
   *  @param seed Pointer to global seed to use for sampling, default NULL.
   */
  SimpleUniformPointSampler(SYCLTargetSharedPtr sycl_target,
                            std::shared_ptr<T> initialisation_points,
                            unsigned int *seed = NULL)
      : sycl_target(sycl_target), initialisation_points(initialisation_points) {
    this->point_distribution = std::uniform_int_distribution<int>(
        0, this->initialisation_points->npoints_total - 1);

    const int rank = this->sycl_target->comm_pair.rank_parent;
    const int size = this->sycl_target->comm_pair.size_parent;
    MPI_Comm comm = this->sycl_target->comm_pair.comm_parent;

    unsigned int seed_gen;
    if (rank == 0) {
      if (seed == NULL) {
        std::random_device rd;
        seed_gen = rd();
      } else {
        seed_gen = *seed;
      }
    }

    MPICHK(MPI_Bcast(&seed_gen, 1, MPI_UNSIGNED, 0, comm));
    this->rng = std::mt19937(seed_gen);
    this->rng_seed = seed_gen;

    std::vector<int> scan_output(size);
    MPICHK(MPI_Scan(&this->initialisation_points->npoints_local,
                    scan_output.data(), 1, MPI_INT, MPI_SUM, comm));

    this->index_start = scan_output[rank];
    this->index_end = (rank == (size - 1))
                          ? this->initialisation_points->npoints_total - 1
                          : scan_output[rank + 1];
  }

  /**
   * Sample N local point indices (globally) and place the indices in the
   * provided output container.
   *
   * @param num_samples Number of points to sample.
   * @param output_container Output, push_back will be called on this instance
   * to add point indices.
   * @returns Number of points sampled on this MPI rank.
   */
  template <typename U>
  inline int get_samples(const int num_samples, U &output_container) {

    int count = 0;
    for (int samplex = 0; samplex < num_samples; samplex++) {
      int sample = this->point_distribution(this->rng);
      if ((sample >= index_start) && (sample <= index_end)) {
        count++;
        output_container.push_back(sample - index_start);
      }
    }

    return count;
  }
};

} // namespace NESO

#endif
