#ifndef __COMPOSITE_INTERSECTION_H_
#define __COMPOSITE_INTERSECTION_H_

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include <neso_particles.hpp>
using namespace NESO::Particles;

#include <nektar_interface/geometry_transport/packed_geom_2d.hpp>
#include <nektar_interface/particle_cell_mapping/x_map_newton_kernel.hpp>
#include <nektar_interface/particle_mesh_interface.hpp>
#include <nektar_interface/special_functions.hpp>

#include "composite_collections.hpp"

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace NESO::Newton;

namespace NESO::CompositeInteraction {

/**
 *  High-level class to detect and compute the intersection of a particle
 *  trajectory and a Nektar++ composite.
 */
class CompositeIntersection {
protected:
  const int ndim;
  const int num_cells;
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;
  std::unique_ptr<BufferDevice<int>> d_cell_min_maxes;
  std::unique_ptr<MeshHierarchyMapper> mesh_hierarchy_mapper;
  std::unique_ptr<BufferDeviceHost<int>> dh_max_bounding_box_size;
  std::unique_ptr<BufferDeviceHost<INT>> dh_mh_cells;
  std::unique_ptr<BufferDeviceHost<int>> dh_mh_cells_index;
  /// Exit tolerance for Newton iteration.
  REAL newton_tol;
  /// Maximum number of Newton iterations.
  INT newton_max_iteration;

  template <typename T> inline void check_iteration_set(std::shared_ptr<T>) {
    static_assert(std::is_same_v<T, ParticleGroup> ||
                  std::is_same_v<T, ParticleSubGroup>);
  }

  inline ParticleGroupSharedPtr
  get_particle_group(ParticleGroupSharedPtr iteration_set) {
    return iteration_set;
  }
  inline ParticleGroupSharedPtr
  get_particle_group(ParticleSubGroupSharedPtr iteration_set) {
    return iteration_set->get_particle_group();
  }

  /**
   * TODO
   */
  template <typename T>
  void find_cells(std::shared_ptr<T> iteration_set, std::set<INT> &cells);

  /**
   *  TODO
   */
  template <typename T>
  void find_intersections(std::shared_ptr<T> iteration_set,
                          ParticleDatSharedPtr<INT> dat_composite,
                          ParticleDatSharedPtr<REAL> dat_positions);

public:
  /// The CompositeCollections used to detect intersections.
  std::shared_ptr<CompositeCollections> composite_collections;
  /// Disable (implicit) copies.
  CompositeIntersection(const CompositeIntersection &st) = delete;
  /// Disable (implicit) copies.
  CompositeIntersection &operator=(CompositeIntersection const &a) = delete;

  /// SYCLTarget to use for computation.
  SYCLTargetSharedPtr sycl_target;

  /// The NESO::Particles Sym<REAL> used to store the previous particle
  /// position.
  const static inline Sym<REAL> previous_position_sym =
      Sym<REAL>("NESO_COMP_INT_PREV_POS");

  const static inline std::string output_sym_position_name =
      "NESO_COMP_INT_OUTPUT_POS";
  const static inline std::string output_sym_composite_name =
      "NESO_COMP_INT_OUTPUT_COMP";

  /// The composite indices for which the class detects intersections with.
  const std::vector<int> composite_indices;

  /**
   * TODO
   */
  void free();

  /**
   *  TODO
   */
  CompositeIntersection(
      SYCLTargetSharedPtr sycl_target,
      ParticleMeshInterfaceSharedPtr particle_mesh_interface,
      std::vector<int> &composite_indices,
      ParameterStoreSharedPtr config = std::make_shared<ParameterStore>());

  /**
   *  Method to store the current particle positions before an integration step.
   *
   *  @param iteration_set Particles to store current positions of.
   *  @param output_sym_composite_name Optionally specifiy the property to
   *  store composite intersection information in. Otherwise use the default.
   */
  template <typename T>
  void pre_integration(std::shared_ptr<T> iteration_set,
                       Sym<INT> output_sym_composite = Sym<INT>(
                           CompositeIntersection::output_sym_composite_name));

  /**
   *  TODO
   */
  template <typename T>
  void execute(std::shared_ptr<T> iteration_set,
               Sym<INT> output_sym_composite =
                   Sym<INT>(CompositeIntersection::output_sym_composite_name),
               Sym<REAL> output_sym_position =
                   Sym<REAL>(CompositeIntersection::output_sym_position_name));

  /**
   *  TODO
   */
  template <typename T>
  std::map<int, ParticleSubGroupSharedPtr>
  get_intersections(std::shared_ptr<T> iteration_set,
                    Sym<INT> output_sym_composite = Sym<INT>(
                        CompositeIntersection::output_sym_composite_name),
                    Sym<REAL> output_sym_position = Sym<REAL>(
                        CompositeIntersection::output_sym_position_name));
};

} // namespace NESO::CompositeInteraction

#endif
