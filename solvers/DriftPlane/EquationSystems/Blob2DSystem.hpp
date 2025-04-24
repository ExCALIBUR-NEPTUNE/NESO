
#ifndef __NESOSOLVERS_DRIFTPLANE_BLOB2DSYSTEM_HPP__
#define __NESOSOLVERS_DRIFTPLANE_BLOB2DSYSTEM_HPP__

#include <SolverUtils/AdvectionSystem.h>

#include "DriftPlaneSystem.hpp"

namespace NESO::Solvers::DriftPlane {

/**
 * @brief An equation system for the Blob2D solver.
 */
class Blob2DSystem : public DriftPlaneSystem {
public:
  /// Allow the memory manager to allocate shared pointers of this class.
  friend class Nektar::MemoryManager<Blob2DSystem>;

  /// Creates an instance of this class.
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &session,
         const SD::MeshGraphSharedPtr &graph) {
    SU::EquationSystemSharedPtr p =
        Nektar::MemoryManager<Blob2DSystem>::AllocateSharedPtr(session, graph);
    p->InitObject();
    return p;
  }

  /// Label used to statically initialise a function ptr for the create method.
  static std::string class_name;

  /// Default destructor.
  virtual ~Blob2DSystem() = default;

protected:
  /// Protected constructor (Instances are constructed via a factory).
  Blob2DSystem(const LU::SessionReaderSharedPtr &session,
               const SD::MeshGraphSharedPtr &graph);

  virtual void create_riemann_solver() override;
  virtual void
  explicit_time_int(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                    Array<OneD, Array<OneD, NekDouble>> &out_arr,
                    const NekDouble time) override;
  virtual void v_InitObject(bool create_fields) override;
};
} // namespace NESO::Solvers::DriftPlane

#endif // __NESOSOLVERS_DRIFTPLANE_BLOB2DSYSTEM_HPP__
