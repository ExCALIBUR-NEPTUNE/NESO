#ifndef __EMPTY_PARTSYS_H_
#define __EMPTY_PARTSYS_H_
/**
 * @file empty_partsys.hpp
 * @brief A dummy system to use when templating EqnSys or TimeEvoEqnSysBase if
 * no particle functionality is required.
 *
 * Design choice: While fluid-only (single template param) forms of the
 * classes in solver_base could be written, using a dummy particle system is
 * simpler and makes it easier to add a real system later, should the need
 * arise.
 */

#include <nektar_interface/solver_base/partsys_base.hpp>

namespace LU = Nektar::LibUtilities;
namespace SD = Nektar::SpatialDomains;

namespace NESO::Particles {

class EmptyPartSys : public PartSysBase {
public:
  EmptyPartSys(LU::SessionReaderSharedPtr session, SD::MeshGraphSharedPtr graph,
               MPI_Comm comm = MPI_COMM_WORLD)
      : PartSysBase(session, graph, {}, comm) {}
};
} // namespace NESO::Particles

#endif