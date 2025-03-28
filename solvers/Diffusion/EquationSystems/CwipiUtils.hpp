#ifndef __NESOSOLVERS_DIFFUSION_CWIPIUTILS_HPP__
#define __NESOSOLVERS_DIFFUSION_CWIPIUTILS_HPP__

#include <string>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <MultiRegions/ExpList.h>
#include <SolverUtils/Core/Coupling.h>
#include <neso_particles.hpp>

namespace LU = Nektar::LibUtilities;
namespace MR = Nektar::MultiRegions;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::Diffusion {
inline SU::CouplingSharedPtr
construct_coupling_obj(const LU::SessionReaderSharedPtr &session,
                       const MR::ExpListSharedPtr &field) {
  // Init coupling obj
  const std::string coupling_node_path("Nektar/Coupling");
  NESOASSERT(session->DefinesElement(coupling_node_path),
             "No Coupling node found in session; bailing out.");
  TiXmlElement *coupling_node = session->GetElement(coupling_node_path);
  NESOASSERT(coupling_node->Attribute("TYPE"),
             "Missing TYPE attribute in Coupling");
  std::string coupling_type = coupling_node->Attribute("TYPE");
  NESOASSERT(!coupling_type.empty(),
             "'COUPLING' xml node must have a  non-empty 'TYPE' attribute.");

  return SU::GetCouplingFactory().CreateInstance(coupling_type, field);
}
} // namespace NESO::Solvers::Diffusion
#endif // __NESOSOLVERS_DIFFUSION_CWIPIUTILS_HPP__