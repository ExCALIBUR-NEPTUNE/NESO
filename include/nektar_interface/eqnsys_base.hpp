#ifndef __EQNSYS_BASE_H_
#define __EQNSYS_BASE_H_

#include <SolverUtils/EquationSystem.h>

#include <type_traits>

#include "nektar_interface/partsys_base.hpp"
#include "nektar_interface/utilities.hpp"

namespace LU = Nektar::LibUtilities;
namespace SD = Nektar::SpatialDomains;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers {

// Fwd declare NEKEQNSYS, PARTSYS as classes
class NEKEQNSYS;
class PARTSYS;

template <typename NEKEQNSYS, typename PARTSYS>
class EqnSysBase : public NEKEQNSYS {
  // Template param must derive from Nektar's EquationSystem base class
  static_assert(std::is_base_of<SU::EquationSystem, NEKEQNSYS>(),
                "Template arg to EqnSysBase must derive from "
                "Nektar::SolverUtils::EquationSystem");
  // Particle system must derive from PartSysBase
  static_assert(
      std::is_base_of<PartSysBase, PARTSYS>(),
      "PARTSYS template arg to EqnSysBase must derive from PartSysBase");

public:
  /// Cleanup particle system on destruction
  virtual ~EqnSysBase() {}

protected:
  EqnSysBase(const LU::SessionReaderSharedPtr &session,
             const SD::MeshGraphSharedPtr &graph)
      : NEKEQNSYS(session, graph), m_field_to_index(session->GetVariables()),
        m_required_flds() {}

  /// Particle system
  std::shared_ptr<PARTSYS> particle_sys;

  /// Field name => index mapper
  NESO::NektarFieldIndexMap m_field_to_index;
  /// List of field names required by the solver
  std::vector<std::string> m_required_flds;

  virtual void load_params(){};

  /**
   * @brief Check required fields are all defined and have the same number of
   * quad points
   */
  void validate_fields() {
    int npts_exp = NEKEQNSYS::GetNpoints();
    for (auto &fld_name : m_required_flds) {
      int idx = m_field_to_index.get_idx(fld_name);
      // Check field exists

      std::string err_msg = "Required field [" + fld_name + "] is not defined.";
      NESOASSERT(idx >= 0, err_msg.c_str());

      // Check fields all have the same number of quad points
      int npts = this->m_fields[idx]->GetNpoints();
      err_msg = "Expecting " + std::to_string(npts_exp) +
                " quad points, but field '" + fld_name + "' has " +
                std::to_string(npts) +
                ". Check NUMMODES is the same for all required fields.";
      NESOASSERT(npts == npts_exp, err_msg.c_str());
    }
  }

  virtual void v_InitObject(bool create_fields) override {
    NEKEQNSYS::v_InitObject(create_fields);

    // Ensure that the session file defines all required variables
    validate_fields();

    // Load parameters
    load_params();
  }
};

} // namespace NESO::Solvers
#endif