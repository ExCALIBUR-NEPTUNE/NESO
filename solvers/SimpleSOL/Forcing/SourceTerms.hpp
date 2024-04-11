#ifndef __SIMPLESOL_SOURCETERMS_H_
#define __SIMPLESOL_SOURCETERMS_H_

#include "nektar_interface/utilities.hpp"
#include <SolverUtils/Forcing/Forcing.h>

namespace LU = Nektar::LibUtilities;
namespace MR = Nektar::MultiRegions;
namespace SD = Nektar::SpatialDomains;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers {

class SourceTerms : public SU::Forcing {
public:
  friend class MemoryManager<SourceTerms>;

  /// Creates an instance of this class
  static SU::ForcingSharedPtr
  create(const LU::SessionReaderSharedPtr &session,
         const std::weak_ptr<SU::EquationSystem> &equation_sys,
         const Array<OneD, MR::ExpListSharedPtr> &fields,
         const unsigned int &num_src_fields,
         const TiXmlElement *force_xml_node) {
    SU::ForcingSharedPtr forcing_obj =
        MemoryManager<SourceTerms>::AllocateSharedPtr(session, equation_sys);
    forcing_obj->InitObject(fields, num_src_fields, force_xml_node);
    return forcing_obj;
  }

  /// Name of the class
  static std::string class_name;

protected:
  virtual void v_InitObject(const Array<OneD, MR::ExpListSharedPtr> &fields,
                            const unsigned int &num_src_fields,
                            const TiXmlElement *force_xml_node) override;

  virtual void v_Apply(const Array<OneD, MR::ExpListSharedPtr> &fields,
                       const Array<OneD, Array<OneD, NekDouble>> &in_arr,
                       Array<OneD, Array<OneD, NekDouble>> &out_arr,
                       const NekDouble &time) override;

private:
  SourceTerms(const LU::SessionReaderSharedPtr &session,
              const std::weak_ptr<SU::EquationSystem> &equation_sys);

  // Angle between source orientation and x-axis
  NekDouble m_theta;
  // Pre-computed coords along source-oriented axis
  Array<OneD, NekDouble> m_s;

  NektarFieldIndexMap field_to_index;

  // Source parameters
  NekDouble m_smax;
  NekDouble m_mu;
  NekDouble m_sigma;
  NekDouble m_rho_prefac;
  NekDouble m_u_prefac;
  NekDouble m_E_prefac;
};

} // namespace NESO::Solvers

#endif
