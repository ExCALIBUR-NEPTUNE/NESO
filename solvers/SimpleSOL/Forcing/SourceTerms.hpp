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
         const std::weak_ptr<SU::EquationSystem> &pEquation,
         const Array<OneD, MR::ExpListSharedPtr> &pFields,
         const unsigned int &pNumForcingFields, const TiXmlElement *pForce) {
    SU::ForcingSharedPtr p =
        MemoryManager<SourceTerms>::AllocateSharedPtr(session, pEquation);
    p->InitObject(pFields, pNumForcingFields, pForce);
    return p;
  }

  /// Name of the class
  static std::string class_name;

protected:
  virtual void v_InitObject(const Array<OneD, MR::ExpListSharedPtr> &pFields,
                            const unsigned int &pNumForcingFields,
                            const TiXmlElement *pForce) override;

  virtual void v_Apply(const Array<OneD, MR::ExpListSharedPtr> &fields,
                       const Array<OneD, Array<OneD, NekDouble>> &inarray,
                       Array<OneD, Array<OneD, NekDouble>> &outarray,
                       const NekDouble &time) override;

private:
  SourceTerms(const LU::SessionReaderSharedPtr &session,
              const std::weak_ptr<SU::EquationSystem> &pEquation);

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
