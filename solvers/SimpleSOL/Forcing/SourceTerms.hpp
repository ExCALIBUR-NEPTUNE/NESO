#ifndef __NESOSOLVERS_SIMPLESOL_SOURCETERMS_HPP__
#define __NESOSOLVERS_SIMPLESOL_SOURCETERMS_HPP__

#include <SolverUtils/Forcing/Forcing.h>
#include <nektar_interface/utilities.hpp>

namespace LU = Nektar::LibUtilities;
namespace MR = Nektar::MultiRegions;
namespace SD = Nektar::SpatialDomains;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::SimpleSOL {

class SourceTerms : public SU::Forcing {
public:
  friend class Nektar::MemoryManager<SourceTerms>;

  /// Creates an instance of this class
  static SU::ForcingSharedPtr
  create(const LU::SessionReaderSharedPtr &session,
         const std::weak_ptr<SU::EquationSystem> &equation_sys,
         const Array<OneD, MR::ExpListSharedPtr> &fields,
         const unsigned int &num_src_fields,
         const TiXmlElement *force_xml_node) {
    SU::ForcingSharedPtr forcing_obj =
        Nektar::MemoryManager<SourceTerms>::AllocateSharedPtr(session,
                                                              equation_sys);
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

  // Field indices
  NektarFieldIndexMap field_to_index;

  // Angle between source orientation and x-axis
  NekDouble theta;

  // Pre-computed (1D) coord along source-oriented axis
  Array<OneD, NekDouble> s;

  // Source parameters
  NekDouble E_prefac;
  NekDouble mu;
  NekDouble rho_prefac;
  NekDouble sigma;
  NekDouble smax;
  NekDouble u_prefac;
};

} // namespace NESO::Solvers::SimpleSOL

#endif // __NESOSOLVERS_SIMPLESOL_SOURCETERMS_HPP__
