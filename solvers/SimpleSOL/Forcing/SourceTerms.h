///////////////////////////////////////////////////////////////////////////////
//
// File: SourceTerms.h
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description: Forcing for axi-symmetric flow.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_SOLVERUTILS_FORCINGAXISYM
#define NEKTAR_SOLVERUTILS_FORCINGAXISYM

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
  create(const LU::SessionReaderSharedPtr &pSession,
         const std::weak_ptr<SU::EquationSystem> &pEquation,
         const Array<OneD, MR::ExpListSharedPtr> &pFields,
         const unsigned int &pNumForcingFields, const TiXmlElement *pForce) {
    SU::ForcingSharedPtr p =
        MemoryManager<SourceTerms>::AllocateSharedPtr(pSession, pEquation);
    p->InitObject(pFields, pNumForcingFields, pForce);
    return p;
  }

  /// Name of the class
  static std::string className;

protected:
  virtual void v_InitObject(const Array<OneD, MR::ExpListSharedPtr> &pFields,
                            const unsigned int &pNumForcingFields,
                            const TiXmlElement *pForce);

  virtual void v_Apply(const Array<OneD, MR::ExpListSharedPtr> &fields,
                       const Array<OneD, Array<OneD, NekDouble>> &inarray,
                       Array<OneD, Array<OneD, NekDouble>> &outarray,
                       const NekDouble &time);

private:
  SourceTerms(const LU::SessionReaderSharedPtr &pSession,
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
