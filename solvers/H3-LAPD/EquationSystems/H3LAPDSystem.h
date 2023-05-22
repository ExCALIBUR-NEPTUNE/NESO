///////////////////////////////////////////////////////////////////////////////
//
// File H3LAPDSystem.h
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
// Description: Header for the Hermes-3 LAPD equation system
//
///////////////////////////////////////////////////////////////////////////////

#ifndef H3LAPDSYSTEM_H
#define H3LAPDSYSTEM_H

#include "nektar_interface/utilities.hpp"

#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <SolverUtils/AdvectionSystem.h>
#include <SolverUtils/EquationSystem.h>
#include <SolverUtils/Forcing/Forcing.h>

namespace Nektar {

class H3LAPDSystem : virtual public SolverUtils::AdvectionSystem {
public:
  friend class MemoryManager<H3LAPDSystem>;

  /// Name of class.
  static std::string className;

  /// Creates an instance of this class.
  static SolverUtils::EquationSystemSharedPtr
  create(const LibUtilities::SessionReaderSharedPtr &pSession,
         const SpatialDomains::MeshGraphSharedPtr &pGraph) {
    SolverUtils::EquationSystemSharedPtr p =
        MemoryManager<H3LAPDSystem>::AllocateSharedPtr(pSession, pGraph);
    p->InitObject();
    return p;
  }

  /// Default destructor.
  virtual ~H3LAPDSystem() = default;

protected:
  /// Protected constructor. Since we use a factory pattern, objects should be
  /// constructed via the SolverUtils::EquationSystem factory.
  H3LAPDSystem(const LibUtilities::SessionReaderSharedPtr &pSession,
               const SpatialDomains::MeshGraphSharedPtr &pGraph);

  // Field name => index mapper
  NESO::NektarFieldIndexMap m_field_to_index;
  // List of field names required by the solver
  std::vector<std::string> m_required_flds;

  // Forcing/source terms
  std::vector<SolverUtils::ForcingSharedPtr> m_forcing;

  void ExplicitTimeInt(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                       Array<OneD, Array<OneD, NekDouble>> &outarray,
                       const NekDouble time);

  void GetFluxVector(const Array<OneD, Array<OneD, NekDouble>> &physfield,
                     Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);

  void GetFluxVectorDiff(
      const Array<OneD, Array<OneD, NekDouble>> &inarray,
      const Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &qfield,
      Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &viscousTensor);

  Array<OneD, NekDouble> &GetNormalVelocity();

  virtual void v_InitObject(bool DeclareField) override;

  void ValidateFieldList();
};

} // namespace Nektar
#endif
