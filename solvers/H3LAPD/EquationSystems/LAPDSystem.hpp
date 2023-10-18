///////////////////////////////////////////////////////////////////////////////
//
// File LAPDSystem.hpp
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
// Description: LAPD equation system.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef LAPDSYSTEM_H
#define LAPDSYSTEM_H

#include "DriftReducedSystem.hpp"
#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <SolverUtils/EquationSystem.h>

namespace Nektar {

class LAPDSystem : virtual public DriftReducedSystem {
public:
  friend class MemoryManager<LAPDSystem>;

  /// Name of class.
  static std::string className;

  /// Creates an instance of this class.
  static SolverUtils::EquationSystemSharedPtr
  create(const LibUtilities::SessionReaderSharedPtr &pSession,
         const SpatialDomains::MeshGraphSharedPtr &pGraph) {
    SolverUtils::EquationSystemSharedPtr p =
        MemoryManager<LAPDSystem>::AllocateSharedPtr(pSession, pGraph);
    p->InitObject();
    return p;
  }

protected:
  LAPDSystem(const LibUtilities::SessionReaderSharedPtr &pSession,
             const SpatialDomains::MeshGraphSharedPtr &pGraph);

  void
  AddCollisionTerms(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                    Array<OneD, Array<OneD, NekDouble>> &outarray);

  void AddEParTerms(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                    Array<OneD, Array<OneD, NekDouble>> &outarray);
  void AddGradPTerms(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                     Array<OneD, Array<OneD, NekDouble>> &outarray);

  void CalcCollisionFreqs(const Array<OneD, NekDouble> &ne,
                          Array<OneD, NekDouble> &coeffs);
  void CalcCoulombLogarithm(const Array<OneD, NekDouble> &ne,
                            Array<OneD, NekDouble> &LogLambda);
  virtual void CalcEAndAdvVels(
      const Array<OneD, const Array<OneD, NekDouble>> &inarray) override;

  virtual void
  ExplicitTimeInt(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                  Array<OneD, Array<OneD, NekDouble>> &outarray,
                  const NekDouble time) override;

  void
  GetFluxVectorIons(const Array<OneD, Array<OneD, NekDouble>> &physfield,
                    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);

  void GetFluxVectorPD(const Array<OneD, Array<OneD, NekDouble>> &physfield,
                       Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &flux);

  virtual void
  GetPhiSolveRHS(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                 Array<OneD, NekDouble> &rhs) override;
  Array<OneD, NekDouble> &GetVnAdvIons();
  Array<OneD, NekDouble> &GetVnAdvPD();

  virtual void LoadParams() override;

  virtual void v_InitObject(bool DeclareField) override;

  // Charge unit
  NekDouble m_charge_e;
  // Ion mass;
  NekDouble m_md;
  // Electron mass;
  NekDouble m_me;
  // Ion temperature in eV
  NekDouble m_Td;
  // Electron temperature in eV
  NekDouble m_Te;
  //---------------------------------------------------------------------------
  // Factors used in collision coeff calculation
  // Density-independent part of the Coulomb logarithm; read from config
  NekDouble m_coulomb_log_const;
  // Pre-factor used when calculating collision frequencies; read from config
  NekDouble m_nu_ei_const;
  // Factor to convert densities (back) to SI; used in Coulomb logarithm calc
  NekDouble m_n_to_SI;
  //---------------------------------------------------------------------------
  // Advection objects
  SolverUtils::AdvectionSharedPtr m_advIons;
  SolverUtils::AdvectionSharedPtr m_advPD;
  // Riemann solver objects
  SolverUtils::RiemannSolverSharedPtr m_riemannSolverIons;
  SolverUtils::RiemannSolverSharedPtr m_riemannSolverPD;
  // Storage for advection velocities dotted with element face normals
  Array<OneD, NekDouble> m_traceVnIons;
  Array<OneD, NekDouble> m_traceVnPD;
  // Storage for ion advection velocities
  Array<OneD, Array<OneD, NekDouble>> m_vAdvIons;
  /*Storage for difference between elec, ion parallel velocities. Has size ndim
   so that it can be used in advection operation; non-parallel values are 0
   */
  Array<OneD, Array<OneD, NekDouble>> m_vAdvDiffPar;
  // Storage for electron, ion parallel velocities
  Array<OneD, NekDouble> m_vParIons;
};

} // namespace Nektar
#endif
