///////////////////////////////////////////////////////////////////////////////
//
// File SOLSystem.h
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
// Description: Auxiliary functions for the 1D SOL system
//
///////////////////////////////////////////////////////////////////////////////

#ifndef SOLSYSTEM_H
#define SOLSYSTEM_H

#include "nektar_interface/utilities.hpp"

#include <CompressibleFlowSolver/Misc/VariableConverter.h>
#include <LocalRegions/Expansion2D.h>
#include <LocalRegions/Expansion3D.h>
#include <MultiRegions/GlobalMatrixKey.h>
#include <SolverUtils/Advection/Advection.h>
#include <SolverUtils/Diffusion/Diffusion.h>
#include <SolverUtils/Filters/FilterInterfaces.hpp>
#include <SolverUtils/Forcing/Forcing.h>
#include <SolverUtils/RiemannSolvers/RiemannSolver.h>
#include <SolverUtils/UnsteadySystem.h>
#include <boost/core/ignore_unused.hpp>
namespace Nektar {

class SOLSystem : virtual public SolverUtils::UnsteadySystem,
                  virtual public SolverUtils::FluidInterface {
public:
  friend class MemoryManager<SOLSystem>;

  /// Creates an instance of this class.
  static SolverUtils::EquationSystemSharedPtr
  create(const LibUtilities::SessionReaderSharedPtr &pSession,
         const SpatialDomains::MeshGraphSharedPtr &pGraph) {
    SolverUtils::EquationSystemSharedPtr p =
        MemoryManager<SOLSystem>::AllocateSharedPtr(pSession, pGraph);
    p->InitObject();
    return p;
  }

  /// Name of class.
  static std::string className;

  virtual ~SOLSystem();

  virtual void
  GetDensity(const Array<OneD, const Array<OneD, NekDouble>> &physfield,
             Array<OneD, NekDouble> &density) override final;

  /// Function to get estimate of min h/p factor per element
  Array<OneD, NekDouble> GetElmtMinHP(void);

  virtual void
  GetPressure(const Array<OneD, const Array<OneD, NekDouble>> &physfield,
              Array<OneD, NekDouble> &pressure) override final;

  virtual void
  GetVelocity(const Array<OneD, const Array<OneD, NekDouble>> &physfield,
              Array<OneD, Array<OneD, NekDouble>> &velocity) override final;

  virtual bool HasConstantDensity() override final { return false; }

protected:
  SOLSystem(const LibUtilities::SessionReaderSharedPtr &pSession,
            const SpatialDomains::MeshGraphSharedPtr &pGraph);

  SolverUtils::AdvectionSharedPtr m_advObject;
  SolverUtils::DiffusionSharedPtr m_diffusion;
  NESO::NektarFieldIndexMap m_field_to_index;
  // Forcing term
  std::vector<SolverUtils::ForcingSharedPtr> m_forcing;
  NekDouble m_gamma;
  /// Names of fields that will be time integrated
  std::vector<std::string> m_int_fld_names;
  // List of field names required by the solver
  std::vector<std::string> m_required_flds;
  // Auxiliary object to convert variables
  VariableConverterSharedPtr m_varConv;
  Array<OneD, Array<OneD, NekDouble>> m_vecLocs;

  void DoAdvection(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                   Array<OneD, Array<OneD, NekDouble>> &outarray,
                   const NekDouble time,
                   const Array<OneD, const Array<OneD, NekDouble>> &pFwd,
                   const Array<OneD, const Array<OneD, NekDouble>> &pBwd);

  void DoOdeProjection(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                       Array<OneD, Array<OneD, NekDouble>> &outarray,
                       const NekDouble time);

  virtual void
  DoOdeRhs(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
           Array<OneD, Array<OneD, NekDouble>> &outarray, const NekDouble time);

  void GetElmtTimeStep(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                       Array<OneD, NekDouble> &tstep);

  void GetFluxVector(const Array<OneD, const Array<OneD, NekDouble>> &physfield,
                     TensorOfArray3D<NekDouble> &flux);

  NekDouble GetGamma() { return m_gamma; }

  const Array<OneD, const Array<OneD, NekDouble>> &GetNormals() {
    return m_traceNormals;
  }

  const Array<OneD, const Array<OneD, NekDouble>> &GetVecLocs() {
    return m_vecLocs;
  }

  void InitAdvection();

  // virtual void v_AppendOutput1D(
  //     Array<OneD, Array<OneD, NekDouble>> &solution1D) override final{};

  // virtual Array<OneD, NekDouble>
  // v_GetMaxStdVelocity(const NekDouble SpeedSoundFactor) override final;

  // virtual NekDouble v_GetTimeStep(
  //     const Array<OneD, const Array<OneD, NekDouble>> &inarray) override
  //     final;

  virtual void v_InitObject(bool DeclareField) override;
  void ValidateFieldList();
};

} // namespace Nektar
#endif
