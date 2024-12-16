#ifndef __SOLVER_IMPLICITHELPER_H_
#define __SOLVER_IMPLICITHELPER_H_

#include <LibUtilities/LinearAlgebra/NekNonlinSysIter.h>

namespace LU = Nektar::LibUtilities;
namespace MR = Nektar::MultiRegions;

namespace NESO::Solvers {

class ImplicitHelper {
public:
  ImplicitHelper(LU::SessionReaderSharedPtr session,
                 Array<OneD, MR::ExpListSharedPtr> fields,
                 LU::TimeIntegrationSchemeOperators &ode, int nFields)
      : m_session(session), m_fields(fields), m_ode(ode), m_nFields(nFields) {
    m_comm = session->GetComm()->GetSpaceComm();
  }

  /// Default destructor.
  ~ImplicitHelper() = default;

public:
  void ImplicitTimeInt(const Array<OneD, const Array<OneD, NekDouble>> &inpnts,
                       Array<OneD, Array<OneD, NekDouble>> &outpnt,
                       const NekDouble time, const NekDouble lambda) {
    m_TimeIntegLambda = lambda;
    m_bndEvaluateTime = time;
    unsigned int npoints = m_fields[0]->GetNpoints();

    Array<OneD, NekDouble> inarray(m_nFields * npoints);
    Array<OneD, NekDouble> outarray(m_nFields * npoints);
    Array<OneD, NekDouble> tmp;

    for (int i = 0; i < m_nFields; ++i) {
      int noffset = i * npoints;
      Vmath::Vcopy(npoints, inpnts[i], 1, tmp = inarray + noffset, 1);
    }

    ImplicitTimeInt1D(inarray, outarray);

    for (int i = 0; i < m_nFields; ++i) {
      int noffset = i * npoints;
      Vmath::Vcopy(npoints, outarray + noffset, 1, outpnt[i], 1);
    }
  }

  void InitialiseNonlinSysSolver() {
    int ntotal = m_nFields * m_fields[0]->GetNpoints();

    // Create the key to hold settings for nonlin solver
    LU::NekSysKey key = LU::NekSysKey();

    // Load required LinSys parameters:
    m_session->LoadParameter("NekLinSysMaxIterations",
                             key.m_NekLinSysMaxIterations, 30);
    m_session->LoadParameter("LinSysMaxStorage", key.m_LinSysMaxStorage, 30);
    m_session->LoadParameter("LinSysRelativeTolInNonlin",
                             key.m_NekLinSysTolerance, 5.0E-2);
    m_session->LoadParameter("GMRESMaxHessMatBand", key.m_KrylovMaxHessMatBand,
                             31);

    // Load required NonLinSys parameters:
    m_session->LoadParameter("JacobiFreeEps", m_jacobiFreeEps, 5.0E-8);
    m_session->LoadParameter("NekNonlinSysMaxIterations",
                             key.m_NekNonlinSysMaxIterations, 10);
    m_session->LoadParameter("NewtonRelativeIteTol",
                             key.m_NekNonLinSysTolerance, 1.0E-12);
    WARNINGL0(!m_session->DefinesParameter("NewtonAbsoluteIteTol"),
              "Please specify NewtonRelativeIteTol instead of "
              "NewtonAbsoluteIteTol in XML session file");
    m_session->LoadParameter("NonlinIterTolRelativeL2",
                             key.m_NonlinIterTolRelativeL2, 1.0E-3);
    m_session->LoadSolverInfo("LinSysIterSolverTypeInNonlin",
                              key.m_LinSysIterSolverTypeInNonlin, "GMRES");

    // Set up operators
    LU::NekSysOperators nekSysOp;
    nekSysOp.DefineNekSysResEval(&ImplicitHelper::NonlinSysEvaluator1D, this);
    nekSysOp.DefineNekSysLhsEval(&ImplicitHelper::MatrixMultiplyMatrixFree,
                                 this);
    nekSysOp.DefineNekSysPrecon(&ImplicitHelper::DoNullPrecon, this);

    // Initialize non-linear system
    m_nonlinsol = LU::GetNekNonlinSysIterFactory().CreateInstance(
        "Newton", m_session, m_comm->GetRowComm(), ntotal, key);
    m_nonlinsol->SetSysOperators(nekSysOp);
  }

protected:
  // Implicit solver parameters
  int m_nFields = 0;
  int m_TotNewtonIts = 0;
  int m_TotLinIts = 0;
  int m_TotImpStages = 0;
  NekDouble m_jacobiFreeEps = 5.0E-08;
  NekDouble m_bndEvaluateTime = 0.0;
  NekDouble m_TimeIntegLambda = 0.0;
  NekDouble m_inArrayNorm = -1.0;

  LU::SessionReaderSharedPtr m_session;
  LU::NekNonlinSysIterSharedPtr m_nonlinsol;
  LU::CommSharedPtr m_comm;
  LU::TimeIntegrationSchemeOperators &m_ode;

  Array<OneD, MR::ExpListSharedPtr> m_fields;

  void ImplicitTimeInt1D(const Array<OneD, const NekDouble> &inarray,
                         Array<OneD, NekDouble> &out) {
    CalcRefValues(inarray);
    m_nonlinsol->SetRhsMagnitude(m_inArrayNorm);
    m_TotNewtonIts += m_nonlinsol->SolveSystem(inarray.size(), inarray, out, 0);
    m_TotLinIts += m_nonlinsol->GetNtotLinSysIts();
    m_TotImpStages++;
  }

  void CalcRefValues(const Array<OneD, const NekDouble> &inarray) {
    unsigned int npoints = m_fields[0]->GetNpoints();

    Array<OneD, NekDouble> magnitdEstimat(m_nFields, 0.0);

    for (int i = 0; i < m_nFields; ++i) {
      int offset = i * npoints;
      magnitdEstimat[i] =
          Vmath::Dot(npoints, inarray + offset, inarray + offset);
    }
    m_comm->GetSpaceComm()->AllReduce(magnitdEstimat, Nektar::LU::ReduceSum);

    m_inArrayNorm = 0.0;
    for (int i = 0; i < m_nFields; ++i) {
      m_inArrayNorm += magnitdEstimat[i];
    }
  }

  void NonlinSysEvaluator1D(const Array<OneD, const NekDouble> &inarray,
                            Array<OneD, NekDouble> &out,
                            [[maybe_unused]] const bool &flag) {
    unsigned int npoints = m_fields[0]->GetNpoints();
    Array<OneD, Array<OneD, NekDouble>> in2D(m_nFields);
    Array<OneD, Array<OneD, NekDouble>> out2D(m_nFields);
    for (int i = 0; i < m_nFields; ++i) {
      int offset = i * npoints;
      in2D[i] = inarray + offset;
      out2D[i] = out + offset;
    }
    NonlinSysEvaluator(in2D, out2D);
  }

  void
  NonlinSysEvaluator(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                     Array<OneD, Array<OneD, NekDouble>> &out) {
    unsigned int npoints = m_fields[0]->GetNpoints();
    Array<OneD, Array<OneD, NekDouble>> inpnts(m_nFields);
    for (int i = 0; i < m_nFields; ++i) {
      inpnts[i] = Array<OneD, NekDouble>(npoints, 0.0);
    }

    m_ode.DoProjection(inarray, inpnts, m_bndEvaluateTime);
    m_ode.DoOdeRhs(inpnts, out, m_bndEvaluateTime);

    for (int i = 0; i < m_nFields; ++i) {
      Vmath::Svtvp(npoints, -m_TimeIntegLambda, out[i], 1, inarray[i], 1,
                   out[i], 1);
      Vmath::Vsub(npoints, out[i], 1,
                  m_nonlinsol->GetRefSourceVec() + i * npoints, 1, out[i], 1);
    }
  }

  void MatrixMultiplyMatrixFree(const Array<OneD, const NekDouble> &inarray,
                                Array<OneD, NekDouble> &out,
                                [[maybe_unused]] const bool &flag) {
    const Array<OneD, const NekDouble> solref = m_nonlinsol->GetRefSolution();
    const Array<OneD, const NekDouble> resref = m_nonlinsol->GetRefResidual();

    unsigned int ntotal = inarray.size();
    NekDouble magninarray = Vmath::Dot(ntotal, inarray, inarray);
    m_comm->GetSpaceComm()->AllReduce(magninarray, Nektar::LU::ReduceSum);
    NekDouble eps =
        m_jacobiFreeEps * sqrt((sqrt(m_inArrayNorm) + 1.0) / magninarray);

    Array<OneD, NekDouble> solplus{ntotal};
    Array<OneD, NekDouble> resplus{ntotal};

    Vmath::Svtvp(ntotal, eps, inarray, 1, solref, 1, solplus, 1);
    NonlinSysEvaluator1D(solplus, resplus, flag);
    Vmath::Vsub(ntotal, resplus, 1, resref, 1, out, 1);
    Vmath::Smul(ntotal, 1.0 / eps, out, 1, out, 1);
  }

  void DoNullPrecon(const Array<OneD, const NekDouble> &inarray,
                    Array<OneD, NekDouble> &outarray, const bool &flag) {
    Vmath::Vcopy(inarray.size(), inarray, 1, outarray, 1);
  }
};

} // namespace NESO::Solvers

#endif
