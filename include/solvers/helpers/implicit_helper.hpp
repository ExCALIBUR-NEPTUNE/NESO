#ifndef __SOLVER_IMPLICITHELPER_H_
#define __SOLVER_IMPLICITHELPER_H_

#include <LibUtilities/LinearAlgebra/NekNonlinSysIter.h>

namespace LU = Nektar::LibUtilities;
namespace MR = Nektar::MultiRegions;

namespace NESO::Solvers {

class ImplicitHelper {
public:
  ImplicitHelper(LU::SessionReaderSharedPtr session_in,
                 Array<OneD, MR::ExpListSharedPtr> fields_in,
                 LU::TimeIntegrationSchemeOperators &ode_in, int n_fields_in)
      : session(session_in), fields(fields_in), ode(ode_in),
        n_fields(n_fields_in), projected_in_arr(n_fields_in) {
    this->comm = session->GetComm()->GetSpaceComm();

    // Set n_pts, then make sure all fields have the same points
    this->n_pts = this->fields[0]->GetNpoints();
    for (auto ifld = 0; ifld < this->fields.size(); ifld++) {
      NESOASSERT(this->fields[ifld]->GetNpoints() == this->n_pts,
                 "ImplicitHelper: Found fields with different number of quad "
                 "points; this class assumes point arrays of equal size.");
    }

    // Allocate storage used in non_lin_sys_evaluator
    for (int ifld = 0; ifld < this->n_fields; ++ifld) {
      this->projected_in_arr[ifld] = Array<OneD, NekDouble>(this->n_pts);
    }
  }

  /// Default destructor.
  ~ImplicitHelper() = default;

public:
  void
  implicit_time_int(const Array<OneD, const Array<OneD, NekDouble>> &in_pts,
                    Array<OneD, Array<OneD, NekDouble>> &out_pts,
                    const NekDouble time, const NekDouble lambda) {
    this->time_int_lambda = lambda;
    this->time = time;

    /*
     * Copy in/out points for each field into flattened arrays of size
     * N_pts_per_field*N_fields
     */
    Array<OneD, NekDouble> in_arr_flat(this->n_fields * this->n_pts);
    Array<OneD, NekDouble> out_arr_flat(this->n_fields * this->n_pts);
    Array<OneD, NekDouble> tmp;
    for (auto ifld = 0; ifld < this->n_fields; ++ifld) {
      int offset = ifld * this->n_pts;
      Vmath::Vcopy(this->n_pts, in_pts[ifld], 1, tmp = in_arr_flat + offset, 1);
    }

    // Pass flattened arrays to solver
    implicit_time_int_1D(in_arr_flat, out_arr_flat);

    // Put the result back into a field-indexed output array
    for (auto ifld = 0; ifld < this->n_fields; ++ifld) {
      int offset = ifld * this->n_pts;
      Vmath::Vcopy(this->n_pts, out_arr_flat + offset, 1, out_pts[ifld], 1);
    }
  }

  void init_non_lin_sys_solver() {

    // Create a key to hold settings for the non-linear solver
    LU::NekSysKey key = LU::NekSysKey();

    // Load parameters for the linear solver
    this->session->LoadParameter("NekLinSysMaxIterations",
                                 key.m_NekLinSysMaxIterations, 30);
    this->session->LoadParameter("LinSysMaxStorage", key.m_LinSysMaxStorage,
                                 30);
    this->session->LoadParameter("LinSysRelativeTolInNonlin",
                                 key.m_NekLinSysTolerance, 5.0E-2);
    this->session->LoadParameter("GMRESMaxHessMatBand",
                                 key.m_KrylovMaxHessMatBand, 31);

    // Load parameters for the non-linear solver
    this->session->LoadParameter("JacobiFreeEps", this->jacobi_free_eps,
                                 5.0E-8);
    this->session->LoadParameter("NekNonlinSysMaxIterations",
                                 key.m_NekNonlinSysMaxIterations, 10);
    this->session->LoadParameter("NewtonRelativeIteTol",
                                 key.m_NekNonLinSysTolerance, 1.0E-12);
    WARNINGL0(!this->session->DefinesParameter("NewtonAbsoluteIteTol"),
              "Please specify NewtonRelativeIteTol instead of "
              "NewtonAbsoluteIteTol in XML session file");
    this->session->LoadParameter("NonlinIterTolRelativeL2",
                                 key.m_NonlinIterTolRelativeL2, 1.0E-3);
    this->session->LoadSolverInfo("LinSysIterSolverTypeInNonlin",
                                  key.m_LinSysIterSolverTypeInNonlin, "GMRES");

    // Set up operators
    LU::NekSysOperators operators;
    operators.DefineNekSysResEval(&ImplicitHelper::non_lin_sys_evaluator_1D,
                                  this);
    operators.DefineNekSysLhsEval(&ImplicitHelper::matrix_multiply_MF, this);
    operators.DefineNekSysPrecon(&ImplicitHelper::do_null_precon, this);

    // Initialize non-linear system
    int n_pts_all_fields = this->n_fields * this->n_pts;
    this->non_lin_solver = LU::GetNekNonlinSysIterFactory().CreateInstance(
        "Newton", this->session, this->comm->GetRowComm(), n_pts_all_fields,
        key);
    this->non_lin_solver->SetSysOperators(operators);
  }

protected:
  // Implicit solver parameters
  NekDouble array_norm = -1.0;
  NekDouble jacobi_free_eps = 5.0E-08;
  int n_fields = 0;
  NekDouble time = 0.0;
  NekDouble time_int_lambda = 0.0;
  int tot_imp_stages = 0;
  int tot_lin_its = 0;
  int tot_newton_its = 0;

  // Store number of points per field as a member var for convenience
  unsigned int n_pts;

  // Pointers to external objects, passed in at construction
  LU::CommSharedPtr comm;
  Array<OneD, MR::ExpListSharedPtr> fields;
  LU::NekNonlinSysIterSharedPtr non_lin_solver;
  LU::TimeIntegrationSchemeOperators &ode;
  LU::SessionReaderSharedPtr session;

  // Storage used in non_lin_sys_evaluator
  Array<OneD, Array<OneD, NekDouble>> projected_in_arr;

  /**
   * Calculate array norm.
   */
  void calc_ref_vals(const Array<OneD, const NekDouble> &in_arr) {

    Array<OneD, NekDouble> fld_array_norms(this->n_fields, 0.0);

    // Compute array norm for each field
    for (auto ifld = 0; ifld < this->n_fields; ++ifld) {
      int offset = ifld * this->n_pts;
      fld_array_norms[ifld] =
          Vmath::Dot(this->n_pts, in_arr + offset, in_arr + offset);
    }
    this->comm->GetSpaceComm()->AllReduce(fld_array_norms, LU::ReduceSum);

    // Sum over all fields
    this->array_norm = 0.0;
    for (auto ifld = 0; ifld < this->n_fields; ++ifld) {
      this->array_norm += fld_array_norms[ifld];
    }
  }

  /** No-op preconditioner. */
  void do_null_precon(const Array<OneD, const NekDouble> &in_arr,
                      Array<OneD, NekDouble> &out_arr, const bool &flag) {
    Vmath::Vcopy(in_arr.size(), in_arr, 1, out_arr, 1);
  }

  /**
   * Solve the non-linear system.
   */
  void implicit_time_int_1D(const Array<OneD, const NekDouble> &in_arr,
                            Array<OneD, NekDouble> &out) {
    calc_ref_vals(in_arr);
    this->non_lin_solver->SetRhsMagnitude(this->array_norm);
    this->tot_newton_its +=
        this->non_lin_solver->SolveSystem(in_arr.size(), in_arr, out, 0);
    this->tot_lin_its += this->non_lin_solver->GetNtotLinSysIts();
    this->tot_imp_stages++;
  }

  /**
   * Roll up flattened in/out arrays and pass them to the system evaluator.
   * Used by the Matrix-Free operator.
   */
  void non_lin_sys_evaluator_1D(const Array<OneD, const NekDouble> &in_arr_flat,
                                Array<OneD, NekDouble> &out_arr_flat,
                                [[maybe_unused]] const bool &flag) {
    // Copy in/out values in field-indexed arrays
    Array<OneD, Array<OneD, NekDouble>> in_arr(this->n_fields);
    Array<OneD, Array<OneD, NekDouble>> out_arr(this->n_fields);
    for (auto ifld = 0; ifld < this->n_fields; ++ifld) {
      int offset = ifld * this->n_pts;
      in_arr[ifld] = in_arr_flat + offset;
      out_arr[ifld] = out_arr_flat + offset;
    }
    non_lin_sys_evaluator(in_arr, out_arr);
  }

  /**
   * Do implicit step.
   */
  void
  non_lin_sys_evaluator(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                        Array<OneD, Array<OneD, NekDouble>> &out_arr) {

    // Zero temporary array
    for (int ifld = 0; ifld < this->n_fields; ++ifld) {
      Vmath::Zero(this->n_pts, this->projected_in_arr[ifld], 1);
    }

    // Do projection, then evaluate RHS
    this->ode.DoProjection(in_arr, this->projected_in_arr, this->time);
    this->ode.DoOdeRhs(this->projected_in_arr, out_arr, this->time);

    for (int ifld = 0; ifld < this->n_fields; ++ifld) {
      // u_{i+1} = u_i - lambda*rhs
      Vmath::Svtvp(this->n_pts, -this->time_int_lambda, out_arr[ifld], 1,
                   in_arr[ifld], 1, out_arr[ifld], 1);
      // Subtract reference source vector
      Vmath::Vsub(this->n_pts, out_arr[ifld], 1,
                  this->non_lin_solver->GetRefSourceVec() + ifld * this->n_pts,
                  1, out_arr[ifld], 1);
    }
  }

  void matrix_multiply_MF(const Array<OneD, const NekDouble> &in_arr,
                          Array<OneD, NekDouble> &out,
                          [[maybe_unused]] const bool &flag) {
    const Array<OneD, const NekDouble> ref_sln =
        this->non_lin_solver->GetRefSolution();
    const Array<OneD, const NekDouble> ref_res =
        this->non_lin_solver->GetRefResidual();

    unsigned int n_tot = in_arr.size();
    NekDouble magninarray = Vmath::Dot(n_tot, in_arr, in_arr);
    this->comm->GetSpaceComm()->AllReduce(magninarray, LU::ReduceSum);
    NekDouble eps = this->jacobi_free_eps *
                    sqrt((sqrt(this->array_norm) + 1.0) / magninarray);

    Array<OneD, NekDouble> soln_plus{n_tot};
    Array<OneD, NekDouble> res_plus{n_tot};

    Vmath::Svtvp(n_tot, eps, in_arr, 1, ref_sln, 1, soln_plus, 1);
    non_lin_sys_evaluator_1D(soln_plus, res_plus, flag);
    Vmath::Vsub(n_tot, res_plus, 1, ref_res, 1, out, 1);
    Vmath::Smul(n_tot, 1.0 / eps, out, 1, out, 1);
  }
};

} // namespace NESO::Solvers

#endif
