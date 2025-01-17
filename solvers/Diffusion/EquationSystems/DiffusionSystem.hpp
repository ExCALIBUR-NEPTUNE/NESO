#ifndef DIFFUSION_DIFFUSIONSYSTEM_H
#define DIFFUSION_DIFFUSIONSYSTEM_H

#include <SolverUtils/Diffusion/Diffusion.h>
#include <SolverUtils/UnsteadySystem.h>

#include "nektar_interface/solver_base/empty_partsys.hpp"
#include "nektar_interface/solver_base/time_evolved_eqnsys_base.hpp"

namespace LU = Nektar::LibUtilities;
namespace SD = Nektar::SpatialDomains;
namespace SR = Nektar::StdRegions;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::Diffusion {
class DiffusionSystem
    : public TimeEvoEqnSysBase<SU::UnsteadySystem, Particles::EmptyPartSys> {
public:
  friend class MemoryManager<DiffusionSystem>;

  /// Creates an instance of this class
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &session,
         const SD::MeshGraphSharedPtr &graph) {
    SU::EquationSystemSharedPtr p =
        MemoryManager<DiffusionSystem>::AllocateSharedPtr(session, graph);
    p->InitObject();
    return p;
  }
  /// Name of class
  static std::string className;

  /// Destructor
  virtual ~DiffusionSystem();

protected:
  bool use_spec_van_visc;
  /// Cut-off ratio from which to start decaying modes
  NekDouble sVV_cutoff_ratio;
  /// Diffusion coefficient of SVV modes
  NekDouble sVV_diff_coeff;

  DiffusionSystem(const LU::SessionReaderSharedPtr &session,
                  const SD::MeshGraphSharedPtr &graph);

  virtual void v_InitObject(bool DeclareField = true) override;
  virtual void v_GenerateSummary(SU::SummaryList &s) override;

  void
  do_ode_projection(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                    Array<OneD, Array<OneD, NekDouble>> &out_arr,
                    const NekDouble time);
  void
  do_implicit_solve(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                    Array<OneD, Array<OneD, NekDouble>> &out_arr,
                    NekDouble time, NekDouble lambda);

private:
  /// User-supplied parameters used to set Helmsolve factors/variable coeffs
  NekDouble epsilon;
  NekDouble k_perp;
  NekDouble k_par;
  NekDouble n;
  NekDouble theta;

  /// Factors and variable coefficients for Helmsolve
  SR::ConstFactorMap helmsolve_factors;
  SR::VarCoeffMap helmsolve_varcoeffs;
};
} // namespace NESO::Solvers::Diffusion

#endif
