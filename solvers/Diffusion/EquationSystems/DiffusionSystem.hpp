#ifndef __NESOSOLVERS_DIFFUSION_DIFFUSIONSYSTEM_HPP__
#define __NESOSOLVERS_DIFFUSION_DIFFUSIONSYSTEM_HPP__

#include <SolverUtils/Diffusion/Diffusion.h>
#include <SolverUtils/UnsteadySystem.h>

#include <nektar_interface/solver_base/empty_partsys.hpp>
#include <nektar_interface/solver_base/time_evolved_eqnsys_base.hpp>

namespace LU = Nektar::LibUtilities;
namespace NC = Nektar::Collections;
namespace SD = Nektar::SpatialDomains;
namespace SR = Nektar::StdRegions;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::Diffusion {

class DiffusionSystem
    : public TimeEvoEqnSysBase<SU::UnsteadySystem, Particles::EmptyPartSys> {
public:
  friend class Nektar::MemoryManager<DiffusionSystem>;

  /// Creates an instance of this class
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &session,
         const SD::MeshGraphSharedPtr &graph) {
    SU::EquationSystemSharedPtr p =
        Nektar::MemoryManager<DiffusionSystem>::AllocateSharedPtr(session,
                                                                  graph);
    p->InitObject();
    return p;
  }
  /// Name of class
  static std::string class_name;

  /// Destructor
  virtual ~DiffusionSystem(){};

protected:
  DiffusionSystem(const LU::SessionReaderSharedPtr &session,
                  const SD::MeshGraphSharedPtr &graph);

  /// User-supplied parameters used to set Helmsolve factors/variable coeffs
  NekDouble epsilon;
  NekDouble k_perp;
  NekDouble k_par;
  NekDouble n;
  NekDouble theta;

  /// Cut-off ratio from which to start decaying modes
  NekDouble sVV_cutoff_ratio;
  /// Diffusion coefficient of SVV modes
  NekDouble sVV_diff_coeff;
  bool use_spec_van_visc;

  /// Factors and variable coefficients for Helmsolve
  SR::ConstFactorMap helmsolve_factors;
  SR::VarCoeffMap helmsolve_varcoeffs;

  void
  do_implicit_solve(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                    Array<OneD, Array<OneD, NekDouble>> &out_arr,
                    NekDouble time, NekDouble lambda);

  void
  do_ode_projection(const Array<OneD, const Array<OneD, NekDouble>> &in_arr,
                    Array<OneD, Array<OneD, NekDouble>> &out_arr,
                    const NekDouble time);
  NC::ImplementationType get_collection_type();
  virtual void load_params() override;
  void setup_helmsolve_coeffs();

  virtual void v_InitObject(bool DeclareField = true) override;
  virtual void v_GenerateSummary(SU::SummaryList &s) override;
};
} // namespace NESO::Solvers::Diffusion

#endif // __NESOSOLVERS_DIFFUSION_DIFFUSIONSYSTEM_HPP__
