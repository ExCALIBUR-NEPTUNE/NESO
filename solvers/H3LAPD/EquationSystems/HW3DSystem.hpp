#ifndef H3LAPD_HW3D_SYSTEM_H
#define H3LAPD_HW3D_SYSTEM_H

#include "nektar_interface/utilities.hpp"

#include <LibUtilities/Memory/NekMemoryManager.hpp>
#include <SolverUtils/AdvectionSystem.h>
#include <SolverUtils/Diffusion/Diffusion.h>
#include <SolverUtils/EquationSystem.h>
#include <SolverUtils/Forcing/Forcing.h>
#include <SolverUtils/RiemannSolvers/RiemannSolver.h>
#include <solvers/solver_callback_handler.hpp>

#include "../Diagnostics/GrowthRatesRecorder.hpp"
#include "../Diagnostics/MassRecorder.hpp"
#include "HWSystem.hpp"

namespace LU = Nektar::LibUtilities;
namespace MR = Nektar::MultiRegions;
namespace SD = Nektar::SpatialDomains;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::H3LAPD {

/**
 * @brief 3D Hasegawa-Wakatani equation system.
 * @details Intended as an intermediate step towards the full LAPD equation
 * system. Evolves ne, w, phi only, no momenta, no ions.
 */
class HW3DSystem : public HWSystem {
public:
  friend class MemoryManager<HW3DSystem>;

  /// Name of class
  static std::string class_name;

  /**
   * @brief Creates an instance of this class.
   */
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &session,
         const SD::MeshGraphSharedPtr &graph) {
    SU::EquationSystemSharedPtr p =
        MemoryManager<HW3DSystem>::AllocateSharedPtr(session, graph);
    p->InitObject();
    return p;
  }

protected:
  HW3DSystem(const LU::SessionReaderSharedPtr &session,
             const SD::MeshGraphSharedPtr &graph);

  void
  calc_par_dyn_term(const Array<OneD, const Array<OneD, NekDouble>> &in_arr);

  void
  explicit_time_int(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                    Array<OneD, Array<OneD, NekDouble>> &outarray,
                    const NekDouble time) override;

  void get_flux_vector_diff(
      const Array<OneD, Array<OneD, NekDouble>> &in_arr,
      const Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &q_field,
      Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &viscous_tensor);

  void load_params() final;

  virtual void v_InitObject(bool DeclareField) override;

private:
  // Diffusion type
  std::string diff_type;

  // Diffusion object
  SU::DiffusionSharedPtr diffusion;

  // Array for storage of parallel dynamics term
  Array<OneD, NekDouble> par_dyn_term;

  // Arrays to store temporary fields and values for the diffusion operation
  Array<OneD, MultiRegions::ExpListSharedPtr> diff_fields;
  Array<OneD, Array<OneD, NekDouble>> diff_in_arr, diff_out_arr;
};

} // namespace NESO::Solvers::H3LAPD
#endif // H3LAPD_HW3D_SYSTEM_H