#ifndef H3LAPD_HW2DIN3D_SYSTEM_H
#define H3LAPD_HW2DIN3D_SYSTEM_H

#include "nektar_interface/utilities.hpp"

#include <LibUtilities/Memory/NekMemoryManager.hpp>
// #include <SolverUtils/AdvectionSystem.h>
// #include <SolverUtils/EquationSystem.h>
// #include <SolverUtils/Forcing/Forcing.h>
// #include <SolverUtils/RiemannSolvers/RiemannSolver.h>
// #include <solvers/solver_callback_handler.hpp>

// #include "../Diagnostics/GrowthRatesRecorder.hpp"
// #include "../Diagnostics/MassRecorder.hpp"
#include "HWSystem.hpp"

namespace LU = Nektar::LibUtilities;
// namespace MR = Nektar::MultiRegions;
namespace SD = Nektar::SpatialDomains;
// namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::H3LAPD {

/**
 * @brief 2D Hasegawa-Wakatani equation system designed to work in a 3D domain.
 * @details Intended as an intermediate step towards the full LAPD equation
 * system. Evolves ne, w, phi only, no momenta, no ions.
 */
class HW2Din3DSystem : public HWSystem {
public:
  friend class MemoryManager<HW2Din3DSystem>;

  /// Name of class
  static std::string class_name;

  /**
   * @brief Creates an instance of this class.
   */
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &session,
         const SD::MeshGraphSharedPtr &graph) {
    SU::EquationSystemSharedPtr p =
        MemoryManager<HW2Din3DSystem>::AllocateSharedPtr(session, graph);
    p->InitObject();
    return p;
  }

protected:
  HW2Din3DSystem(const LU::SessionReaderSharedPtr &session,
                 const SD::MeshGraphSharedPtr &graph);

  void
  explicit_time_int(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
                    Array<OneD, Array<OneD, NekDouble>> &outarray,
                    const NekDouble time) override;

  void load_params() final;

  virtual void v_InitObject(bool DeclareField) override;
};

} // namespace NESO::Solvers::H3LAPD
#endif // H3LAPD_HW2DIN3D_SYSTEM_H