#ifndef H3LAPD_HW2DIN3D_SYSTEM_H
#define H3LAPD_HW2DIN3D_SYSTEM_H

#include "nektar_interface/utilities.hpp"

#include "HWSystem.hpp"
#include <LibUtilities/Memory/NekMemoryManager.hpp>

namespace LU = Nektar::LibUtilities;
namespace SD = Nektar::SpatialDomains;

namespace NESO::Solvers::H3LAPD {

/**
 * @brief 2D Hasegawa-Wakatani equation system designed to work in a 3D domain.
 * @details Evolves ne, w, phi only, no momenta, no ions.
 */
class HW2Din3DSystem : public HWSystem {
public:
  friend class MemoryManager<HW2Din3DSystem>;

  /// Name of class
  static std::string class_name;
  /// For enum
  static std::string eq_name;

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
