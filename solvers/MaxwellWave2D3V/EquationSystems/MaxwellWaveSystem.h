#ifndef NEKTAR_SOLVERS_EQUATIONSYSTEMS_MAXWELL_WAVE_SYSTEM_H
#define NEKTAR_SOLVERS_EQUATIONSYSTEMS_MAXWELL_WAVE_SYSTEM_H

#include <map>
#include <string>

#include <SolverUtils/Diffusion/Diffusion.h>
#include <SolverUtils/EquationSystem.h>

#include "UnitConverter.hpp"

using namespace Nektar::SolverUtils;

namespace Nektar
{
class MaxwellWaveSystem : public EquationSystem
{
public:
  std::map<std::string, int> field_to_index;

  friend class MemoryManager<MaxwellWaveSystem>;

  /// Creates an instance of this class
  static EquationSystemSharedPtr create(
      const LibUtilities::SessionReaderSharedPtr &pSession,
      const SpatialDomains::MeshGraphSharedPtr &pGraph)
  {
      EquationSystemSharedPtr p =
          MemoryManager<MaxwellWaveSystem>::AllocateSharedPtr(pSession, pGraph);
      p->InitObject();
      return p;
  }
  /// Name of class
  static std::string className;

  virtual ~MaxwellWaveSystem();
  /**
   *  Helper function to map from field name to field indices.
   *
   *  @param name Field name
   *  @returns index (probably 0 or 1).
   */
  int GetFieldIndex(const std::string name);

  double m_DtMultiplier;
  double m_theta;

  void setDtMultiplier(const double dtMultiplier);

  void setTheta(const double theta);

  double timeStep();

  void SetVolume(const double volume);
  void ChargeConservationSwitch(const bool onoff);

protected:
  StdRegions::ConstFactorMap m_factors;
  MaxwellWaveSystem(const LibUtilities::SessionReaderSharedPtr &pSession,
          const SpatialDomains::MeshGraphSharedPtr &pGraph);

  void v_InitObject(bool DeclareFields = true) override;
  void v_DoSolve() override;
  void v_GenerateSummary(SolverUtils::SummaryList &s) override;

  void ChargeConservation(const int phi_index,
                          const int phi_minus_index,
                          const int jx_index,
                          const int jy_index);

  void LorenzGaugeSolve(const int field_t_index,
                        const int field_t_minus1_index,
                        const int source_index);
  void ElectricFieldSolve();
  void MagneticFieldSolve();

  void SubtractMean(int field_index);

  void ElectricFieldSolvePhi(const int E,
                             const int phi,
                             const int phi_minus,
                             MultiRegions::Direction direction,
                             const int nPts);

  void ElectricFieldSolveA(const int E, const int A, const int A_minus,
                           const int nPts);
  void MagneticFieldSolveCurl(const int x, const int y, const int z, const int nPts);

  DiffusionSharedPtr m_diffusion;

  void GetDiffusionFluxVector(
    const Array<OneD, Array<OneD, NekDouble>> &in_arr,
    const Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &q_field,
    Array<OneD, Array<OneD, Array<OneD, NekDouble>>> &viscous_tensor);

//  void DoOdeProjection(const Array<OneD, const Array<OneD, NekDouble>> &inarray,
//                     Array<OneD, Array<OneD, NekDouble>> &outarray,
//                     const double time);

private:
  virtual Array<OneD, bool> v_GetSystemSingularChecks();

  std::shared_ptr<UnitConverter> m_unitConverter;

  double m_B0x;
  double m_B0y;
  double m_B0z;
  double m_volume;
  bool m_perform_charge_conservation;

//  std::map<int, Array<OneD, NekDouble>> m_mapIntToArray;
};
} // namespace Nektar

#endif
