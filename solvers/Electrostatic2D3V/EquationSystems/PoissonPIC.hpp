#ifndef __NESOSOLVERS_ELECTROSTATIC2D3V_POISSONPIC_HPP__
#define __NESOSOLVERS_ELECTROSTATIC2D3V_POISSONPIC_HPP__

#include <map>
#include <string>

#include <SolverUtils/EquationSystem.h>

namespace LU = Nektar::LibUtilities;
namespace SD = Nektar::SpatialDomains;
namespace SR = Nektar::StdRegions;
namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers::Electrostatic2D3V {
class PoissonPIC : public SU::EquationSystem {
public:
  std::map<std::string, int> field_to_index;

  friend class Nektar::MemoryManager<PoissonPIC>;

  /// Creates an instance of this class
  static SU::EquationSystemSharedPtr
  create(const LU::SessionReaderSharedPtr &pSession,
         const SD::MeshGraphSharedPtr &pGraph) {
    SU::EquationSystemSharedPtr p =
        Nektar::MemoryManager<PoissonPIC>::AllocateSharedPtr(pSession, pGraph);
    p->InitObject();
    return p;
  }
  /// Name of class
  static std::string className1;
  static std::string className2;

  virtual ~PoissonPIC();
  /**
   *  Helper function to map from field name to field indices.
   *
   *  @param name Field name (probably either "u" or "rho").
   *  @returns index (probably 0 or 1).
   */
  int GetFieldIndex(const std::string name);

protected:
  SR::ConstFactorMap m_factors;
  PoissonPIC(const LU::SessionReaderSharedPtr &pSession,
             const SD::MeshGraphSharedPtr &pGraph);

  virtual void v_InitObject(bool DeclareFields = true);
  virtual void v_DoSolve();
  virtual void v_GenerateSummary(SU::SummaryList &s);

private:
  virtual Nektar::Array<Nektar::OneD, bool> v_GetSystemSingularChecks();
};
} // namespace NESO::Solvers::Electrostatic2D3V

#endif // __NESOSOLVERS_ELECTROSTATIC2D3V_POISSONPIC_HPP__
