#ifndef __NESOSOLVERS_ELECTROSTATIC2D3V_POISSONPIC_HPP__
#define __NESOSOLVERS_ELECTROSTATIC2D3V_POISSONPIC_HPP__

#include <map>
#include <string>

#include <SolverUtils/EquationSystem.h>
using namespace Nektar::SolverUtils;

namespace Nektar {
class PoissonPIC : public EquationSystem {
public:
  std::map<std::string, int> field_to_index;

  friend class MemoryManager<PoissonPIC>;

  /// Creates an instance of this class
  static EquationSystemSharedPtr
  create(const LibUtilities::SessionReaderSharedPtr &pSession,
         const SpatialDomains::MeshGraphSharedPtr &pGraph) {
    EquationSystemSharedPtr p =
        MemoryManager<PoissonPIC>::AllocateSharedPtr(pSession, pGraph);
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
  StdRegions::ConstFactorMap m_factors;
  PoissonPIC(const LibUtilities::SessionReaderSharedPtr &pSession,
             const SpatialDomains::MeshGraphSharedPtr &pGraph);

  virtual void v_InitObject(bool DeclareFields = true);
  virtual void v_DoSolve();
  virtual void v_GenerateSummary(SolverUtils::SummaryList &s);

private:
  virtual Array<OneD, bool> v_GetSystemSingularChecks();
};
} // namespace Nektar

#endif // __NESOSOLVERS_ELECTROSTATIC2D3V_POISSONPIC_HPP__
