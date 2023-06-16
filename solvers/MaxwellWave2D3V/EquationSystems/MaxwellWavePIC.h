#ifndef NEKTAR_SOLVERS_EQUATIONSYSTEMS_MAXWELL_WAVE_PIC_H
#define NEKTAR_SOLVERS_EQUATIONSYSTEMS_MAXWELL_WAVE_PIC_H

#include <map>
#include <string>

#include <SolverUtils/UnsteadySystem.h>

#include "UnitConverter.hpp"

using namespace Nektar::SolverUtils;

namespace Nektar
{
class MaxwellWavePIC : public UnsteadySystem
{
public:
    std::map<std::string, int> field_to_index;

    friend class MemoryManager<MaxwellWavePIC>;

    /// Creates an instance of this class
    static EquationSystemSharedPtr create(
        const LibUtilities::SessionReaderSharedPtr &pSession,
        const SpatialDomains::MeshGraphSharedPtr &pGraph)
    {
        EquationSystemSharedPtr p =
            MemoryManager<MaxwellWavePIC>::AllocateSharedPtr(pSession, pGraph);
        p->InitObject();
        return p;
    }
    /// Name of class
    static std::string className1;

    virtual ~MaxwellWavePIC();
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

protected:
    StdRegions::ConstFactorMap m_factors;
    MaxwellWavePIC(const LibUtilities::SessionReaderSharedPtr &pSession,
            const SpatialDomains::MeshGraphSharedPtr &pGraph);

    virtual void v_InitObject(bool DeclareFields = true);
    virtual void v_DoSolve();
    virtual void v_GenerateSummary(SolverUtils::SummaryList &s);
    void LorenzGuageSolve(const int field_t_index,
                          const int field_t_minus1_index,
                          const int source_index);
    void ElectricFieldSolve();
    void MagneticFieldSolve();

    void ElectricFieldSolvePhi(const int E,
                               const int phi,
                               const int phi_minus,
                               MultiRegions::Direction direction,
                               const int nPts);

    void ElectricFieldSolveA(const int E, const int A, const int A_minus,
                             const int nPts);
    void MagneticFieldSolveCurl(const int x, const int y, const int z, const int nPts);
private:
    virtual Array<OneD, bool> v_GetSystemSingularChecks();

    double m_B0x;
    double m_B0y;
    double m_B0z;

    std::shared_ptr<UnitConverter> m_unitConverter;
};
} // namespace Nektar

#endif
