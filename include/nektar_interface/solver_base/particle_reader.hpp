#ifndef __PARTICLE_READER_H_
#define __PARTICLE_READER_H_

#include <LibUtilities/BasicUtils/SessionReader.h>

using namespace Nektar;
namespace LU = Nektar::LibUtilities;

namespace NESO::Particles {

typedef std::pair<LU::ParameterMap, LU::FunctionMap> SpeciesMap;
typedef std::map<std::string, SpeciesMap> SpeciesMapList;

enum class ParticleBoundaryConditionType {
  ePeriodic,
  eReflective,
  eNotDefined
};

typedef std::map<std::string, ParticleBoundaryConditionType>
    SpeciesBoundaryList;
typedef std::map<int, SpeciesBoundaryList> ParticleBoundaryList;

typedef std::map<std::string, std::string> ReactionParamMap;
typedef std::tuple<std::string, std::vector<std::string>, ReactionParamMap>
    ReactionMap;
typedef std::map<std::string, ReactionMap> ReactionMapList;

class ParticleReader;
typedef std::shared_ptr<ParticleReader> ParticleReaderSharedPtr;

class ParticleReader {
public:
  ParticleReader(const LU::SessionReaderSharedPtr session)
      : m_session(session) {};

  /// @brief Reads the particle tag from xml document
  void ReadParticles();

  /// @brief Reads info related to particles
  void ReadInfo();
  /// Returns the value of the particle info.
  const std::string &GetInfo(const std::string &pName) const;

  /// @brief  Reads parameters related to particles
  /// @param particles
  void ReadParameters(TiXmlElement *particles);
  /// Checks if a parameter is specified in the XML document.
  bool DefinesParameter(const std::string &name) const;
  /// Returns the value of the specified parameter.
  const NekDouble &GetParameter(const std::string &pName) const;

  /// @brief  Reads functions related to a species (e.g. Initial Conditions)
  /// @param particles
  /// @param functions
  void ReadSpeciesFunctions(TiXmlElement *particles,
                            LU::FunctionMap &functions);

  /// @brief Reads the list of species defined under particles
  /// @param particles
  void ReadSpecies(TiXmlElement *particles);

  const SpeciesMapList &GetSpecies() const { return m_species; }

  /// @brief Reads the particle boundary conditions
  /// @param particles
  void ReadBoundary(TiXmlElement *particles);

  /// @brief Reads the particle boundary conditions
  /// @param particles
  void ReadReactions(TiXmlElement *particles);
  const ReactionMapList &GetReactions() const { return m_reactions; }

  void LoadSpeciesParameter(const std::string &pSpecies,
                            const std::string &pName, int &pVar) const;
  void LoadSpeciesParameter(const std::string &pSpecies,
                            const std::string &pName, NekDouble &pVar) const;

  /// Load an integer parameter
  void LoadParameter(const std::string &name, int &var) const;
  /// Load an size_t parameter
  void LoadParameter(const std::string &name, size_t &var) const;
  /// Check for and load an integer parameter.
  void LoadParameter(const std::string &name, int &var, const int &def) const;
  /// Check for and load an size_t parameter.
  void LoadParameter(const std::string &name, size_t &var,
                     const size_t &def) const;
  /// Load a double precision parameter
  void LoadParameter(const std::string &name, NekDouble &var) const;
  /// Check for and load a double-precision parameter.
  void LoadParameter(const std::string &name, NekDouble &var,
                     const NekDouble &def) const;

private:
  LU::SessionReaderSharedPtr m_session;
  /// Map of particle info (e.g. Particle System name)
  std::map<std::string, std::string> m_particleInfo;
  // Map of specied
  SpeciesMapList m_species;
  LU::ParameterMap m_parameters;
  LU::InterpreterSharedPtr m_interpreter;
  /// Functions.
  LU::FunctionMap m_functions;

  ParticleBoundaryList m_boundaryConditions;

  ReactionMapList m_reactions;

  void ParseEquals(const std::string &line, std::string &lhs, std::string &rhs);
};

} // namespace NESO::Particles
#endif