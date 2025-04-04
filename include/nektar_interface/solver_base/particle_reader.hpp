///////////////////////////////////////////////////////////////////////////////
//
// File: particle_reader.hpp
// Based on nektar/library/LibUtilities/BasicUtils/SessionReader.hpp
// at https://gitlab.nektar.info/nektar by "Nektar++ developers"
//
///////////////////////////////////////////////////////////////////////////////

#ifndef __PARTICLE_READER_H_
#define __PARTICLE_READER_H_

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <neso_particles/typedefs.hpp>

namespace LU = Nektar::LibUtilities;
using Nektar::NekDouble;

namespace NESO::Particles {

// {name, parameters, initial, sources}
typedef std::tuple<std::string, LU::ParameterMap,
                   std::pair<int, LU::FunctionVariableMap>,
                   std::vector<std::pair<int, LU::FunctionVariableMap>>>
    SpeciesMap;
typedef std::map<int, SpeciesMap> SpeciesMapList;

enum class ParticleBoundaryConditionType {
  ePeriodic,
  eReflective,
  eNotDefined
};

typedef std::map<int, ParticleBoundaryConditionType> SpeciesBoundaryList;
typedef std::map<int, SpeciesBoundaryList> ParticleBoundaryList;

typedef std::tuple<std::string, std::vector<int>, LU::ParameterMap> ReactionMap;
typedef std::map<int, ReactionMap> ReactionMapList;

class ParticleReader;
typedef std::shared_ptr<ParticleReader> ParticleReaderSharedPtr;

class ParticleReader {
public:
  ParticleReader(const LU::SessionReaderSharedPtr session)
      : session(session), interpreter(session->GetInterpreter()) {};

  /// @brief Reads the particle tag from xml document
  void read_particles();

  /// @brief Reads info related to particles
  void read_info();
  /// Checks if info is specified in the XML document.
  bool defines_info(const std::string &name) const;
  /// Returns the value of the particle info.
  const std::string &get_info(const std::string &name) const;

  /// @brief  Reads parameters related to particles
  /// @param particles
  void read_parameters(TiXmlElement *particles);
  /// Checks if a parameter is specified in the XML document.
  bool defines_parameter(const std::string &name) const;
  /// Returns the value of the specified parameter.
  const NekDouble &get_parameter(const std::string &name) const;

  /// @brief  Reads initial conditions for a species
  /// @param particles
  /// @param initial
  void read_species_initial(TiXmlElement *particles,
                            std::pair<int, LU::FunctionVariableMap> &initial);
  /// @brief  Reads the sources defined for a species
  /// @param particles
  /// @param sources
  void read_species_sources(
      TiXmlElement *particles,
      std::vector<std::pair<int, LU::FunctionVariableMap>> &sources);

  /// @brief Reads the list of species defined under particles
  /// @param particles
  void read_species(TiXmlElement *particles);

  inline const SpeciesMapList &get_species() const { return this->species; }

  /// @brief Reads the particle boundary conditions
  /// @param particles
  void read_boundary(TiXmlElement *particles);

  inline const ParticleBoundaryList &get_boundaries() const {
    return this->boundary_conditions;
  }

  /// @brief Reads the particle boundary conditions
  /// @param particles
  void read_reactions(TiXmlElement *particles);
  inline const ReactionMapList &get_reactions() const {
    return this->reactions;
  }

  /// @brief Loads a species parameter (int)
  /// @param species
  /// @param name
  /// @param var
  void load_species_parameter(const int species, const std::string &name,
                              int &var) const;
  /// @brief Loads a species parameter (double)
  /// @param species
  /// @param name
  /// @param var
  void load_species_parameter(const int species, const std::string &name,
                              NekDouble &var) const;

  int get_species_initial_N(const int species) const;

  /// Returns an EquationSharedPtr to a given function variable.
  LU::EquationSharedPtr get_species_initial(const int species,
                                            const std::string &variable,
                                            const int pDomain = 0) const;
  /// Returns an EquationSharedPtr to a given function variable index.
  LU::EquationSharedPtr get_species_initial(const int species,
                                            const unsigned int &var,
                                            const int pDomain = 0) const;

  const std::vector<std::pair<int, LU::FunctionVariableMap>> &
  get_species_sources(const int species) const;

  /// @brief Loads a reaction parameter (int)
  /// @param reaction
  /// @param name
  /// @param var
  void load_reaction_parameter(const int reaction, const std::string &name,
                               int &var) const;
  /// @brief Loads a reaction parameter (double)
  /// @param reaction
  /// @param name
  /// @param var
  void load_reaction_parameter(const int reaction, const std::string &name,
                               NekDouble &var) const;

  /// Load an integer parameter
  void load_parameter(const std::string &name, int &var) const;
  /// Load an size_t parameter
  void load_parameter(const std::string &name, size_t &var) const;
  /// Check for and load an integer parameter.
  void load_parameter(const std::string &name, int &var, const int &def) const;
  /// Check for and load an size_t parameter.
  void load_parameter(const std::string &name, size_t &var,
                      const size_t &def) const;
  /// Load a double precision parameter
  void load_parameter(const std::string &name, NekDouble &var) const;
  /// Check for and load a double-precision parameter.
  void load_parameter(const std::string &name, NekDouble &var,
                      const NekDouble &def) const;

private:
  // Nektar++ SessionReader
  LU::SessionReaderSharedPtr session;
  // Expression interptreter
  LU::InterpreterSharedPtr interpreter;
  /// Map of particle info (e.g. Particle System name)
  std::map<std::string, std::string> particle_info;
  // Map of species
  SpeciesMapList species;
  // Particle parameters
  LU::ParameterMap parameters;
  /// Functions.
  LU::FunctionMap functions;
  // Boundary conditions
  ParticleBoundaryList boundary_conditions;
  // Reactions
  ReactionMapList reactions;

  void parse_equals(const std::string &line, std::string &lhs,
                    std::string &rhs);
};

} // namespace NESO::Particles
#endif