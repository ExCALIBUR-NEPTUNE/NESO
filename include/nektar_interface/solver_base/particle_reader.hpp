///////////////////////////////////////////////////////////////////////////////
//
// File: particle_reader.hpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description: Particle Reader for NESO
//
///////////////////////////////////////////////////////////////////////////////

#ifndef __PARTICLE_READER_H_
#define __PARTICLE_READER_H_

#include <LibUtilities/BasicUtils/SessionReader.h>

using namespace Nektar;
namespace LU = Nektar::LibUtilities;

namespace NESO::Particles {

typedef std::tuple<std::string, LU::ParameterMap, LU::FunctionMap> SpeciesMap;
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
      : session(session), interpreter(session->GetInterpreter()){};

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

  /// @brief  Reads functions related to a species (e.g. Initial Conditions)
  /// @param particles
  /// @param functions
  void read_species_functions(TiXmlElement *particles,
                              LU::FunctionMap &functions);

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