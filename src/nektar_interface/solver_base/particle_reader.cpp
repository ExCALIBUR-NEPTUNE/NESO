#include "../../../include/nektar_interface/solver_base/particle_reader.hpp"
#include <fstream>
#include <iostream>
#include <string>

#include <boost/algorithm/string.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>

#include <tinyxml.h>

#include <LibUtilities/BasicUtils/CheckedCast.hpp>
#include <LibUtilities/BasicUtils/Equation.h>
#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <LibUtilities/BasicUtils/ParseUtils.h>
#include <LibUtilities/Interpreter/Interpreter.h>
#include <LibUtilities/Memory/NekMemoryManager.hpp>

#include <boost/format.hpp>
#include <boost/program_options.hpp>

using namespace Nektar;

namespace NESO::Particles {
/**
 *
 */
void ParticleReader::ParseEquals(const std::string &line, std::string &lhs,
                                 std::string &rhs) {
  /// Pull out lhs and rhs and eliminate any spaces.
  size_t beg = line.find_first_not_of(" ");
  size_t end = line.find_first_of("=");
  // Check for no parameter name
  if (beg == end) {
    throw 1;
  }
  // Check for no parameter value
  if (end != line.find_last_of("=")) {
    throw 1;
  }
  // Check for no equals sign
  if (end == std::string::npos) {
    throw 1;
  }

  lhs = line.substr(line.find_first_not_of(" "), end - beg);
  lhs = lhs.substr(0, lhs.find_last_not_of(" ") + 1);
  rhs = line.substr(line.find_last_of("=") + 1);
  rhs = rhs.substr(rhs.find_first_not_of(" "));
  rhs = rhs.substr(0, rhs.find_last_not_of(" ") + 1);
}

void ParticleReader::ReadInfo() {
  ASSERTL0(&m_session->GetDocument(), "No XML document loaded.");

  TiXmlHandle docHandle(&m_session->GetDocument());
  TiXmlElement *particles;

  // Look for all data in PARTICLES block.
  particles = docHandle.FirstChildElement("NEKTAR")
                  .FirstChildElement("PARTICLES")
                  .Element();
  if (!particles) {
    return;
  }
  m_particleInfo.clear();

  TiXmlElement *particleInfoElement = particles->FirstChildElement("INFO");

  if (particleInfoElement) {
    TiXmlElement *particleInfo = particleInfoElement->FirstChildElement("I");

    while (particleInfo) {
      std::stringstream tagcontent;
      tagcontent << *particleInfo;
      // read the property name
      ASSERTL0(particleInfo->Attribute("PROPERTY"),
               "Missing PROPERTY attribute in particle info "
               "XML element: \n\t'" +
                   tagcontent.str() + "'");
      std::string particleProperty = particleInfo->Attribute("PROPERTY");
      ASSERTL0(!particleProperty.empty(),
               "PROPERTY attribute must be non-empty in XML "
               "element: \n\t'" +
                   tagcontent.str() + "'");

      // make sure that solver property is capitalised
      std::string particlePropertyUpper =
          boost::to_upper_copy(particleProperty);

      // read the value
      ASSERTL0(particleInfo->Attribute("VALUE"),
               "Missing VALUE attribute in particle info "
               "XML element: \n\t'" +
                   tagcontent.str() + "'");
      std::string particleValue = particleInfo->Attribute("VALUE");
      ASSERTL0(!particleValue.empty(),
               "VALUE attribute must be non-empty in XML "
               "element: \n\t'" +
                   tagcontent.str() + "'");

      // Set Variable
      m_particleInfo[particlePropertyUpper] = particleValue;
      particleInfo = particleInfo->NextSiblingElement("I");
    }
  }
}

/**
 *
 */
bool ParticleReader::DefinesInfo(const std::string &pName) const {
  std::string vName = boost::to_upper_copy(pName);
  return m_particleInfo.find(vName) != m_particleInfo.end();
}
/**
 * If the parameter is not defined, termination occurs. Therefore, the
 * parameters existence should be tested for using #DefinesParameter
 * before calling this function.
 *
 * @param   pName       The name of a floating-point parameter.
 * @returns The value of the floating-point parameter.
 */
const std::string &ParticleReader::GetInfo(const std::string &pName) const {
  std::string vName = boost::to_upper_copy(pName);
  auto infoIter = m_particleInfo.find(vName);

  ASSERTL0(infoIter != m_particleInfo.end(),
           "Unable to find requested info: " + pName);

  return infoIter->second;
}

void ParticleReader::ReadParameters(TiXmlElement *particles) {
  m_parameters.clear();

  TiXmlElement *parameters = particles->FirstChildElement("PARAMETERS");

  // See if we have parameters defined.  They are optional so we go on
  // if not.
  if (parameters) {
    TiXmlElement *parameter = parameters->FirstChildElement("P");

    // Multiple nodes will only occur if there is a comment in
    // between definitions.
    while (parameter) {
      std::stringstream tagcontent;
      tagcontent << *parameter;
      TiXmlNode *node = parameter->FirstChild();

      while (node && node->Type() != TiXmlNode::TINYXML_TEXT) {
        node = node->NextSibling();
      }

      if (node) {
        // Format is "paramName = value"
        std::string line = node->ToText()->Value(), lhs, rhs;

        try {
          ParseEquals(line, lhs, rhs);
        } catch (...) {
          NEKERROR(ErrorUtil::efatal, "Syntax error in parameter expression '" +
                                          line + "' in XML element: \n\t'" +
                                          tagcontent.str() + "'");
        }

        // We want the list of parameters to have their RHS
        // evaluated, so we use the expression evaluator to do
        // the dirty work.
        if (!lhs.empty() && !rhs.empty()) {
          NekDouble value = 0.0;
          try {
            LibUtilities::Equation expession(m_interpreter, rhs);
            value = expession.Evaluate();
          } catch (const std::runtime_error &) {
            NEKERROR(ErrorUtil::efatal, "Error evaluating parameter expression"
                                        " '" +
                                            rhs + "' in XML element: \n\t'" +
                                            tagcontent.str() + "'");
          }
          m_interpreter->SetParameter(lhs, value);
          boost::to_upper(lhs);
          m_parameters[lhs] = value;
        }
      }
      parameter = parameter->NextSiblingElement();
    }
  }
}

/**
 *
 */
bool ParticleReader::DefinesParameter(const std::string &pName) const {
  std::string vName = boost::to_upper_copy(pName);
  return m_parameters.find(vName) != m_parameters.end();
}

/**
 * If the parameter is not defined, termination occurs. Therefore, the
 * parameters existence should be tested for using #DefinesParameter
 * before calling this function.
 *
 * @param   pName       The name of a floating-point parameter.
 * @returns The value of the floating-point parameter.
 */
const NekDouble &ParticleReader::GetParameter(const std::string &pName) const {
  std::string vName = boost::to_upper_copy(pName);
  auto paramIter = m_parameters.find(vName);

  ASSERTL0(paramIter != m_parameters.end(),
           "Unable to find requested parameter: " + pName);

  return paramIter->second;
}

void ParticleReader::ReadSpeciesFunctions(TiXmlElement *specie,
                                          LU::FunctionMap &functions) {
  functions.clear();

  if (!specie) {
    return;
  }

  // Scan through conditions section looking for functions.
  TiXmlElement *function = specie->FirstChildElement("FUNCTION");

  while (function) {
    std::stringstream tagcontent;
    tagcontent << *function;

    // Every function must have a NAME attribute
    ASSERTL0(function->Attribute("NAME"),
             "Functions must have a NAME attribute defined in XML "
             "element: \n\t'" +
                 tagcontent.str() + "'");
    std::string functionStr = function->Attribute("NAME");
    ASSERTL0(!functionStr.empty(),
             "Functions must have a non-empty name in XML "
             "element: \n\t'" +
                 tagcontent.str() + "'");

    // Store function names in uppercase to remain case-insensitive.
    boost::to_upper(functionStr);

    // Retrieve first entry (variable, or file)
    TiXmlElement *element = function;
    TiXmlElement *variable = element->FirstChildElement();

    // Create new function structure with default type of none.
    LU::FunctionVariableMap functionVarMap;

    // Process all entries in the function block
    while (variable) {
      LU::FunctionVariableDefinition funcDef;
      std::string conditionType = variable->Value();

      // If no var is specified, assume wildcard
      std::string variableStr;
      if (!variable->Attribute("VAR")) {
        variableStr = "*";
      } else {
        variableStr = variable->Attribute("VAR");
      }

      // Parse list of variables
      std::vector<std::string> variableList;
      ParseUtils::GenerateVector(variableStr, variableList);

      // If no domain is specified, put to 0
      std::string domainStr;
      if (!variable->Attribute("DOMAIN")) {
        domainStr = "0";
      } else {
        domainStr = variable->Attribute("DOMAIN");
      }

      // Parse list of domains
      std::vector<std::string> varSplit;
      std::vector<unsigned int> domainList;
      ParseUtils::GenerateSeqVector(domainStr, domainList);

      // if no evars is specified, put "x y z t"
      std::string evarsStr = "x y z t";
      if (variable->Attribute("EVARS")) {
        evarsStr = evarsStr + std::string(" ") + variable->Attribute("EVARS");
      }

      // Expressions are denoted by E
      if (conditionType == "E") {
        funcDef.m_type = LU::eFunctionTypeExpression;

        // Expression must have a VALUE.
        ASSERTL0(variable->Attribute("VALUE"),
                 "Attribute VALUE expected for function '" + functionStr +
                     "'.");
        std::string fcnStr = variable->Attribute("VALUE");
        ASSERTL0(!fcnStr.empty(),
                 (std::string("Expression for var: ") + variableStr +
                  std::string(" must be specified."))
                     .c_str());

        // set expression
        funcDef.m_expression = MemoryManager<LU::Equation>::AllocateSharedPtr(
            m_interpreter, fcnStr, evarsStr);
      }

      // Files are denoted by F
      else if (conditionType == "F") {
        // Check if transient or not
        if (variable->Attribute("TIMEDEPENDENT") &&
            boost::lexical_cast<bool>(variable->Attribute("TIMEDEPENDENT"))) {
          funcDef.m_type = LU::eFunctionTypeTransientFile;
        } else {
          funcDef.m_type = LU::eFunctionTypeFile;
        }

        // File must have a FILE.
        ASSERTL0(variable->Attribute("FILE"),
                 "Attribute FILE expected for function '" + functionStr + "'.");
        std::string filenameStr = variable->Attribute("FILE");
        ASSERTL0(!filenameStr.empty(),
                 "A filename must be specified for the FILE "
                 "attribute of function '" +
                     functionStr + "'.");

        std::vector<std::string> fSplit;
        boost::split(fSplit, filenameStr, boost::is_any_of(":"));
        ASSERTL0(fSplit.size() == 1 || fSplit.size() == 2,
                 "Incorrect filename specification in function " + functionStr +
                     "'. "
                     "Specify variables inside file as: "
                     "filename:var1,var2");

        // set the filename
        fs::path fullpath = fSplit[0];
        // fs::path ftype = fullpath.extension();
        // if (fullpath.parent_path().extension() == ".pit") {
        //   std::string filename = fullpath.stem().string();
        //   fullpath = fullpath.parent_path();
        //   size_t start = filename.find_last_of("_") + 1;
        //   int index = atoi(filename.substr(start, filename.size()).c_str());
        //   fullpath /= filename.substr(0, start) +
        //               std::to_string(index +
        //               m_comm->GetTimeComm()->GetRank()) + ftype.string();
        // }
        funcDef.m_filename = fullpath.string();

        if (fSplit.size() == 2) {
          ASSERTL0(variableList[0] != "*",
                   "Filename variable mapping not valid "
                   "when using * as a variable inside "
                   "function '" +
                       functionStr + "'.");

          boost::split(varSplit, fSplit[1], boost::is_any_of(","));
          ASSERTL0(varSplit.size() == variableList.size(),
                   "Filename variables should contain the "
                   "same number of variables defined in "
                   "VAR in function " +
                       functionStr + "'.");
        }
      }

      // Nothing else supported so throw an error
      else {
        std::stringstream tagcontent;
        tagcontent << *variable;

        NEKERROR(ErrorUtil::efatal,
                 "Identifier " + conditionType + " in function " +
                     std::string(function->Attribute("NAME")) +
                     " is not recognised in XML element: \n\t'" +
                     tagcontent.str() + "'");
      }

      // Add variables to function
      for (unsigned int i = 0; i < variableList.size(); ++i) {
        for (unsigned int j = 0; j < domainList.size(); ++j) {
          // Check it has not already been defined
          std::pair<std::string, int> key(variableList[i], domainList[j]);
          auto fcnsIter = functionVarMap.find(key);
          ASSERTL0(fcnsIter == functionVarMap.end(),
                   "Error setting expression '" + variableList[i] +
                       " in domain " + std::to_string(domainList[j]) +
                       "' in function '" + functionStr +
                       "'. "
                       "Expression has already been defined.");

          if (varSplit.size() > 0) {
            LU::FunctionVariableDefinition funcDef2 = funcDef;
            funcDef2.m_fileVariable = varSplit[i];
            functionVarMap[key] = funcDef2;
          } else {
            functionVarMap[key] = funcDef;
          }
        }
      }
      variable = variable->NextSiblingElement();
    }

    // Add function definition to map
    functions[functionStr] = functionVarMap;
    function = function->NextSiblingElement("FUNCTION");
  }
}

void ParticleReader::ReadSpecies(TiXmlElement *particles) {
  TiXmlElement *species = particles->FirstChildElement("SPECIES");
  if (species) {
    TiXmlElement *specie = species->FirstChildElement("S");

    while (specie) {
      std::stringstream tagcontent;
      tagcontent << *specie;
      std::string id = specie->Attribute("ID");
      ASSERTL0(!id.empty(), "Missing ID attribute in Species XML "
                            "element: \n\t'" +
                                tagcontent.str() + "'");

      std::string name = specie->Attribute("NAME");
      ASSERTL0(!name.empty(),
               "NAME attribute must be non-empty in XML element:\n\t'" +
                   tagcontent.str() + "'");
      SpeciesMap species_map;
      std::get<0>(species_map) = name;

      TiXmlElement *parameter = specie->FirstChildElement("P");

      // Multiple nodes will only occur if there is a comment in
      // between definitions.
      while (parameter) {
        std::stringstream tagcontent;
        tagcontent << *parameter;
        TiXmlNode *node = parameter->FirstChild();

        while (node && node->Type() != TiXmlNode::TINYXML_TEXT) {
          node = node->NextSibling();
        }

        if (node) {
          // Format is "paramName = value"
          std::string line = node->ToText()->Value(), lhs, rhs;

          try {
            ParseEquals(line, lhs, rhs);
          } catch (...) {
            NEKERROR(ErrorUtil::efatal,
                     "Syntax error in parameter expression '" + line +
                         "' in XML element: \n\t'" + tagcontent.str() + "'");
          }

          // We want the list of parameters to have their RHS
          // evaluated, so we use the expression evaluator to do
          // the dirty work.
          if (!lhs.empty() && !rhs.empty()) {
            NekDouble value = 0.0;
            try {
              LibUtilities::Equation expession(m_interpreter, rhs);
              value = expession.Evaluate();
            } catch (const std::runtime_error &) {
              NEKERROR(ErrorUtil::efatal,
                       "Error evaluating parameter expression"
                       " '" +
                           rhs + "' in XML element: \n\t'" + tagcontent.str() +
                           "'");
            }
            m_interpreter->SetParameter(lhs, value);
            boost::to_upper(lhs);
            std::get<1>(species_map)[lhs] = value;
          }
        }
        parameter = parameter->NextSiblingElement();
      }

      ReadSpeciesFunctions(specie, std::get<2>(species_map));
      specie = specie->NextSiblingElement("S");

      m_species[std::stoi(id)] = species_map;
    }
  }
}

void ParticleReader::ReadBoundary(TiXmlElement *particles) {
  // Protect against multiple reads.
  if (m_boundaryConditions.size() != 0) {
    return;
  }

  // Read REGION tags
  TiXmlElement *boundaryConditionsElement =
      particles->FirstChildElement("BOUNDARYINTERACTION");

  if (boundaryConditionsElement) {
    TiXmlElement *regionElement =
        boundaryConditionsElement->FirstChildElement("REGION");

    // Read C(Composite), P (Periodic) tags
    while (regionElement) {
      SpeciesBoundaryList boundaryConditions;

      int boundaryRegionID;
      int err = regionElement->QueryIntAttribute("REF", &boundaryRegionID);
      ASSERTL0(err == TIXML_SUCCESS,
               "Error reading boundary region reference.");

      ASSERTL0(m_boundaryConditions.count(boundaryRegionID) == 0,
               "Boundary region '" + std::to_string(boundaryRegionID) +
                   "' appears multiple times.");

      // Find the boundary region corresponding to this ID.
      std::string boundaryRegionIDStr;
      std::ostringstream boundaryRegionIDStrm(boundaryRegionIDStr);
      boundaryRegionIDStrm << boundaryRegionID;

      // if (m_boundaryRegions.count(boundaryRegionID) == 0) {
      //   regionElement = regionElement->NextSiblingElement("REGION");
      //   continue;
      // }

      // ASSERTL0(m_boundaryRegions.count(boundaryRegionID) == 1,
      //          "Boundary region " +
      //              boost::lexical_cast<std::string>(boundaryRegionID) +
      //              " not found");

      TiXmlElement *conditionElement = regionElement->FirstChildElement();

      while (conditionElement) {
        // Check type.
        std::string conditionType = conditionElement->Value();
        std::string attrData;
        bool isTimeDependent = false;

        // All species are specified, or else all species are zero.
        TiXmlAttribute *attr = conditionElement->FirstAttribute();

        SpeciesMapList::iterator iter;
        std::string attrName;
        attrData = conditionElement->Attribute("SPECIES");
        int speciesID = std::stoi(attrData);
        if (conditionType == "C") {
          if (attrData.empty()) {
            // All species are reflect.
            for (const auto &species : m_species) {
              boundaryConditions[species.first] =
                  ParticleBoundaryConditionType::eReflective;
            }
          } else {
            if (attr) {
              std::string equation, userDefined, filename;

              while (attr) {

                attrName = attr->Name();

                if (attrName == "SPECIES") {
                  // if VAR do nothing
                } else if (attrName == "VALUE") {
                  ASSERTL0(
                      attrName == "VALUE",
                      (std::string("Unknown attribute: ") + attrName).c_str());

                  attrData = attr->Value();
                  ASSERTL0(!attrData.empty(),
                           "VALUE attribute must be specified.");

                } else {
                  ASSERTL0(false, (std::string("Unknown boundary "
                                               "condition attribute: ") +
                                   attrName)
                                      .c_str());
                }
                attr = attr->Next();
              }
              boundaryConditions[speciesID] =
                  ParticleBoundaryConditionType::eReflective;
            } else {
              // This variable's condition is zero.
            }
          }
        }

        else if (conditionType == "P") {
          if (attrData.empty()) {
            ASSERTL0(false, "Periodic boundary conditions should "
                            "be explicitly defined");
          } else {
            if (attr) {
              std::string userDefined;
              std::vector<unsigned int> periodicBndRegionIndex;
              while (attr) {
                attrName = attr->Name();

                if (attrName == "SPECIES") {
                  // if VAR do nothing
                } else if (attrName == "USERDEFINEDTYPE") {
                  // Do stuff for the user defined attribute
                  attrData = attr->Value();
                  ASSERTL0(!attrData.empty(),
                           "USERDEFINEDTYPE attribute must have "
                           "associated value.");

                  userDefined = attrData;
                  isTimeDependent = boost::iequals(attrData, "TimeDependent");
                } else if (attrName == "VALUE") {
                  attrData = attr->Value();
                  ASSERTL0(!attrData.empty(),
                           "VALUE attribute must have associated "
                           "value.");

                  int beg = attrData.find_first_of("[");
                  int end = attrData.find_first_of("]");
                  std::string periodicBndRegionIndexStr =
                      attrData.substr(beg + 1, end - beg - 1);
                  ASSERTL0(beg < end, (std::string("Error reading periodic "
                                                   "boundary region definition "
                                                   "for boundary region: ") +
                                       boundaryRegionIDStrm.str())
                                          .c_str());

                  bool parseGood = ParseUtils::GenerateSeqVector(
                      periodicBndRegionIndexStr.c_str(),
                      periodicBndRegionIndex);

                  ASSERTL0(parseGood && (periodicBndRegionIndex.size() == 1),
                           (std::string("Unable to read periodic boundary "
                                        "condition for boundary "
                                        "region: ") +
                            boundaryRegionIDStrm.str())
                               .c_str());
                }
                attr = attr->Next();
              }
              boundaryConditions[speciesID] =
                  ParticleBoundaryConditionType::ePeriodic;
            } else {
              ASSERTL0(false, "Periodic boundary conditions should "
                              "be explicitly defined");
            }
          }
        }

        conditionElement = conditionElement->NextSiblingElement();
      }

      m_boundaryConditions[boundaryRegionID] = boundaryConditions;
      regionElement = regionElement->NextSiblingElement("REGION");
    }
  }
}

void ParticleReader::ReadReactions(TiXmlElement *particles) {
  TiXmlElement *reactions = particles->FirstChildElement("REACTIONS");
  if (reactions) {
    TiXmlElement *reaction = reactions->FirstChildElement("R");

    while (reaction) {
      std::stringstream tagcontent;
      tagcontent << *reaction;
      std::string id = reaction->Attribute("ID");
      ASSERTL0(!id.empty(), "Missing ID attribute in Reaction XML "
                            "element: \n\t'" +
                                tagcontent.str() + "'");
      std::string type = reaction->Attribute("TYPE");
      ASSERTL0(!type.empty(),
               "TYPE attribute must be non-empty in XML element:\n\t'" +
                   tagcontent.str() + "'");
      ReactionMap reaction_map;
      std::get<0>(reaction_map) = type;
      std::string species = reaction->Attribute("SPECIES");
      boost::split(std::get<1>(reaction_map), species, boost::is_any_of(","));

      for (const auto &s : std::get<1>(reaction_map)) {
        ASSERTL0(
            m_species.find(std::stoi(s)) != m_species.end(),
            "Species '" + s +
                "' not found.  Ensure it is specified under the <SPECIES> tag");
      }

      TiXmlElement *info = reaction->FirstChildElement("P");
      while (info) {
        tagcontent.clear();
        tagcontent << *info;
        // read the property name
        ASSERTL0(info->Attribute("PROPERTY"),
                 "Missing PROPERTY attribute in "
                 "Reaction  '" +
                     id + "' in XML element: \n\t'" + tagcontent.str() + "'");
        std::string property = info->Attribute("PROPERTY");
        ASSERTL0(!property.empty(), "Reactions properties must have a "
                                    "non-empty name for Reaction '" +
                                        id + "' in XML element: \n\t'" +
                                        tagcontent.str() + "'");

        // make sure that solver property is capitalised
        std::string propertyUpper = boost::to_upper_copy(property);

        // read the value
        ASSERTL0(info->Attribute("VALUE"),
                 "Missing VALUE attribute in Reaction '" + id +
                     "' in XML element: \n\t" + tagcontent.str() + "'");
        std::string value = info->Attribute("VALUE");
        ASSERTL0(!value.empty(), "Reactions properties must have a "
                                 "non-empty value for Reaction '" +
                                     id + "' in XML element: \n\t'" +
                                     tagcontent.str() + "'");
        std::get<2>(reaction_map)[property] = value;
        info = info->NextSiblingElement("P");
      }

      reaction = reaction->NextSiblingElement("R");
      m_reactions[id] = reaction_map;
    }
  }
}

void ParticleReader::ReadParticles() {
  // Check we actually have a document loaded.
  ASSERTL0(&m_session->GetDocument(), "No XML document loaded.");

  TiXmlHandle docHandle(&m_session->GetDocument());
  TiXmlElement *particles;

  // Look for all data in PARTICLES block.
  particles = docHandle.FirstChildElement("NEKTAR")
                  .FirstChildElement("PARTICLES")
                  .Element();

  if (!particles) {
    return;
  }
  ReadParameters(particles);
  ReadSpecies(particles);
  ReadBoundary(particles);
  ReadReactions(particles);
}

void ParticleReader::LoadSpeciesParameter(const int pSpecies,
                                          const std::string &pName,
                                          int &pVar) const {
  std::string vName = boost::to_upper_copy(pName);
  auto map = std::get<1>(m_species.at(pSpecies));
  auto paramIter = map.find(vName);
  ASSERTL0(paramIter != map.end(),
           "Required parameter '" + pName + "' not specified in session.");
  NekDouble param = round(paramIter->second);
  pVar = LU::checked_cast<int>(param);
}

void ParticleReader::LoadSpeciesParameter(const int pSpecies,
                                          const std::string &pName,
                                          NekDouble &pVar) const {
  std::string vName = boost::to_upper_copy(pName);
  auto map = std::get<1>(m_species.at(pSpecies));
  auto paramIter = map.find(vName);
  ASSERTL0(paramIter != map.end(),
           "Required parameter '" + pName + "' not specified in session.");
  pVar = paramIter->second;
}

/**
 *
 */
void ParticleReader::LoadParameter(const std::string &pName, int &pVar) const {
  std::string vName = boost::to_upper_copy(pName);
  auto paramIter = m_parameters.find(vName);
  ASSERTL0(paramIter != m_parameters.end(),
           "Required parameter '" + pName + "' not specified in session.");
  NekDouble param = round(paramIter->second);
  pVar = LU::checked_cast<int>(param);
}

/**
 *
 */
void ParticleReader::LoadParameter(const std::string &pName, int &pVar,
                                   const int &pDefault) const {
  std::string vName = boost::to_upper_copy(pName);
  auto paramIter = m_parameters.find(vName);
  if (paramIter != m_parameters.end()) {
    NekDouble param = round(paramIter->second);
    pVar = LU::checked_cast<int>(param);
  } else {
    pVar = pDefault;
  }
}

/**
 *
 */
void ParticleReader::LoadParameter(const std::string &pName,
                                   size_t &pVar) const {
  std::string vName = boost::to_upper_copy(pName);
  auto paramIter = m_parameters.find(vName);
  ASSERTL0(paramIter != m_parameters.end(),
           "Required parameter '" + pName + "' not specified in session.");
  NekDouble param = round(paramIter->second);
  pVar = LU::checked_cast<int>(param);
}

/**
 *
 */
void ParticleReader::LoadParameter(const std::string &pName, size_t &pVar,
                                   const size_t &pDefault) const {
  std::string vName = boost::to_upper_copy(pName);
  auto paramIter = m_parameters.find(vName);
  if (paramIter != m_parameters.end()) {
    NekDouble param = round(paramIter->second);
    pVar = LU::checked_cast<int>(param);
  } else {
    pVar = pDefault;
  }
}

/**
 *
 */
void ParticleReader::LoadParameter(const std::string &pName,
                                   NekDouble &pVar) const {
  std::string vName = boost::to_upper_copy(pName);
  auto paramIter = m_parameters.find(vName);
  ASSERTL0(paramIter != m_parameters.end(),
           "Required parameter '" + pName + "' not specified in session.");
  pVar = paramIter->second;
}

/**
 *
 */
void ParticleReader::LoadParameter(const std::string &pName, NekDouble &pVar,
                                   const NekDouble &pDefault) const {
  std::string vName = boost::to_upper_copy(pName);
  auto paramIter = m_parameters.find(vName);
  if (paramIter != m_parameters.end()) {
    pVar = paramIter->second;
  } else {
    pVar = pDefault;
  }
}
} // namespace NESO::Particles
