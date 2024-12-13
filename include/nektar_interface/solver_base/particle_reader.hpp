#ifndef __PARTICLE_READER_H_
#define __PARTICLE_READER_H_

#include <LibUtilities/BasicUtils/SessionReader.h>

namespace LU = Nektar::LibUtilities;

namespace NESO::Particles {
class ParticleReader;
typedef std::shared_ptr<ParticleReader> ParticleReaderSharedPtr;

class ParticleReader {
public:
  ParticleReader(const LU::SessionReaderSharedPtr session) : m_session(session){};

  /// @brief Reads the particle tag from xml document
  void ReadParticles();
  /// @brief Reads info related to particles
  void ReadInfo();
  /// @brief  Reads parameters related to particles
  /// @param particles
  void ReadParameters(TiXmlElement *particles);

  /// @brief Reads the list of species defined under particles
  /// @param particles
  void ReadSpecies(TiXmlElement *particles);

  /// @brief Reads the particle boundary conditions
  /// @params particles
  void ReadBoundary(TiXmlElement *particles);

private:
  LU::SessionReaderSharedPtr m_session;
  /// Map of particle info (e.g. Particle System name)
  std::map<std::string, std::string> m_particleInfo;
  LU::ParameterMap m_parameters;
  LU::InterpreterSharedPtr m_interpreter;
};

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

void ParticleReader::ReadParameters(TiXmlElement *particles) {
  m_parameters.clear();

  TiXmlElement *parameters = particles->FirstChildElement("PARAMETERS");

  // See if we have parameters defined.  They are optional so we go on
  // if not.
  if (parameters) {
    TiXmlElement *parameter = parameters->FirstChildElement("P");

    LU::ParameterMap caseSensitiveParameters;

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
          caseSensitiveParameters[lhs] = value;
          boost::to_upper(lhs);
          m_parameters[lhs] = value;
        }
      }
      parameter = parameter->NextSiblingElement();
    }
  }
}

void ParticleReader::ReadSpecies(TiXmlElement *particles) {
  TiXmlElement *species = particles->FirstChildElement("SPECIES");
  TiXmlElement *specie = species->FirstChildElement("S");

  while (specie) {
    nSpecies++;
    std::stringstream tagcontent;
    tagcontent << *specie;

    ASSERTL0(specie->Attribute("ID"), "Missing ID attribute in Species XML "
                                      "element: \n\t'" +
                                          tagcontent.str() + "'");
    std::string name = species->Attribute("NAME");
    ASSERTL0(!name.empty(),
             "NAME attribute must be non-empty in XML element:\n\t'" +
                 tagcontent.str() + "'");

    // generate a list of species.
    std::vector<std::string> species;
    bool valid = ParseUtils::GenerateVector(species, varStrings);

    ASSERTL0(valid, "Unable to process list of variable in XML "
                    "element \n\t'" +
                        tagcontent.str() + "'");

    if (varStrings.size()) {
      TiXmlElement *info = specie->FirstChildElement("P");

      while (info) {
        tagcontent.clear();
        tagcontent << *info;
        // read the property name
        ASSERTL0(info->Attribute("PROPERTY"),
                 "Missing PROPERTY attribute in "
                 "Species  '" +
                     name + "' in XML element: \n\t'" + tagcontent.str() + "'");
        std::string property = info->Attribute("PROPERTY");
        ASSERTL0(!property.empty(), "Species properties must have a "
                                    "non-empty name for Species '" +
                                        name + "' in XML element: \n\t'" +
                                        tagcontent.str() + "'");

        // make sure that solver property is capitalised
        std::string propertyUpper = boost::to_upper_copy(property);

        // read the value
        ASSERTL0(info->Attribute("VALUE"),
                 "Missing VALUE attribute in Species '" + name +
                     "' in XML element: \n\t" + tagcontent.str() + "'");
        std::string value = info->Attribute("VALUE");
        ASSERTL0(!value.empty(), "Species properties must have a "
                                 "non-empty value for Species '" +
                                     name + "' in XML element: \n\t'" +
                                     tagcontent.str() + "'");

        info = info->NextSiblingElement("P");
      }
      specie = specie->NextSiblingElement("S");
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

  ASSERTL0(boundaryConditionsElement, "Boundary conditions must be specified.");

  TiXmlElement *regionElement =
      boundaryConditionsElement->FirstChildElement("REGION");

  // Read C(Composite), P (Periodic) tags
  while (regionElement) {

    int boundaryRegionID;
    int err = regionElement->QueryIntAttribute("REF", &boundaryRegionID);
    ASSERTL0(err == TIXML_SUCCESS, "Error reading boundary region reference.");

    ASSERTL0(m_boundaryConditions.count(boundaryRegionID) == 0,
             "Boundary region '" + std::to_string(boundaryRegionID) +
                 "' appears multiple times.");

    // Find the boundary region corresponding to this ID.
    std::string boundaryRegionIDStr;
    std::ostringstream boundaryRegionIDStrm(boundaryRegionIDStr);
    boundaryRegionIDStrm << boundaryRegionID;

    if (m_boundaryRegions.count(boundaryRegionID) == 0) {
      regionElement = regionElement->NextSiblingElement("REGION");
      continue;
    }

    ASSERTL0(m_boundaryRegions.count(boundaryRegionID) == 1,
             "Boundary region " +
                 boost::lexical_cast<std::string>(boundaryRegionID) + " not found");

    // Find the communicator that belongs to this ID
    LU::CommSharedPtr boundaryRegionComm =
        m_boundaryCommunicators[boundaryRegionID];

    TiXmlElement *conditionElement = regionElement->FirstChildElement();
    std::vector<std::string> vars = m_session->GetVariables();

    while (conditionElement) {
      // Check type.
      std::string conditionType = conditionElement->Value();
      std::string attrData;
      bool isTimeDependent = false;

      // All have var specified, or else all variables are zero.
      TiXmlAttribute *attr = conditionElement->FirstAttribute();

      std::vector<std::string>::iterator iter;
      std::string attrName;

      attrData = conditionElement->Attribute("SPECIES");

      if (!attrData.empty()) {
        iter = std::find(vars.begin(), vars.end(), attrData);
        ASSERTL0(iter != vars.end(),
                 (std::string("Cannot find variable: ") + attrData).c_str());
      }

      if (conditionType == "C") {
        if (attrData.empty()) {
          // All species are reflect.
          for (auto &varIter : vars) {
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

                equation = attrData;
              } else {
                ASSERTL0(false, (std::string("Unknown boundary "
                                             "condition attribute: ") +
                                 attrName)
                                    .c_str());
              }
              attr = attr->Next();
            }
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
                    periodicBndRegionIndexStr.c_str(), periodicBndRegionIndex);

                ASSERTL0(parseGood && (periodicBndRegionIndex.size() == 1),
                         (std::string("Unable to read periodic boundary "
                                      "condition for boundary "
                                      "region: ") +
                          boundaryRegionIDStrm.str())
                             .c_str());
              }
              attr = attr->Next();
            }
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
}
} // namespace NESO::Particles
#endif