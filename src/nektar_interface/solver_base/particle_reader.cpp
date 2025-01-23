///////////////////////////////////////////////////////////////////////////////
//
// File: particle_reader.cpp
// Based on nektar/library/LibUtilities/BasicUtils/SessionReader.cpp
// at https://gitlab.nektar.info/nektar by "Nektar++ developers"
//
///////////////////////////////////////////////////////////////////////////////

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

using Nektar::ParseUtils;
template <typename DataType>
using MemoryManager = Nektar::MemoryManager<DataType>;

namespace NESO::Particles {
/**
 *
 */
void ParticleReader::parse_equals(const std::string &line, std::string &lhs,
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

void ParticleReader::read_info() {
  NESOASSERT(&this->session->GetDocument(), "No XML document loaded.");

  TiXmlHandle docHandle(&this->session->GetDocument());
  TiXmlElement *particles;

  // Look for all data in PARTICLES block.
  particles = docHandle.FirstChildElement("NEKTAR")
                  .FirstChildElement("PARTICLES")
                  .Element();
  if (!particles) {
    return;
  }
  this->particle_info.clear();

  TiXmlElement *particle_info_element = particles->FirstChildElement("INFO");

  if (particle_info_element) {
    TiXmlElement *particle_info_i =
        particle_info_element->FirstChildElement("I");

    while (particle_info_i) {
      std::stringstream tagcontent;
      tagcontent << *particle_info_i;
      // read the property name
      NESOASSERT(particle_info_i->Attribute("PROPERTY"),
                 "Missing PROPERTY attribute in particle info "
                 "XML element: \n\t'" +
                     tagcontent.str() + "'");
      std::string particle_property = particle_info_i->Attribute("PROPERTY");
      NESOASSERT(!particle_property.empty(),
                 "PROPERTY attribute must be non-empty in XML "
                 "element: \n\t'" +
                     tagcontent.str() + "'");

      // make sure that solver property is capitalised
      std::string particle_property_upper =
          boost::to_upper_copy(particle_property);

      // read the value
      NESOASSERT(particle_info_i->Attribute("VALUE"),
                 "Missing VALUE attribute in particle info "
                 "XML element: \n\t'" +
                     tagcontent.str() + "'");
      std::string particle_value = particle_info_i->Attribute("VALUE");
      NESOASSERT(!particle_value.empty(),
                 "VALUE attribute must be non-empty in XML "
                 "element: \n\t'" +
                     tagcontent.str() + "'");

      // Set Variable
      this->particle_info[particle_property_upper] = particle_value;
      particle_info_i = particle_info_i->NextSiblingElement("I");
    }
  }
}

/**
 *
 */
bool ParticleReader::defines_info(const std::string &name) const {
  std::string name_upper = boost::to_upper_copy(name);
  return this->particle_info.find(name_upper) != this->particle_info.end();
}
/**
 * If the parameter is not defined, termination occurs. Therefore, the
 * parameters existence should be tested for using #DefinesParameter
 * before calling this function.
 *
 * @param   name       The name of a floating-point parameter.
 * @returns The value of the floating-point parameter.
 */
const std::string &ParticleReader::get_info(const std::string &name) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto info_iter = this->particle_info.find(name);

  NESOASSERT(info_iter != this->particle_info.end(),
             "Unable to find requested info: " + name);

  return info_iter->second;
}

void ParticleReader::read_parameters(TiXmlElement *particles) {
  this->parameters.clear();

  TiXmlElement *parameters = particles->FirstChildElement("PARAMETERS");

  // See if we have parameters defined.  They are optional so we go on
  // if not.
  if (parameters) {
    TiXmlElement *parameter_p = parameters->FirstChildElement("P");

    // Multiple nodes will only occur if there is a comment in
    // between definitions.
    while (parameter_p) {
      std::stringstream tagcontent;
      tagcontent << *parameter_p;
      TiXmlNode *node = parameter_p->FirstChild();

      while (node && node->Type() != TiXmlNode::TINYXML_TEXT) {
        node = node->NextSibling();
      }

      if (node) {
        // Format is "paramName = value"
        std::string line = node->ToText()->Value(), lhs, rhs;

        try {
          parse_equals(line, lhs, rhs);
        } catch (...) {
          NESOASSERT(false, "Syntax error in parameter expression '" + line +
                                "' in XML element: \n\t'" + tagcontent.str() +
                                "'");
        }

        // We want the list of parameters to have their RHS
        // evaluated, so we use the expression evaluator to do
        // the dirty work.
        if (!lhs.empty() && !rhs.empty()) {
          NekDouble value = 0.0;
          try {
            LU::Equation expession(this->interpreter, rhs);
            value = expession.Evaluate();
          } catch (const std::runtime_error &) {
            NESOASSERT(false, "Error evaluating parameter expression"
                              " '" +
                                  rhs + "' in XML element: \n\t'" +
                                  tagcontent.str() + "'");
          }
          this->interpreter->SetParameter(lhs, value);
          boost::to_upper(lhs);
          this->parameters[lhs] = value;
        }
      }
      parameter_p = parameter_p->NextSiblingElement();
    }
  }
}

/**
 *
 */
bool ParticleReader::defines_parameter(const std::string &name) const {
  std::string name_upper = boost::to_upper_copy(name);
  return this->parameters.find(name_upper) != this->parameters.end();
}

/**
 * If the parameter is not defined, termination occurs. Therefore, the
 * parameters existence should be tested for using #DefinesParameter
 * before calling this function.
 *
 * @param   name       The name of a floating-point parameter.
 * @returns The value of the floating-point parameter.
 */
const NekDouble &ParticleReader::get_parameter(const std::string &name) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto param_iter = this->parameters.find(name_upper);

  NESOASSERT(param_iter != this->parameters.end(),
             "Unable to find requested parameter: " + name);

  return param_iter->second;
}

void ParticleReader::read_species_functions(TiXmlElement *specie,
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
    NESOASSERT(function->Attribute("NAME"),
               "Functions must have a NAME attribute defined in XML "
               "element: \n\t'" +
                   tagcontent.str() + "'");
    std::string function_str = function->Attribute("NAME");
    NESOASSERT(!function_str.empty(),
               "Functions must have a non-empty name in XML "
               "element: \n\t'" +
                   tagcontent.str() + "'");

    // Store function names in uppercase to remain case-insensitive.
    boost::to_upper(function_str);

    // Retrieve first entry (variable, or file)
    TiXmlElement *element = function;
    TiXmlElement *variable = element->FirstChildElement();

    // Create new function structure with default type of none.
    LU::FunctionVariableMap function_var_map;

    // Process all entries in the function block
    while (variable) {
      LU::FunctionVariableDefinition func_def;
      std::string condition_type = variable->Value();

      // If no var is specified, assume wildcard
      std::string variable_str;
      if (!variable->Attribute("VAR")) {
        variable_str = "*";
      } else {
        variable_str = variable->Attribute("VAR");
      }

      // Parse list of variables
      std::vector<std::string> variable_list;
      ParseUtils::GenerateVector(variable_str, variable_list);

      // If no domain is specified, put to 0
      std::string domain_str;
      if (!variable->Attribute("DOMAIN")) {
        domain_str = "0";
      } else {
        domain_str = variable->Attribute("DOMAIN");
      }

      // Parse list of domains
      std::vector<std::string> var_split;
      std::vector<unsigned int> domain_list;
      ParseUtils::GenerateSeqVector(domain_str, domain_list);

      // if no evars is specified, put "x y z t"
      std::string evars_str = "x y z t";
      if (variable->Attribute("EVARS")) {
        evars_str = evars_str + std::string(" ") + variable->Attribute("EVARS");
      }

      // Expressions are denoted by E
      if (condition_type == "E") {
        func_def.m_type = LU::eFunctionTypeExpression;

        // Expression must have a VALUE.
        NESOASSERT(variable->Attribute("VALUE"),
                   "Attribute VALUE expected for function '" + function_str +
                       "'.");
        std::string fcn_str = variable->Attribute("VALUE");
        NESOASSERT(!fcn_str.empty(),
                   (std::string("Expression for var: ") + variable_str +
                    std::string(" must be specified."))
                       .c_str());

        // set expression
        func_def.m_expression = MemoryManager<LU::Equation>::AllocateSharedPtr(
            this->interpreter, fcn_str, evars_str);
      }

      // Files are denoted by F
      else if (condition_type == "F") {
        // Check if transient or not
        if (variable->Attribute("TIMEDEPENDENT") &&
            boost::lexical_cast<bool>(variable->Attribute("TIMEDEPENDENT"))) {
          func_def.m_type = LU::eFunctionTypeTransientFile;
        } else {
          func_def.m_type = LU::eFunctionTypeFile;
        }

        // File must have a FILE.
        NESOASSERT(variable->Attribute("FILE"),
                   "Attribute FILE expected for function '" + function_str +
                       "'.");
        std::string filename_str = variable->Attribute("FILE");
        NESOASSERT(!filename_str.empty(),
                   "A filename must be specified for the FILE "
                   "attribute of function '" +
                       function_str + "'.");

        std::vector<std::string> f_split;
        boost::split(f_split, filename_str, boost::is_any_of(":"));
        NESOASSERT(f_split.size() == 1 || f_split.size() == 2,
                   "Incorrect filename specification in function " +
                       function_str +
                       "'. "
                       "Specify variables inside file as: "
                       "filename:var1,var2");

        // set the filename
        fs::path fullpath = f_split[0];
        func_def.m_filename = fullpath.string();

        if (f_split.size() == 2) {
          NESOASSERT(variable_list[0] != "*",
                     "Filename variable mapping not valid "
                     "when using * as a variable inside "
                     "function '" +
                         function_str + "'.");

          boost::split(var_split, f_split[1], boost::is_any_of(","));
          NESOASSERT(var_split.size() == variable_list.size(),
                     "Filename variables should contain the "
                     "same number of variables defined in "
                     "VAR in function " +
                         function_str + "'.");
        }
      }

      // Nothing else supported so throw an error
      else {
        std::stringstream tagcontent;
        tagcontent << *variable;

        NESOASSERT(false, "Identifier " + condition_type + " in function " +
                              std::string(function->Attribute("NAME")) +
                              " is not recognised in XML element: \n\t'" +
                              tagcontent.str() + "'");
      }

      // Add variables to function
      for (unsigned int i = 0; i < variable_list.size(); ++i) {
        for (unsigned int j = 0; j < domain_list.size(); ++j) {
          // Check it has not already been defined
          std::pair<std::string, int> key(variable_list[i], domain_list[j]);
          auto fcns_iter = function_var_map.find(key);
          NESOASSERT(fcns_iter == function_var_map.end(),
                     "Error setting expression '" + variable_list[i] +
                         " in domain " + std::to_string(domain_list[j]) +
                         "' in function '" + function_str +
                         "'. "
                         "Expression has already been defined.");

          if (var_split.size() > 0) {
            LU::FunctionVariableDefinition func_def2 = func_def;
            func_def2.m_fileVariable = var_split[i];
            function_var_map[key] = func_def2;
          } else {
            function_var_map[key] = func_def;
          }
        }
      }
      variable = variable->NextSiblingElement();
    }

    // Add function definition to map
    functions[function_str] = function_var_map;
    function = function->NextSiblingElement("FUNCTION");
  }
}

void ParticleReader::read_species(TiXmlElement *particles) {
  TiXmlElement *species = particles->FirstChildElement("SPECIES");
  if (species) {
    TiXmlElement *specie = species->FirstChildElement("S");

    while (specie) {
      std::stringstream tagcontent;
      tagcontent << *specie;
      std::string id = specie->Attribute("ID");
      NESOASSERT(!id.empty(), "Missing ID attribute in Species XML "
                              "element: \n\t'" +
                                  tagcontent.str() + "'");

      std::string name = specie->Attribute("NAME");
      NESOASSERT(!name.empty(),
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
            parse_equals(line, lhs, rhs);
          } catch (...) {
            NESOASSERT(false, "Syntax error in parameter expression '" + line +
                                  "' in XML element: \n\t'" + tagcontent.str() +
                                  "'");
          }

          // We want the list of parameters to have their RHS
          // evaluated, so we use the expression evaluator to do
          // the dirty work.
          if (!lhs.empty() && !rhs.empty()) {
            NekDouble value = 0.0;
            try {
              LU::Equation expession(this->interpreter, rhs);
              value = expession.Evaluate();
            } catch (const std::runtime_error &) {
              NESOASSERT(false, "Error evaluating parameter expression"
                                " '" +
                                    rhs + "' in XML element: \n\t'" +
                                    tagcontent.str() + "'");
            }
            this->interpreter->SetParameter(lhs, value);
            boost::to_upper(lhs);
            std::get<1>(species_map)[lhs] = value;
          }
        }
        parameter = parameter->NextSiblingElement();
      }

      read_species_functions(specie, std::get<2>(species_map));
      specie = specie->NextSiblingElement("S");

      this->species[std::stoi(id)] = species_map;
    }
  }
}

void ParticleReader::read_boundary(TiXmlElement *particles) {
  // Protect against multiple reads.
  if (this->boundary_conditions.size() != 0) {
    return;
  }

  // Read REGION tags
  TiXmlElement *boundary_conditions_element =
      particles->FirstChildElement("BOUNDARYINTERACTION");

  if (boundary_conditions_element) {
    TiXmlElement *region_element =
        boundary_conditions_element->FirstChildElement("REGION");

    // Read C(Composite), P (Periodic) tags
    while (region_element) {
      SpeciesBoundaryList species_boundary_conditions;

      int boundary_region_id;
      int err = region_element->QueryIntAttribute("REF", &boundary_region_id);
      NESOASSERT(err == TIXML_SUCCESS,
                 "Error reading boundary region reference.");

      NESOASSERT(this->boundary_conditions.count(boundary_region_id) == 0,
                 "Boundary region '" + std::to_string(boundary_region_id) +
                     "' appears multiple times.");

      // Find the boundary region corresponding to this ID.
      std::string boundary_region_id_str;
      std::ostringstream boundary_region_id_strm(boundary_region_id_str);
      boundary_region_id_strm << boundary_region_id;

      TiXmlElement *condition_element = region_element->FirstChildElement();

      while (condition_element) {
        // Check type.
        std::string condition_type = condition_element->Value();
        std::string attr_data;
        bool is_time_dependent = false;

        // All species are specified, or else all species are zero.
        TiXmlAttribute *attr = condition_element->FirstAttribute();

        SpeciesMapList::iterator iter;
        std::string attr_name;
        attr_data = condition_element->Attribute("SPECIES");
        int species_id = std::stoi(attr_data);
        if (condition_type == "C") {
          if (attr_data.empty()) {
            // All species are reflect.
            for (const auto &species : this->species) {
              species_boundary_conditions[species.first] =
                  ParticleBoundaryConditionType::eReflective;
            }
          } else {
            if (attr) {
              std::string equation, user_defined, filename;

              while (attr) {

                attr_name = attr->Name();

                if (attr_name == "SPECIES") {
                  // if VAR do nothing
                } else if (attr_name == "VALUE") {
                  NESOASSERT(
                      attr_name == "VALUE",
                      (std::string("Unknown attribute: ") + attr_name).c_str());

                  attr_data = attr->Value();
                  NESOASSERT(!attr_data.empty(),
                             "VALUE attribute must be specified.");

                } else {
                  NESOASSERT(false, (std::string("Unknown boundary "
                                                 "condition attribute: ") +
                                     attr_name)
                                        .c_str());
                }
                attr = attr->Next();
              }
              species_boundary_conditions[species_id] =
                  ParticleBoundaryConditionType::eReflective;
            }
          }
        }

        else if (condition_type == "P") {
          if (attr_data.empty()) {
            NESOASSERT(false, "Periodic boundary conditions should "
                              "be explicitly defined");
          } else {
            if (attr) {
              std::string user_defined;
              std::vector<unsigned int> periodic_bnd_region_index;
              while (attr) {
                attr_name = attr->Name();

                if (attr_name == "SPECIES") {
                  // if VAR do nothing
                } else if (attr_name == "user_definedTYPE") {
                  // Do stuff for the user defined attribute
                  attr_data = attr->Value();
                  NESOASSERT(!attr_data.empty(),
                             "user_definedTYPE attribute must have "
                             "associated value.");

                  user_defined = attr_data;
                  is_time_dependent =
                      boost::iequals(attr_data, "TimeDependent");
                } else if (attr_name == "VALUE") {
                  attr_data = attr->Value();
                  NESOASSERT(!attr_data.empty(),
                             "VALUE attribute must have associated "
                             "value.");

                  int beg = attr_data.find_first_of("[");
                  int end = attr_data.find_first_of("]");
                  std::string periodic_bnd_region_index_str =
                      attr_data.substr(beg + 1, end - beg - 1);
                  NESOASSERT(beg < end,
                             (std::string("Error reading periodic "
                                          "boundary region definition "
                                          "for boundary region: ") +
                              boundary_region_id_strm.str())
                                 .c_str());

                  bool parse_good = ParseUtils::GenerateSeqVector(
                      periodic_bnd_region_index_str.c_str(),
                      periodic_bnd_region_index);

                  NESOASSERT(parse_good &&
                                 (periodic_bnd_region_index.size() == 1),
                             (std::string("Unable to read periodic boundary "
                                          "condition for boundary "
                                          "region: ") +
                              boundary_region_id_strm.str())
                                 .c_str());
                }
                attr = attr->Next();
              }
              species_boundary_conditions[species_id] =
                  ParticleBoundaryConditionType::ePeriodic;
            } else {
              NESOASSERT(false, "Periodic boundary conditions should "
                                "be explicitly defined");
            }
          }
        }

        condition_element = condition_element->NextSiblingElement();
      }

      this->boundary_conditions[boundary_region_id] =
          species_boundary_conditions;
      region_element = region_element->NextSiblingElement("REGION");
    }
  }
}

void ParticleReader::read_reactions(TiXmlElement *particles) {
  TiXmlElement *reactions_element = particles->FirstChildElement("REACTIONS");
  if (reactions_element) {
    TiXmlElement *reaction_r = reactions_element->FirstChildElement("R");

    while (reaction_r) {
      std::stringstream tagcontent;
      tagcontent << *reaction_r;
      std::string id = reaction_r->Attribute("ID");
      NESOASSERT(!id.empty(), "Missing ID attribute in Reaction XML "
                              "element: \n\t'" +
                                  tagcontent.str() + "'");
      std::string type = reaction_r->Attribute("TYPE");
      NESOASSERT(!type.empty(),
                 "TYPE attribute must be non-empty in XML element:\n\t'" +
                     tagcontent.str() + "'");
      ReactionMap reaction_map;
      std::get<0>(reaction_map) = type;
      std::string species = reaction_r->Attribute("SPECIES");
      std::vector<std::string> species_list;
      boost::split(species_list, species, boost::is_any_of(","));

      for (const auto &s : species_list) {
        NESOASSERT(this->species.find(std::stoi(s)) != this->species.end(),
                   "Species '" + s +
                       "' not found.  Ensure it is specified under the "
                       "<SPECIES> tag");
        std::get<1>(reaction_map).push_back(std::stoi(s));
      }

      TiXmlElement *parameter = reaction_r->FirstChildElement("P");
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
            parse_equals(line, lhs, rhs);
          } catch (...) {
            NESOASSERT(false, "Syntax error in parameter expression '" + line +
                                  "' in XML element: \n\t'" + tagcontent.str() +
                                  "'");
          }

          // We want the list of parameters to have their RHS
          // evaluated, so we use the expression evaluator to do
          // the dirty work.
          if (!lhs.empty() && !rhs.empty()) {
            NekDouble value = 0.0;
            try {
              LU::Equation expession(this->interpreter, rhs);
              value = expession.Evaluate();
            } catch (const std::runtime_error &) {
              NESOASSERT(false, "Error evaluating parameter expression"
                                " '" +
                                    rhs + "' in XML element: \n\t'" +
                                    tagcontent.str() + "'");
            }
            this->interpreter->SetParameter(lhs, value);
            boost::to_upper(lhs);
            std::get<2>(reaction_map)[lhs] = value;
          }
        }
        parameter = parameter->NextSiblingElement();
      }

      reaction_r = reaction_r->NextSiblingElement("R");
      this->reactions[std::stoi(id)] = reaction_map;
    }
  }
}

void ParticleReader::read_particles() {
  // Check we actually have a document loaded.
  NESOASSERT(&this->session->GetDocument(), "No XML document loaded.");

  TiXmlHandle docHandle(&this->session->GetDocument());
  TiXmlElement *particles;

  // Look for all data in PARTICLES block.
  particles = docHandle.FirstChildElement("NEKTAR")
                  .FirstChildElement("PARTICLES")
                  .Element();

  if (!particles) {
    return;
  }
  read_parameters(particles);
  read_species(particles);
  read_boundary(particles);
  read_reactions(particles);
}

void ParticleReader::load_species_parameter(const int species,
                                            const std::string &name,
                                            int &var) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto map = std::get<1>(this->species.at(species));
  auto param_iter = map.find(name_upper);
  NESOASSERT(param_iter != map.end(),
             "Required parameter '" + name + "' not specified in session.");
  NekDouble param = round(param_iter->second);
  var = LU::checked_cast<int>(param);
}

void ParticleReader::load_species_parameter(const int species,
                                            const std::string &name,
                                            NekDouble &var) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto map = std::get<1>(this->species.at(species));
  auto param_iter = map.find(name_upper);
  NESOASSERT(param_iter != map.end(),
             "Required parameter '" + name + "' not specified in session.");
  var = param_iter->second;
}

void ParticleReader::load_reaction_parameter(const int reaction,
                                             const std::string &name,
                                             int &var) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto map = std::get<2>(this->reactions.at(reaction));
  auto param_iter = map.find(name_upper);
  NESOASSERT(param_iter != map.end(),
             "Required parameter '" + name + "' not specified in session.");
  NekDouble param = round(param_iter->second);
  var = LU::checked_cast<int>(param);
}

void ParticleReader::load_reaction_parameter(const int reaction,
                                             const std::string &name,
                                             NekDouble &var) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto map = std::get<2>(this->reactions.at(reaction));
  auto param_iter = map.find(name_upper);
  NESOASSERT(param_iter != map.end(),
             "Required parameter '" + name + "' not specified in session.");
  var = param_iter->second;
}

/**
 *
 */
void ParticleReader::load_parameter(const std::string &name, int &var) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto param_iter = this->parameters.find(name_upper);
  NESOASSERT(param_iter != this->parameters.end(),
             "Required parameter '" + name + "' not specified in session.");
  NekDouble param = round(param_iter->second);
  var = LU::checked_cast<int>(param);
}

/**
 *
 */
void ParticleReader::load_parameter(const std::string &name, int &var,
                                    const int &pDefault) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto param_iter = this->parameters.find(name_upper);
  if (param_iter != this->parameters.end()) {
    NekDouble param = round(param_iter->second);
    var = LU::checked_cast<int>(param);
  } else {
    var = pDefault;
  }
}

/**
 *
 */
void ParticleReader::load_parameter(const std::string &name,
                                    size_t &var) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto param_iter = this->parameters.find(name_upper);
  NESOASSERT(param_iter != this->parameters.end(),
             "Required parameter '" + name + "' not specified in session.");
  NekDouble param = round(param_iter->second);
  var = LU::checked_cast<int>(param);
}

/**
 *
 */
void ParticleReader::load_parameter(const std::string &name, size_t &var,
                                    const size_t &pDefault) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto param_iter = this->parameters.find(name_upper);
  if (param_iter != this->parameters.end()) {
    NekDouble param = round(param_iter->second);
    var = LU::checked_cast<int>(param);
  } else {
    var = pDefault;
  }
}

/**
 *
 */
void ParticleReader::load_parameter(const std::string &name,
                                    NekDouble &var) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto param_iter = this->parameters.find(name_upper);
  NESOASSERT(param_iter != this->parameters.end(),
             "Required parameter '" + name + "' not specified in session.");
  var = param_iter->second;
}

/**
 *
 */
void ParticleReader::load_parameter(const std::string &name, NekDouble &var,
                                    const NekDouble &pDefault) const {
  std::string name_upper = boost::to_upper_copy(name);
  auto param_iter = this->parameters.find(name_upper);
  if (param_iter != this->parameters.end()) {
    var = param_iter->second;
  } else {
    var = pDefault;
  }
}
} // namespace NESO::Particles
