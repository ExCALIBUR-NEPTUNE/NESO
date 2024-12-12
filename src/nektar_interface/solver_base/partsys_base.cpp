#include "../../../include/nektar_interface/solver_base/partsys_base.hpp"
#include <SolverUtils/EquationSystem.h>
#include <mpi.h>
#include <nektar_interface/geometry_transport/halo_extension.hpp>
#include <nektar_interface/particle_interface.hpp>
#include <neso_particles.hpp>
#include <type_traits>

namespace NESO::Particles {

ParticleSystemFactory &GetParticleSystemFactory() {
  static ParticleSystemFactory instance;
  return instance;
}

PartSysBase::PartSysBase(const LU::SessionReaderSharedPtr session,
                         const SD::MeshGraphSharedPtr graph, MPI_Comm comm,
                         PartSysOptions options)
    : session(session), graph(graph), comm(comm),
      ndim(graph->GetSpaceDimension()) {

  read_params();

  // Store options
  this->options = options;

  // Create interface between particles and nektar++
  this->particle_mesh_interface =
      std::make_shared<ParticleMeshInterface>(graph, 0, this->comm);
  extend_halos_fixed_offset(this->options.extend_halos_offset,
                            this->particle_mesh_interface);
  this->sycl_target =
      std::make_shared<SYCLTarget>(0, particle_mesh_interface->get_comm());
  this->nektar_graph_local_mapper = std::make_shared<NektarGraphLocalMapper>(
      this->sycl_target, this->particle_mesh_interface);
  this->domain = std::make_shared<Domain>(this->particle_mesh_interface,
                                          this->nektar_graph_local_mapper);

  // // Create ParticleGroup
  // this->particle_group =
  //     std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  // Set up map between cell indices
  this->cell_id_translation = std::make_shared<CellIDTranslation>(
      this->sycl_target, this->particle_group->cell_id_dat,
      this->particle_mesh_interface);
}

/**
 * @details For each entry in the param_vals map (constructed via report_param),
 * write the value to stdout
 * @see also report_param()
 */
void PartSysBase::add_params_report() {

  std::cout << "Particle settings:" << std::endl;
  for (auto const &[param_lbl, param_str_val] : this->param_vals_to_report) {
    std::cout << "  " << param_lbl << ": " << param_str_val << std::endl;
  }
  std::cout << "============================================================="
               "=========="
            << std::endl
            << std::endl;
}

void PartSysBase::free() {
  if (this->h5part) {
    this->h5part->close();
  }
  this->particle_group->free();
  this->sycl_target->free();
  this->particle_mesh_interface->free();
}

bool PartSysBase::is_output_step(int step) {
  return this->output_freq > 0 && (step % this->output_freq) == 0;
}

void PartSysBase::read_params() {

  // Read total number of particles / number per cell from config
  int num_parts_per_cell, num_parts_tot;
  this->session->LoadParameter(NUM_PARTS_TOT_STR, num_parts_tot, -1);
  this->session->LoadParameter(NUM_PARTS_PER_CELL_STR, num_parts_per_cell, -1);

  if (num_parts_tot > 0) {
    this->num_parts_tot = num_parts_tot;
    if (num_parts_per_cell > 0) {
      nprint("Ignoring value of '" + NUM_PARTS_PER_CELL_STR +
             "' because  "
             "'" +
             NUM_PARTS_TOT_STR + "' was specified.");
    }
  } else {
    if (num_parts_per_cell > 0) {
      // Determine the global number of elements
      const int num_elements_local = this->graph->GetNumElements();
      int num_elements_global;
      MPICHK(MPI_Allreduce(&num_elements_local, &num_elements_global, 1,
                           MPI_INT, MPI_SUM, this->comm));

      // compute the global number of particles
      this->num_parts_tot = ((int64_t)num_elements_global) * num_parts_per_cell;

      report_param("Number of particles per cell/element", num_parts_per_cell);
    } else {
      nprint("Particles disabled (Neither '" + NUM_PARTS_TOT_STR +
             "' or "
             "'" +
             NUM_PARTS_PER_CELL_STR + "' are set)");
    }
  }
  report_param("Total number of particles", this->num_parts_tot);

  // Output frequency
  // ToDo Should probably be unsigned, but complicates use of LoadParameter
  this->session->LoadParameter(PART_OUTPUT_FREQ_STR, this->output_freq, 0);
  report_param("Output frequency (steps)", this->output_freq);
}

void PartSysBase::write(const int step) {
  if (this->h5part) {
    if (this->sycl_target->comm_pair.rank_parent == 0) {
      nprint("Writing particle properties at step", step);
    }
    this->h5part->write();
  } else {
    if (this->sycl_target->comm_pair.rank_parent == 0) {
      nprint("Ignoring call to write particle data because an output file "
             "wasn't set up. init_output() not called?");
    }
  }
};

void PartSysBase::InitSpec() { this->particle_spec = ParticleSpec{}; }

void PartSysBase::InitObject() {
  // Create ParticleSpec
  this->InitSpec();
  // Create ParticleGroup
  this->particle_group = std::make_shared<ParticleGroup>(
      this->domain, this->particle_spec, this->sycl_target);
}

void PartSysBase::ReadParameters(TiXmlElement *particles) {
  m_parameters.clear();

  TiXmlElement *parameters = particles->FirstChildElement("PARAMETERS");

  // See if we have parameters defined.  They are optional so we go on
  // if not.
  if (parameters) {
    TiXmlElement *parameter = parameters->FirstChildElement("P");

    ParameterMap caseSensitiveParameters;

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

void PartSysBase::ReadSpecies(TiXmlElement *particles) {
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

void PartSysBase::ReadBoundary(TiXmlElement *particles) {
  // Protect against multiple reads.
  if (m_boundaryConditions.size() != 0) {
    return;
  }

  // Read REGION tags
  TiXmlElement *boundaryConditionsElement =
      particles->FirstChildElement("BOUNDARYINTERACTION");
  LibUtilities::SessionReader::GetXMLElementTimeLevel(boundaryConditionsElement,
                                                      session->GetTimeLevel());
  ASSERTL0(boundaryConditionsElement, "Boundary conditions must be specified.");

  TiXmlElement *regionElement =
      boundaryConditionsElement->FirstChildElement("REGION");

  // Read C(Composite), P (Periodic) tags
  while (regionElement) {
    BoundaryConditionMapShPtr boundaryConditions =
        MemoryManager<BoundaryConditionMap>::AllocateSharedPtr();

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
                 boost::lexical_cast<string>(boundaryRegionID) + " not found");

    // Find the communicator that belongs to this ID
    LibUtilities::CommSharedPtr boundaryRegionComm =
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

void PartSysBase::ReadParticles() {
  // Check we actually have a document loaded.
  ASSERTL0(&session->GetDocument(), "No XML document loaded.");

  TiXmlHandle docHandle(&session->GetDocument());
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
