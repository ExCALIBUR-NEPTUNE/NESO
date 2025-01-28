#include "nektar_interface/solver_base/particle_reader.hpp"
#include "test_helper_utilities.hpp"

TEST(ParticleReader, ReadParticles) {
  TestUtilities::TestResourceSession resource_session("unit_square_0_5.xml",
                                                      "particle_reader.xml");
  ParticleReaderSharedPtr reader =
      std::make_shared<ParticleReader>(resource_session.session);
  resource_session.session->InitSession();
  reader->read_info();
  reader->read_particles();
  EXPECT_EQ(reader->get_info("PARTTYPE"), "TestParticleSystem");

  EXPECT_EQ(std::get<0>(reader->get_species().at(0)), "Argon");

  double mass;
  reader->load_species_parameter(0, "Mass", mass);

  EXPECT_EQ(mass, 40);

  auto func =
      std::get<2>(reader->get_species().at(1)).at("INITIALDISTRIBUTION");

  auto expr = func.at(std::make_pair("n", 0)).m_expression;

  EXPECT_EQ(expr->Evaluate(13, 5), 8);

  EXPECT_EQ(reader->get_boundaries().at(0).at(1),
            ParticleBoundaryConditionType::eReflective);

  EXPECT_EQ(std::get<0>(reader->get_reactions().at(0)), "Ionisation");

  EXPECT_EQ(std::get<1>(reader->get_reactions().at(2))[1], 1);
  double rate;
  reader->load_reaction_parameter(1, "Rate", rate);
  EXPECT_EQ(rate, 100);
}