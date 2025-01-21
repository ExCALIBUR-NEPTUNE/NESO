#include "nektar_interface/solver_base/particle_reader.hpp"
#include "test_helper_utilities.hpp"

TEST(ParticleReader, ReadParticles) {
  TestUtilities::TestResourceSession resource_session("unit_square_0_5.xml",
                                                      "particle_reader.xml");
  ParticleReaderSharedPtr reader =
      std::make_shared<ParticleReader>(resource_session.session);
  resource_session.session->InitSession();
  reader->ReadInfo();
  reader->ReadParticles();
  EXPECT_EQ(reader->GetInfo("PARTTYPE"), "TestParticleSystem");

  EXPECT_EQ(std::get<0>(reader->GetSpecies().at(0)), "Argon");

  double mass;
  reader->LoadSpeciesParameter(0, "Mass", mass);

  EXPECT_EQ(mass, 40);

  auto func = std::get<2>(reader->GetSpecies().at(1)).at("INITIALDISTRIBUTION");

  auto expr = func.at(std::make_pair("n", 0))
                  .m_expression;
  
  EXPECT_EQ(expr->Evaluate(13, 5), 8);

  EXPECT_EQ(reader->GetBoundaries().at(0).at(1),
            ParticleBoundaryConditionType::eReflective);

  EXPECT_EQ(std::get<0>(reader->GetReactions().at(0)), "Ionisation");

  EXPECT_EQ(std::get<1>(reader->GetReactions().at(2))[1], 1);
  double rate;
  reader->LoadReactionParameter(1, "Rate", rate);
  EXPECT_EQ(rate, 100);
}