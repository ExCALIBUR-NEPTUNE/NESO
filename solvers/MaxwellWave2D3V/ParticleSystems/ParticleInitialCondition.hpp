#ifndef PARTICLE_INITIAL_CONDITION_H
#define PARTICLE_INITIAL_CONDITION_H

#include <iostream>
#include <vector>

struct ParticleInitialConditions {
  double charge;
  double mass;
  double temperature;
  double driftenergy;
  double pitch;
  double number_density;
  double weight;
};

std::ostream& operator<<(std::ostream& os, const ParticleInitialConditions& pic)
{
  return os << "Charge, " << pic.charge << ", and mass, " << pic.mass << "\n" <<
    "  Temperature = " << pic.temperature << "\n" <<
    "  Drift energy = " << pic.driftenergy << "\n" <<
    "  Pitch = " << pic.pitch << "\n" <<
    "  Number density = " << pic.number_density << "\n" <<
    "  Weight = " << pic.weight << "\n";
}

double total_number_density(const std::vector<ParticleInitialConditions>& ics) {
  double output = 0.0;
  for (const auto ic : ics) {
    output += ic.number_density;
  }
  return output;
}

double total_charge_density(const std::vector<ParticleInitialConditions>& ics) {
  double output = 0.0;
  for (const auto ic : ics) {
    output += ic.number_density * ic.charge;
  }
  return output;
}

double total_parallel_current_density(const std::vector<ParticleInitialConditions>& ics) {
  double output = 0.0;
  for (const auto ic : ics) {
    auto drift_speed = std::sqrt(2 * ic.driftenergy / ic.mass);
    output += ic.charge * ic.number_density * drift_speed * ic.pitch;
  }
  return output;
}

#endif /* PARTICLE_INITIAL_CONDITION_H */
