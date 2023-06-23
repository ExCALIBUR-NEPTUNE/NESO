#ifndef PARTICLE_INITIAL_CONDITION_H
#define PARTICLE_INITIAL_CONDITION_H

#include <vector>

struct ParticleInitialConditions {
  double charge;
  double mass;
  double temperature;
  double drift;
  double pitch;
  double number_density;
  double weight;
};

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
    auto drift_speed = std::sqrt(2 * ic.drift / ic.mass);
    output += ic.charge * ic.number_density * drift_speed * ic.pitch;
  }
  return output;
}

#endif /* PARTICLE_INITIAL_CONDITION_H */
