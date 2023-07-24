#ifndef __UNITCONVERTER_H_
#define __UNITCONVERTER_H_

#include <math.h>

// #pragma once

namespace {
constexpr double SPEED_OF_LIGHT = 299792458.0;
constexpr double ELEMENTARY_MASS = 9.1093837e-31;
constexpr double ELEMENTARY_CHARGE = 1.60217663e-19;

constexpr double MU_0 = 4.0e-7 * M_PI;
constexpr double EPSILON_0 = 1.0 / MU_0 / SPEED_OF_LIGHT / SPEED_OF_LIGHT;
} // namespace

class UnitConverter {
public:
  UnitConverter(double lengthScale) : m_lengthScale(lengthScale) {
    m_timeScale = m_lengthScale / SPEED_OF_LIGHT;
    m_electricPotentialScale = ELEMENTARY_MASS *
                               std::pow(m_lengthScale / m_timeScale, 2) /
                               ELEMENTARY_CHARGE;
    m_chargeDensityScale =
        m_electricPotentialScale * EPSILON_0 / std::pow(m_lengthScale, 2);
    m_numberDensityScale = m_chargeDensityScale / ELEMENTARY_CHARGE;
    m_magneticPotentialScale =
        m_timeScale * m_electricPotentialScale / m_lengthScale;
    m_currentDensityScale = m_chargeDensityScale * m_lengthScale / m_timeScale;
  }

  inline double si_numberdensity_to_sim(double n) const;
  inline double si_velocity_to_sim(double v) const;
  inline double si_temperature_ev_to_sim(double tev) const;
  inline double si_magneticfield_to_sim(double B) const;
  inline double velocity_to_si(double v) const;
  inline double density_to_si(double v) const;
  inline double time_to_si(double t) const;
  inline double timescale() const;
  inline double lengthscale() const;

  friend std::ostream &operator<<(std::ostream &os, const UnitConverter &uc) {
    return os << "Length scale = " << uc.m_lengthScale << "\n"
              << "Time scale = " << uc.m_timeScale << "\n"
              << "Electric potential scale = " << uc.m_electricPotentialScale
              << "\n"
              << "Charge density scale = " << uc.m_chargeDensityScale << "\n"
              << "Number density scale = " << uc.m_numberDensityScale << "\n"
              << "Magnetic potential scale = " << uc.m_magneticPotentialScale
              << "\n"
              << "Current density scale = " << uc.m_currentDensityScale << "\n"
              << "The speed of light is = "
              << SPEED_OF_LIGHT / uc.m_lengthScale * uc.m_timeScale;
  }

private:
  double m_lengthScale; // multiply position in simulation by this number to get
                        // metres
  double
      m_timeScale; // multiple time in simulation by this number to get seconds
  double m_electricPotentialScale; // scaling for phi as above...
  double m_chargeDensityScale;     // scaling for rho
  double m_numberDensityScale;     // scaling for number density...
  double m_magneticPotentialScale; // for A
  double m_currentDensityScale;    // for J
};

inline double UnitConverter::si_numberdensity_to_sim(double n) const {
  return n / m_numberDensityScale;
}
inline double UnitConverter::si_velocity_to_sim(double v) const {
  return v / (m_lengthScale / m_timeScale);
}
inline double UnitConverter::si_temperature_ev_to_sim(double tev) const {
  return tev * ELEMENTARY_CHARGE /
         (ELEMENTARY_MASS * std::pow(SPEED_OF_LIGHT, 2));
}
inline double UnitConverter::si_magneticfield_to_sim(double B) const {
  return B * m_lengthScale / m_magneticPotentialScale;
}

inline double UnitConverter::velocity_to_si(double v) const {
  return v * m_lengthScale / m_timeScale;
}
inline double UnitConverter::density_to_si(double n) const {
  return n * m_numberDensityScale;
}
inline double UnitConverter::time_to_si(double t) const {
  return t * m_timeScale;
}

inline double UnitConverter::timescale() const { return m_timeScale; }

inline double UnitConverter::lengthscale() const { return m_lengthScale; }

#endif
