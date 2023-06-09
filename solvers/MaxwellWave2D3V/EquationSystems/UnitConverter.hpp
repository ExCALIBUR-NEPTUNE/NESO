#ifndef __UNITCONVERTER_H_
#define __UNITCONVERTER_H_

#include <math.h>

//#pragma once

namespace {
constexpr double SPEED_OF_LIGHT = 299792458.0;
constexpr double ELEMENTARY_MASS = 9.1093837e-31;
constexpr double ELEMENTARY_CHARGE = 1.60217663e-19;
            
constexpr double MU_0 = 4.0e-7 * M_PI;
constexpr double EPSILON_0 = 1.0 / MU_0 / SPEED_OF_LIGHT / SPEED_OF_LIGHT;
}

class UnitConverter {
  public:

    UnitConverter(double lengthscale) : m_lengthscale(lengthscale) {
      m_timescale = m_lengthscale / SPEED_OF_LIGHT;
      m_electricpotentialscale = ELEMENTARY_MASS * std::pow(m_lengthscale / m_timescale, 2)
        / ELEMENTARY_CHARGE;
      m_chargedensityscale = m_electricpotentialscale * EPSILON_0 / std::pow(m_lengthscale, 2);
      m_numberdensityscale = m_chargedensityscale / ELEMENTARY_CHARGE;
      m_magneticpotentialscale = m_timescale * m_electricpotentialscale / m_lengthscale;
      m_currentdensityscale = m_chargedensityscale * m_lengthscale / m_timescale;
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

    friend std::ostream& operator<<(std::ostream &os, const UnitConverter &uc) {
      return os << "Length scale = " << uc.m_lengthscale << "\n"
                << "Time scale = " << uc.m_timescale << "\n"
                << "Electric potential scale = " << uc.m_electricpotentialscale << "\n"
                << "Charge density scale = " << uc.m_chargedensityscale << "\n"
                << "Number density scale = " << uc.m_numberdensityscale << "\n"
                << "Magnetic potential scale = " << uc.m_magneticpotentialscale << "\n"
                << "Current density scale = " << uc.m_currentdensityscale << "\n"
                << "The speed of light is = " << SPEED_OF_LIGHT / uc.m_lengthscale * uc.m_timescale;
    }

  private:
    double m_lengthscale; // multiply position in simulation by this number to get metres
    double m_timescale; // multiple time in simulation by this number to get seconds
    double m_electricpotentialscale; // scaling for phi as above...
    double m_chargedensityscale; // scaling for rho
    double m_numberdensityscale; // scaling for number density...
    double m_magneticpotentialscale; // for A
    double m_currentdensityscale;// for J
};

inline double UnitConverter::si_numberdensity_to_sim(double n) const {
  return n / m_numberdensityscale;
}
inline double UnitConverter::si_velocity_to_sim(double v) const {
  return v / (m_lengthscale / m_timescale);
}
inline double UnitConverter::si_temperature_ev_to_sim(double tev) const {
  return tev * ELEMENTARY_CHARGE / (ELEMENTARY_MASS * std::pow(SPEED_OF_LIGHT, 2));
}
inline double UnitConverter::si_magneticfield_to_sim(double B) const {
  return B * m_lengthscale / m_magneticpotentialscale;
}

inline double UnitConverter::velocity_to_si(double v) const {
  return v * m_lengthscale / m_timescale;
}
inline double UnitConverter::density_to_si(double n) const {
  return n * m_numberdensityscale;
}
inline double UnitConverter::time_to_si(double t) const { return t * m_timescale; }

inline double UnitConverter::timescale() const { return m_timescale; }

inline double UnitConverter::lengthscale() const { return m_lengthscale; }

#endif
