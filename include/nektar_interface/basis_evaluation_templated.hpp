#ifndef __BASIS_EVALUATION_TEMPLATED_H_
#define __BASIS_EVALUATION_TEMPLATED_H_

#include <cstdlib>
#include <neso_particles.hpp>
using namespace NESO::Particles;

namespace NESO {

namespace BasisJacobi {

namespace Templated {

template <size_t n> constexpr size_t pochhammer(const size_t m) {
  return (m + (n - 1)) * pochhammer<n - 1>(m);
}
template <> constexpr size_t pochhammer<0>([[maybe_unused]] const size_t m) {
  return 1;
}

template <size_t p, size_t alpha, size_t beta> struct jacobis;
template <size_t pc, size_t alphac, size_t betac, typename T>
inline const REAL &getj(T *t) {
  return static_cast<jacobis<pc, alphac, betac> *>(t)->value;
}
// template <size_t pc, size_t alphac, size_t betac, typename T>
// inline const REAL getj(T t) {
//   return static_cast<jacobis<pc, alphac, betac>>(t).value;
// }
template <size_t pc, size_t alphac, size_t betac, typename T>
inline const REAL getj(const T &t) {
  return static_cast<const jacobis<pc, alphac, betac> &>(t).value;
}

template <size_t alpha, size_t beta> struct jacobis<0, alpha, beta> {
  const REAL value = 1.0;
  jacobis(const REAL z){};
};
template <size_t alpha, size_t beta>
struct jacobis<1, alpha, beta> : public jacobis<0, alpha, beta> {
  const REAL value;
  jacobis(const REAL z)
      : jacobis<0, alpha, beta>(z),
        value(0.5 * (2.0 * (alpha + 1.0) + (alpha + beta + 2.0) * (z - 1.0))){};
};
template <size_t alpha, size_t beta>
struct jacobis<2, alpha, beta> : public jacobis<1, alpha, beta> {
  const REAL value;
  jacobis(const REAL z)
      : jacobis<1, alpha, beta>(z),
        value(0.125 *
              ((4.0 * (alpha + 1.0) * (alpha + 2.0)) +
               (4.0 * (alpha + beta + 3.0) * (alpha + 2.0)) * (z - (1.0)) +
               ((alpha + beta + 3.0)) * ((alpha + beta + 4.0)) * (z - (1.0)) *
                   (z - (1.0)))){};
};

template <size_t p, size_t alpha, size_t beta>
struct jacobis : public jacobis<p - 1, alpha, beta> {
  constexpr static size_t n = p - 1;
  constexpr static REAL coeff_pnp1 =
      1.0 /
      (2.0 * (n + 1.0) * (n + alpha + beta + 1.0) * (2.0 * n + alpha + beta));
  constexpr static REAL coeff_pn =
      (2.0 * n + alpha + beta + 1.0) * (alpha * alpha - beta * beta);
  constexpr static REAL coeff_pnm1 =
      (-2.0 * (n + alpha) * (n + beta) * (2.0 * n + alpha + beta + 2.0));
  constexpr static REAL coeff_pochhammer = pochhammer<3>(2 * n + alpha + beta);

  const REAL value = 1.0;
  jacobis(const REAL z)
      : jacobis<p - 1, alpha, beta>(z),
        value(coeff_pnp1 * ((coeff_pn + coeff_pochhammer * z) *
                                getj<p - 1, alpha, beta>(this) +
                            coeff_pnm1 * getj<p - 2, alpha, beta>(this))){};
};

template <size_t p, size_t alpha, size_t beta> constexpr auto jacobi() {
  if constexpr (p == 0) {
    return [](const auto z) { return 1.0; };
  } else if constexpr (p == 1) {
    return [](const auto z) {
      return 0.5 * (2.0 * (alpha + 1.0) + (alpha + beta + 2.0) * (z - 1.0));
    };
  } else if constexpr (p == 2) {
    return [](const auto z) {
      return 0.125 *
             ((4.0 * (alpha + 1.0) * (alpha + 2.0)) +
              (4.0 * (alpha + beta + 3.0) * (alpha + 2.0)) * (z - (1.0)) +
              ((alpha + beta + 3.0)) * ((alpha + beta + 4.0)) * (z - (1.0)) *
                  (z - (1.0)));
    };
  } else {
    return [](const auto z) {
      const auto n = p - 1;
      auto pn = jacobi<n, alpha, beta>();
      auto pnm1 = jacobi<n - 1, alpha, beta>();
      const auto coeff_pnp1 = (2.0 * (n + 1.0) * (n + alpha + beta + 1.0) *
                               (2.0 * n + alpha + beta));
      const auto coeff_pn =
          (2.0 * n + alpha + beta + 1.0) * (alpha * alpha - beta * beta) +
          pochhammer<3>(2 * n + alpha + beta) * z;
      const auto coeff_pnm1 =
          (-2.0 * (n + alpha) * (n + beta) * (2.0 * n + alpha + beta + 2.0));
      const REAL v =
          (1.0 / coeff_pnp1) * (coeff_pn * pn(z) + coeff_pnm1 * pnm1(z));
      return v;
    };
  }
}

template <size_t nummodes, size_t px, size_t JPX>
constexpr auto eModifiedA(const jacobis<JPX, 1, 1> &j0) {
  if constexpr (px == 0) {
    return [](const auto z) { return 0.5 * (1.0 - z); };
  } else if constexpr (px == 1) {
    return [](const auto z) { return 0.5 * (1.0 + z); };
  } else {
    return [=](const auto z) {
      return 0.5 * (1.0 - z) * 0.5 * (1.0 + z) * getj<px - 2, 1, 1>(j0);
    };
  }
}

template <size_t PX> struct eModifiedAs;
template <> struct eModifiedAs<0> {
  const REAL value;
  template <size_t JPX>
  eModifiedAs(const REAL z, const jacobis<JPX, 1, 1> &j)
      : value(0.5 * (1.0 - z)) {}
};
template <> struct eModifiedAs<1> : public eModifiedAs<0> {
  const REAL value;
  template <size_t JPX>
  eModifiedAs(const REAL z, const jacobis<JPX, 1, 1> &j)
      : eModifiedAs<0>(z, j), value(0.5 * (1.0 + z)) {}
};
template <size_t PX> struct eModifiedAs : public eModifiedAs<PX - 1> {
  const REAL value;
  template <size_t JPX>
  eModifiedAs(const REAL z, const jacobis<JPX, 1, 1> &j)
      : eModifiedAs<PX - 1>(z, j),
        value(0.5 * (1.0 - z) * 0.5 * (1.0 + z) * getj<PX - 2, 1, 1>(j)) {}
};
template <size_t px, size_t PX>
inline const REAL &get_eModA(const eModifiedAs<PX> &t) {
  return static_cast<const eModifiedAs<px> &>(t).value;
}
template <size_t PX> inline eModifiedAs<PX> make_eModifiedA(const REAL z) {
  jacobis<PX - 2, 1, 1> j(z);
  return eModifiedAs<PX>(z, j);
}

namespace Quadrilateral {

template <size_t PX, size_t px, size_t qx, size_t mode, size_t JPX, size_t JQX>
inline REAL evaluate_inner(const jacobis<JPX, 1, 1> &j0,
                           const jacobis<JQX, 1, 1> &j1, const REAL *dofs,
                           const REAL eta0, const REAL eta1) {
  return dofs[mode] * eModifiedA<PX, px>(j0)(eta0) *
         eModifiedA<PX, qx>(j1)(eta1);
}

template <size_t PX, size_t px, size_t qx, size_t mode, size_t JPX, size_t JQX>
inline REAL inner(const jacobis<JPX, 1, 1> &j0, const jacobis<JQX, 1, 1> &j1,
                  const REAL *dofs, const REAL &eta0, const REAL &eta1) {
  REAL v = evaluate_inner<PX, px, qx, mode>(j0, j1, dofs, eta0, eta1);
  if constexpr (px < (PX - 1)) {
    v += inner<PX, px + 1, qx, mode + 1>(j0, j1, dofs, eta0, eta1);
  } else if constexpr (px == (PX - 1) && qx < (PX - 1)) {
    v += inner<PX, 0, qx + 1, mode + 1>(j0, j1, dofs, eta0, eta1);
  }
  return v;
}

template <size_t PX>
inline REAL evaluate(const REAL *dofs, const REAL eta0, const REAL eta1) {
  jacobis<PX - 2, 1, 1> j0(eta0);
  jacobis<PX - 2, 1, 1> j1(eta1);
  return inner<PX, 0, 0, 0>(j0, j1, dofs, eta0, eta1);
}

} // namespace Quadrilateral

namespace Hexahedron {

template <size_t PX, size_t px, size_t qx, size_t rx, size_t mode, size_t JPX,
          size_t JQX, size_t JRX>
inline REAL evaluate_inner(const jacobis<JPX, 1, 1> &j0,
                           const jacobis<JQX, 1, 1> &j1,
                           const jacobis<JRX, 1, 1> &j2, const REAL *dofs,
                           const REAL eta0, const REAL eta1, const REAL eta2) {

  return dofs[mode] * eModifiedA<PX, px>(j0)(eta0) *
         eModifiedA<PX, qx>(j1)(eta1) * eModifiedA<PX, rx>(j2)(eta2);
}

template <size_t PX, size_t px, size_t qx, size_t rx, size_t mode, size_t JPX,
          size_t JQX, size_t JRX>
inline REAL inner(const jacobis<JPX, 1, 1> &j0, const jacobis<JQX, 1, 1> &j1,
                  const jacobis<JRX, 1, 1> &j2, const REAL *dofs,
                  const REAL &eta0, const REAL &eta1, const REAL &eta2) {
  REAL v =
      evaluate_inner<PX, px, qx, rx, mode>(j0, j1, j2, dofs, eta0, eta1, eta2);
  if constexpr (px < (PX - 1)) {
    v +=
        inner<PX, px + 1, qx, rx, mode + 1>(j0, j1, j2, dofs, eta0, eta1, eta2);
  } else if constexpr (px == (PX - 1) && qx < (PX - 1)) {
    v += inner<PX, 0, qx + 1, rx, mode + 1>(j0, j1, j2, dofs, eta0, eta1, eta2);
  } else if constexpr (px == (PX - 1) && qx == (PX - 1) && (rx < (PX - 1))) {
    v += inner<PX, 0, 0, rx + 1, mode + 1>(j0, j1, j2, dofs, eta0, eta1, eta2);
  }
  return v;
}

template <size_t PX>
inline REAL evaluate(const REAL *dofs, const REAL eta0, const REAL eta1,
                     const REAL eta2) {
  jacobis<PX - 2, 1, 1> j0(eta0);
  jacobis<PX - 2, 1, 1> j1(eta1);
  jacobis<PX - 2, 1, 1> j2(eta2);
  return inner<PX, 0, 0, 0>(j0, j1, dofs, eta0, eta1);
}

template <size_t PX, size_t px, size_t qx, size_t rx, size_t mode>
inline REAL evaluate_inners(const eModifiedAs<PX> &e0,
                            const eModifiedAs<PX> &e1,
                            const eModifiedAs<PX> &e2, const REAL *dofs) {
  return dofs[mode] * get_eModA<px>(e0) * get_eModA<qx>(e1) * get_eModA<rx>(e2);
}

template <size_t PX, size_t px, size_t qx, size_t rx, size_t mode>
inline REAL inners(const eModifiedAs<PX> &e0, const eModifiedAs<PX> &e1,
                   const eModifiedAs<PX> &e2, const REAL *dofs) {
  REAL v = evaluate_inners<PX, px, qx, rx, mode>(e0, e1, e2, dofs);
  if constexpr (px < (PX - 1)) {
    v += inners<PX, px + 1, qx, rx, mode + 1>(e0, e1, e2, dofs);
  } else if constexpr (px == (PX - 1) && qx < (PX - 1)) {
    v += inners<PX, 0, qx + 1, rx, mode + 1>(e0, e1, e2, dofs);
  } else if constexpr (px == (PX - 1) && qx == (PX - 1) && (rx < (PX - 1))) {
    v += inners<PX, 0, 0, rx + 1, mode + 1>(e0, e1, e2, dofs);
  }
  return v;
}

} // namespace Hexahedron

template <typename SPECIALISATION> struct ExpansionLoopingInterface {
  template <size_t NUM_MODES>
  inline REAL evaluate(const REAL *const dofs, const REAL eta0, const REAL eta1,
                       const REAL eta2) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    return underlying.template evaluate_v<NUM_MODES>(dofs, eta0, eta1, eta2);
  }
};

struct TemplatedQuadrilateral
    : public ExpansionLoopingInterface<TemplatedQuadrilateral> {
  template <size_t NUM_MODES>
  inline REAL evaluate_v(const REAL *const dofs, const REAL eta0,
                         const REAL eta1, [[maybe_unused]] const REAL eta2) {
    jacobis<NUM_MODES - 2, 1, 1> j0(eta0);
    jacobis<NUM_MODES - 2, 1, 1> j1(eta1);
    return Quadrilateral::inner<NUM_MODES, 0, 0, 0>(j0, j1, dofs, eta0, eta1);
  }
};

struct TemplatedHexahedronOrig
    : public ExpansionLoopingInterface<TemplatedHexahedronOrig> {
  template <size_t NUM_MODES>
  inline REAL evaluate_v(const REAL *const dofs, const REAL eta0,
                         const REAL eta1, [[maybe_unused]] const REAL eta2) {
    jacobis<NUM_MODES - 2, 1, 1> j0(eta0);
    jacobis<NUM_MODES - 2, 1, 1> j1(eta1);
    jacobis<NUM_MODES - 2, 1, 1> j2(eta2);
    return Hexahedron::inner<NUM_MODES, 0, 0, 0, 0>(j0, j1, j2, dofs, eta0,
                                                    eta1, eta2);
  }
};

struct TemplatedHexahedron
    : public ExpansionLoopingInterface<TemplatedHexahedron> {
  template <size_t NUM_MODES>
  inline REAL evaluate_v(const REAL *const dofs, const REAL eta0,
                         const REAL eta1, [[maybe_unused]] const REAL eta2) {
    auto e0 = make_eModifiedA<NUM_MODES>(eta0);
    auto e1 = make_eModifiedA<NUM_MODES>(eta1);
    auto e2 = make_eModifiedA<NUM_MODES>(eta2);
    return Hexahedron::inners<NUM_MODES, 0, 0, 0, 0>(e0, e1, e2, dofs);
  }
};

} // namespace Templated

} // namespace BasisJacobi

} // namespace NESO

#endif
