#ifndef _NESO_NEKTAR_INTERFACE_PROJECTION_STATIC_CASE_HPP
#define _NESO_NEKTAR_INTERFACE_PROJECTION_STATIC_CASE_HPP
#include <utility>
// Copied from:
// https://stackoverflow.com/questions/68831855/c-automatically-generate-switch-statement-cases-at-compile-time
// Should hopefull expand to the equivilent of a switch rather
// that a chain of ifs. Only diff is offsetting everything by one so you can't
// call it with 0
namespace NESO::Utilities {
template <std::size_t min, class T, class F, T... I>
bool static_case(T value, F &&fn, std::integer_sequence<T, I...>) {
  return ((value == (I + min) &&
           (fn(std::integral_constant<T, I + min>{}), true)) ||
          ...);
}

template <std::size_t min, std::size_t max, class T, class F>
bool static_case(T value, F &&fn) {
  return static_case<min>(value, fn,
                          std::make_integer_sequence<T, max - min>{});
}

} // namespace NESO::Utilities
#endif
