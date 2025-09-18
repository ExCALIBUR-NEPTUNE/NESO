#ifndef _NESO_COMPOSITE_INTERACTION_COMPOSITE_FUNCTION_UTILITY_HPP_
#define _NESO_COMPOSITE_INTERACTION_COMPOSITE_FUNCTION_UTILITY_HPP_

#include "composite_function.hpp"

namespace NESO::CompositeInteraction {

/**
 * Integrate the provided function over the domain. Must be called collectively
 * on the communicator.
 *
 * @param func Function to integrate.
 */
REAL integrate(CompositeFunctionSharedPtr func);

} // namespace NESO::CompositeInteraction

#endif
