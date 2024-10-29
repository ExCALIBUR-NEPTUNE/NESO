#ifndef _NESO_NEKTAR_INTERFACE_PROJECTION_RESTRICT_HPP
#define _NESO_NEKTAR_INTERFACE_PROJECTION_RESTRICT_HPP

#ifndef NESO_RESTRICT
// not sure cuda support __restrict
#if defined(__CUDACC__)
#define NESO_RESTRICT __restrict__
#else
// Everybody else seems fine with this
#define NESO_RESTRICT __restrict
#endif
#endif
#endif
