#pragma once
#include "constants.hpp"
#include "..basis/basis.hpp"
#include "unroll.hpp"
#include <cstdint>

namespace NESO::Project
{

template <int nmode, typename T, int alpha, int beta>
void
tri_ppt(const double eta0,
            const double eta1,
            const double qoi,
            double *dofs)
{
    T local0[nmode];
    T local1[(nmode * (nmode + 1)) << 1];
    Basis::eModA<T, nmode, Constants::cpu_stride, alpha, beta>(eta0, local0);
    Basis::eModB<T, nmode, Constants::cpu_stride, alpha, beta>(eta1, local1);
    int mode = 0;
    NESO_UNROLL_LOOP
    for (int i = 0; i < nmode; ++i) {
        NESO_UNROLL_LOOP
        for (int j = 0; j < nmode - i; ++j) {
            double temp = (mode == 1) ? 1.0 : local0[i];
            dofs[mode] += temp * local1[mode] * qoi;
            mode++;
        }
    }
}
} // namespace Tri
