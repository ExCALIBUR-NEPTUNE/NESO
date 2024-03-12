#pragma once
#include "constants.hpp"
#include "..basis/basis.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include <cmath>
#include <cstdio>
namespace NESO::Project
{
namespace Private
{
//(approx) solving for n in
// tested up to 1000 dofs and
// seems robust - but there could be some
// floating point issues
// p * (p + 1)/2 = N
// to go from dof to corresponding
// emodA entry
template <int nmode>
inline int
calc_tri_row(int N)
{
    double n = double(N + 1);
    double tmp = -.5 + std::sqrt(8 * n + 1) / 2.0;
    return int(std::ceil(tmp)) - 1;
}
// solving for
//(NMODE + 1 + (NMODE +1 - dof))*(dof + 1)/2 = X;
// if NMODE==4
// then
// 0,1,2,3,4 -> 0
// 5,6,7,8   -> 1
// 9,10,11   -> 2
// 12,13     -> 3
// 14        -> 4
// i.e. mapping from dof -> index in emodA array
// TODO: might not be the most efficient way
// a lookup-table might be better
// especially in 3D I guess would be cubic in that case?
template <int NMODE>
int
calc_tri_row_rev(int dof)
{
    double a = double(1 - 2 * (NMODE + 1));
    double n = double(1 + 2 * (dof));
    double tmp = -0.5 * (a + std::sqrt(a * a - 4 * n));
    return int(std::floor(tmp));
}
}
// Serialisation of real thing, also only doing the innner
// loop
// TODO: should have a chunk to simulate the outer parallelism
template <int nmode, typename T, int alpha, int beta>
void
tri_sync(double const *eta0,
             double const *eta1,
             double const *qoi,
             double *dofs,
             int npar)
{
    T localA[Constants::gpu_stride * nmode] = {0.0};
    T localB[Constants::gpu_stride * nmode * (nmode + 1) << 1] = {0.0};
    assert(npar < Constants::gpu_stride);
    for (int par = 0; par < npar; ++par) {
        Basis::eModA<T, nmode, Constants::gpu_stride, alpha, beta>(eta0[par],
                                                            localA + par);
        Basis::eModB<T, nmode, Constants::gpu_stride, alpha, beta>(eta1[par],
                                                            localB + par);
        // TODO - think about this
        // Ideally would do this to the emodA array as it's
        // smaller, but the correction see below ***
        // makes it impossible (AFAICS)
        // Could just store the qoi but that is more mem
        // need to experiment
        for (int qx = 0; qx < nmode * (nmode - 1) * 2; ++qx) {
            localB[par + qx * Constants::gpu_stride] *= qoi[par];
        }
    }
    for (int thd = 0; thd < Constants::local_size; ++thd) {
        auto idx_local = thd;
        auto ndof = nmode * (nmode + 1) / 2;
        // Here we don't consider that there is another "loop" over every innner
        // range until every particle is done
        auto count = std::min(Constants::local_size, std::max(int{0}, npar));
        while (idx_local < ndof) {
            int i = Private::calc_tri_row_rev<nmode>(idx_local);
            double temp = 0.0;
            for (int k = 0; k < count; ++k) {
                // TODO: this correction might be bad or fine (for perf)
                // need to check this
                //(***)
                double correction = (idx_local == 1)
                                        ? 1.0
                                        : localA[i * Constants::gpu_stride + k];
                temp +=
                    correction * localB[idx_local * Constants::gpu_stride + k];
            }
            dofs[idx_local] += temp;
            idx_local += Constants::local_size;
        }
    }
}
} // namespace Tri
