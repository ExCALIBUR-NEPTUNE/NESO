#pragma once
#include "constants.hpp"
#include "..basis/basis.hpp"
#include "unroll.hpp"


namespace NESO::Project
{

template <int nmode, typename T, int alpha, int beta>
void
tet_ppt(double const eta0,
            double const eta1,
            double const eta2,
            double const qoi,
            double *dofs)
{
    T local0[nmode];
    T local1[nmode*(nmode + 1)/2]; //triangle number
    T local2[nmode*(nmode+1)*(nmode+2)/6]; //tet number 
    Basis::eModA<T, nmode, Constants::cpu_stride, alpha, beta>(eta0, local0);
    Basis::eModB<T, nmode, Constants::cpu_stride, alpha, beta>(eta1, local1);
    Basis::eModC<T, nmode, Constants::cpu_stride, alpha, beta>(eta2, local2);
    int modez = 0;
    int modey = 0;
    NESO_UNROLL_LOOP
    for (int i = 0; i < nmode; ++i) {
        NESO_UNROLL_LOOP
        for (int j = 0; j < nmode - i; ++j) {
            NESO_UNROLL_LOOP
            for (int k = 0; k < nmode - j - i; ++k) {
                //TODO: could pull the special cases out of the loop
                //i.e. do the first one,
                //and split the loop so i == 0, j==0 and then 1 is done
                //before
                if  (modez == 1)
                    dofs[modez] += qoi * local2[modez];
                else if  (i == 0 || j == 1)
                    dofs[modez] += qoi * local2[modez] * local1[modey];
                else
                    dofs[modez] += qoi * local2[modez] * local1[modey] * local0[i];
                modez++;
            }
            modey++;
        }
    }
}
} // namespace Quad
