## LAPD e.g.

### Initial conditions

H3 initialises $n_e$ with
$$
             0.1~e^{-x^2} + 10^{-5}~(\mathrm{mixmode}(z) + \mathrm{mixmode}(4*z - x))
$$

From BOUT++ docs:

   The ``mixmode(x)`` function is a mixture of Fourier modes of the form:

   $$
      \mathrm{mixmode}(x) = \sum_{i=1}^{14} \frac{1}{(1 +
      |i-4|)^2}\cos[ix + \phi(i, \mathrm{seed})]
   $$

   where $\phi$ is a random phase between $-\pi$ and
   $+\pi$, which depends on the seed. The factor in front of each
   term is chosen so that the 4th harmonic ($i=4$) has the highest
   amplitude. This is useful mainly for initialising turbulence
   simulations, where a mixture of mode numbers is desired.

OP: Not clear what ICs for fields other than $n_e$ are...