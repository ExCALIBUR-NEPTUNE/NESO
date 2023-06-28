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

---

### Electron-ion collision frequency
From eqns 133,134 in the equations doc:
$$
\nu_{e,i} = \frac{|q_e||q_i|n_{i}{\rm log}\Lambda(1+m_e/m_i)}{3\pi^{3/2}\epsilon_0^{2}m_e^2(v_e^2+v_i^2)^{3/2}}
$$
with 
$$
\begin{align}
{\rm log}\Lambda =&~30 − 0.5 \ln n_e − \ln Z_i + 1.5 \ln Te \\
{\rm log}\Lambda =&~34.14 − 0.5 \ln n_e
\end{align}
$$