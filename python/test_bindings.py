import matplotlib.pyplot as plt
from PyNESO import Mesh, Species, Plasma, Diagnostics, FFT, evolve, Queue

##############
# Parameters #
##############
# Mesh
nintervals = 128  # int
dt = 0.05  # double
nt = 40  # int

# Ions
kinetic_i = False  # Whether the species are treated as kinetic
T_i = 2.0  # Temperature
q_i = -1.0  # Charge
m_i = 1836.2  # Mass
n_i = 1  # Number of particles

# Electrons
kinetic_e = True  # Whether the species are treated as kinetic
T_e = 2.0  # Temperature
q_e = 1.0  # Charge
m_e = 1.0  # Mass
n_e = 12800  # Number of particles

Q = Queue()

mesh = Mesh(nintervals, dt, nt)
ions = Species(mesh, kinetic_i, T_i, q_i, m_i, n_i)
electrons = Species(mesh, kinetic_e, T_e, q_e, m_e, n_e)

plasma = Plasma([ions, electrons])

diagnostics = Diagnostics()
fft = FFT(Q, mesh.nintervals)

mesh.set_initial_field(Q, mesh, plasma, fft)
evolve(Q, mesh, plasma, fft, diagnostics)

print(diagnostics.total_energy)

plt.semilogy(diagnostics.total_energy)
plt.savefig("total_energy.pdf")
