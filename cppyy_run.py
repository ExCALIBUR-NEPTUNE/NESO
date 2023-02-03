import cppyy

# import numpy as np
# from tqdm import tqdm

cppyy.load_library("build/libnesolib.so")
cppyy.load_library("build/build/lib/PyNESO/libPyNESOCppyy.so")
from cppyy.gbl import Mesh, Species, Plasma, Diagnostics, FFT, evolve
from cppyy.gbl import sycl
from cppyy.gbl.std import vector

print(dir(cppyy.gbl))

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

# Q = sycl::queue{sycl::default_selector{}, asyncHandler};
Q = sycl.queue()  # {sycl.default_selector{}, asyncHandler};

mesh = Mesh(nintervals, dt, nt)
ions = Species(mesh, kinetic_i, T_i, q_i, m_i, n_i)
electrons = Species(mesh, kinetic_e, T_e, q_e, m_e, n_e)

species_list = vector[Species]()
species_list.push_back(ions)
species_list.push_back(electrons)
plasma = Plasma(species_list)

diagnostics = Diagnostics()
fft = FFT(Q, mesh.nintervals)

mesh.set_initial_field(Q, mesh, plasma, fft)
evolve(Q, mesh, plasma, fft, diagnostics)

###for it in tqdm(np.arange(1,int(mesh.nt+1))):
###
###    mesh.it = int(it)
###    #print("it %d\n", mesh.it);
###
###    plasma.assemble_rhs(opt, AI, mesh, fft);
###    timestepper.time_advance(opt, AI, mesh, plasma, fft);
###
###    mesh.t += mesh.dt;
###    diagnostics.store_time(mesh);
###
###    plasma.solve_for_electrostatic_potential(AI, mesh);
###    plasma.get_electric_field_from_electrostatic_potential(AI, mesh);
###
###    diagnostics.compute_total_energy(AI, mesh, plasma);
###    fileio.write_time_slice(mesh, plasma, diagnostics);
###
####print("solving_vlasov_poisson: "+str(opt.solving_vlasov_poisson()))
####print("solving_drift_kinetics: "+str(opt.solving_drift_kinetics()))
####print("solving_gyro_or_drift_kinetics: "+str(opt.solving_gyro_or_drift_kinetics()))
####print("exlicit_euler: "+str(opt.explicit_euler()))
####print("timestepper::expected_order: "+str(timestepper.expected_order))
###
####print(np.log(diagnostics.total_energy))
####print(diagnostics.total_energy)
####
####import matplotlib.pyplot as plt
####plt.semilogy(diagnostics.total_energy)
####plt.savefig("out.pdf")
####plt.clf()
###
