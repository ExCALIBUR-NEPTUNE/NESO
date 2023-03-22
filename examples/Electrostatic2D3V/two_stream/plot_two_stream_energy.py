import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes
import matplotlib.ticker as mtick
import h5py
import xml.etree.ElementTree as ET
import math

plt.rc("xtick", labelsize="small")
plt.rc("ytick", labelsize="small")
plt.rc("text", usetex=True)
plt.rc("savefig", dpi=500)
plt.rc("xtick", labelsize="small")
plt.rc("legend", fontsize="medium")
plt.rc("legend", edgecolor="lightgray")
plt.rc("ytick", labelsize="small")
plt.rcParams.update({"font.size": 20})

plt.rc("font", family="serif", serif="Times New Roman")
params = {"text.latex.preamble": r"\usepackage{amsmath}"}

plt.rcParams.update(params)


class Session:
    def __init__(self, filename):
        """
        Helper class to read XML Nektar++ session file.

        :param: filename Nektar++ session file to read.
        """
        self.xml_tree = ET.parse(filename)
        self.xml_root = self.xml_tree.getroot()
        self.parameters = self.xml_root.find("CONDITIONS").find("PARAMETERS").findall("P")

    def get_parameter(self, name, t=float):
        """
        Get value from session file and cast to type.

        :param: name Name of value to extract.
        :param: t Type to cast to.
        :returns: Value cast to type.
        """
        dt = -1
        for px in self.parameters:
            text = [tx for tx in px.itertext()][0].strip()
            if text.startswith(name):
                dt = float(text.split("=")[-1])
        return t(dt)


def get_gamma_and_validate(session, two_stream):
    """
    Compute the expected gradient of the energy growth (in logspace). Computes
    the expected parameter values for the simualtion and checks these
    parameters against the ones computed by the simulation.
    """

    Lx = two_stream["global_data"]["L_x"][0]
    Ly = two_stream["global_data"]["L_y"][0]
    print("Lx", Lx)
    print("Ly", Ly)
    L = Lx
    M = 1

    volume = Lx * Ly
    n = session.get_parameter("particle_number_density") / 2.0
    print("n", n, (16.0 * math.pi * math.pi / (3)))
    num_particles_physical = n * volume

    num_particles_total = session.get_parameter("num_particles_total")

    particle_charge_density = session.get_parameter("particle_charge_density") / 2.0
    q = particle_charge_density * volume / num_particles_physical
    print("q", q)

    q_sim = two_stream["global_data"]["q"][0]
    assert abs(q_sim - q) < 1.0e-14
    m = 1.0
    print("m", m)
    m_sim = two_stream["global_data"]["m"][0]
    assert abs(m - m_sim) < 1.0e-14

    epsilon_0 = 1.0
    print("epsilon_0", epsilon_0)
    v_b = session.get_parameter("particle_initial_velocity")
    print("v_b", v_b)
    PI_s = math.sqrt((q * q * n) / (m * epsilon_0))
    print("PI_s", PI_s)

    k_parallel = math.sqrt(3) * 0.5 * PI_s / v_b
    print(
        "k_parallel",
        k_parallel,
        "v_b k_par,max",
        v_b * k_parallel,
        "Pi_s sqrt(3)/2",
        PI_s * math.sqrt(3) / 2,
    )
    u = v_b * k_parallel / PI_s
    x_minus = (u * u + 1.0 - (4.0 * u * u + 1.0) ** 0.5) ** 0.5
    x_plus = (u * u + 1.0 + (4.0 * u * u + 1.0) ** 0.5) ** 0.5

    x_mm = -x_minus
    x_pm = x_minus
    x_pp = x_plus
    x_mp = -x_plus

    print("x--", x_mm)
    print("x+-", x_pm)
    print("x++", x_pp)
    print("x-+", x_mp)
    print("sqrt(15)/2", 15 ** 0.5 / 2)

    omega_mm = x_mm * PI_s
    omega_pm = x_pm * PI_s
    omega_pp = x_pp * PI_s
    omega_mp = x_mp * PI_s

    print("omega_mm", omega_mm)
    print("omega_pm", omega_pm)
    print("omega_pp", omega_pp)
    print("omega_mp", omega_mp)

    print("PI_s                     :", PI_s)
    print("PI_s: v_b 4pi/sqrt(3)    :", v_b * 4.0 * math.pi / math.sqrt(3))
    print("n                        :", n)
    print(
        "v_b^2 (m/q^2) 16 pi^2/3  :",
        v_b * v_b * (m / (q * q)) * 16 * math.pi ** 2 / 3.0,
    )

    gamma_max = PI_s / 2
    PI_T = PI_s * math.sqrt(2)
    print("PI_s/2                   :", PI_s / 2)
    print("v_b M 2pi/sqrt(3)        :", v_b * 2.0 * math.pi / math.sqrt(3))
    print("PI_T/(2sqrt(2))          :", PI_T / (2.0 * math.sqrt(2)))
    print(
        "(1/2) * sqrt(32pi**2/6)  :",
        0.5 * math.sqrt(((32 * (math.pi ** 2))) / 6),
    )

    gamma_energy = 2 * gamma_max

    # gamma_max = 0.5 * math.sqrt(((32 * (math.pi ** 2))) / 6)
    print("gamma_max", gamma_max, 0.5 * math.sqrt(((32 * (math.pi ** 2))) / 6))

    w = v_b ** 2 * 16 * (math.pi ** 2 / 3) * (2 / num_particles_total)
    print("w", w, 32 * math.pi ** 2 / (3 * num_particles_total))
    nT = 32 * math.pi ** 2 / 3
    print("nT", nT)

    n_general = v_b ** 2 * M ** 2 * (m * epsilon_0 / (q ** 2)) * (16 * math.pi ** 2 / (3 * (L ** 2)))
    print("n_general", n_general)

    w_general = 2.0 * n_general * Lx * Ly
    print("w_general (density):", w_general)
    print("w_general:", w_general / num_particles_total)

    w_sim = two_stream["global_data"]["w"][0]
    print("w_sim", w_sim)
    if not abs(w_sim - w_general / num_particles_total) < 1.0e-14:
        print("CONSISTENCY ERROR with (w_sim, w_general) =", w_sim, w_general / num_particles_total)

    return gamma_energy


def plot_figures(session, two_stream, gamma_energy):
    """
    Plot the potential, kinetic and total energy from the two stream example.
    """

    step_data = two_stream["step_data"]
    step_keys = sorted(step_data.keys(), key=lambda x: int(x))

    N = len(step_keys)
    # The x arrays are all time in these plots.
    x = np.zeros((N,))
    # The y arrays are all energies in these plots.
    field_y = np.zeros((N,))
    potential_y = np.zeros((N,))
    kinetic_y = np.zeros((N,))

    # time step size
    dt = session.get_parameter("particle_time_step")
    assert dt > 0.0

    # for each energy type extract the values from the hdf5 file
    for keyi, keyx in enumerate(step_keys):
        t = dt * int(keyx)
        x[keyi] = t
        field_y[keyi] = step_data[keyx]["field_energy"][0]
        potential_y[keyi] = step_data[keyx]["potential_energy"][0]
        kinetic_y[keyi] = step_data[keyx]["kinetic_energy"][0]
    # these time values are all identical
    potential_x = x
    kinetic_x = x

    # compute the total energy at each timestep
    total_x = kinetic_x
    total_y = potential_y + kinetic_y

    # compute the energy difference between the start and end of the simualtion
    total_initial = total_y[0]
    total_end = total_y[-1]
    print("Energy diff:", abs(total_initial - total_end) / abs(total_initial))

    # assume that the linear (in log space) part we want the gradient of is
    # between the global max and global min
    potential_energy_max_index = np.argmax(potential_y)
    potential_energy_min_index = np.argmin(potential_y[:potential_energy_max_index])

    x_range = potential_energy_max_index - potential_energy_min_index
    potential_energy_max_index -= int(0.2 * x_range)
    potential_energy_min_index += int(0.2 * x_range)

    print("Fit start index:", potential_energy_min_index)
    print("Fit end index  :", potential_energy_max_index)

    dx = potential_x[potential_energy_max_index] - potential_x[potential_energy_min_index]
    dy = np.log(potential_y[potential_energy_max_index]) - np.log(potential_y[potential_energy_min_index])

    print("Gradient                 :", dy / dx)
    print("gamma_energy (theory)    :", gamma_energy)

    # compute the start, end points of a straight line (in log space) that
    # matches the theory for the energy growth
    tx0 = potential_x[potential_energy_min_index]
    tx1 = potential_x[potential_energy_max_index]
    iy0 = math.exp(gamma_energy * tx0)
    iy1 = math.exp(gamma_energy * tx1)
    ishift = 1.0

    # plot field energy and potential energy
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    field_colour = "b"
    potential_colour = "m"
    kinetic_colour = "r"
    total_colour = "k"

    ax.semilogy(
        x,
        field_y,
        color=field_colour,
        linewidth=2,
        markersize=8,
        base=np.e,
    )
    ax2.semilogy(
        potential_x,
        potential_y,
        color=potential_colour,
        linewidth=2,
        markersize=8,
        base=np.e,
    )
    ax2.semilogy(
        [
            potential_x[potential_energy_min_index],
            potential_x[potential_energy_max_index],
        ],
        [
            potential_y[potential_energy_min_index],
            potential_y[potential_energy_max_index],
        ],
        color="r",
        label=f"Fitted gamma: ${dy/dx}$",
        linewidth=2,
        markersize=8,
        linestyle="--",
        base=np.e,
    )

    ax2.semilogy(
        [tx0, tx1],
        [ishift * iy0, ishift * iy1],
        color="k",
        label=f"Theory gamma: ${gamma_energy}$",
        linewidth=2,
        markersize=8,
        linestyle="--",
        base=np.e,
    )
    ax2.legend()

    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Field Energy: $\int_{\Omega} \phi^2 dx$")
    ax2.set_ylabel(r"Potential Energy: $\frac{1}{2}\sum_{i} \phi(\vec{r}_i)q_i$")

    def tick_formater(y, pos):
        return r"$e^{{{:.0f}}}$".format(np.log(y))

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(tick_formater))
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(tick_formater))

    ax.yaxis.label.set_color(field_colour)
    ax2.yaxis.label.set_color(potential_colour)

    fig.savefig("field_energy.pdf")

    # plot potential, kinetic and total energy
    fig = plt.figure(figsize=(10, 7))
    te_ax = fig.add_subplot(111)

    pe_ax = te_ax.twinx()
    ke_ax = te_ax.twinx()
    ke_ax.spines.right.set_position(("axes", 1.15))

    pe_ax.plot(
        potential_x,
        (potential_y - potential_y[0]) / abs(total_initial),
        color=potential_colour,
        label="Potential Energy",
        linewidth=2,
        markersize=8,
    )
    ke_ax.plot(
        kinetic_x,
        (kinetic_y - kinetic_y[0]) / abs(total_initial),
        color=kinetic_colour,
        label="Kinetic Energy",
        linewidth=2,
        markersize=8,
    )
    te_ax.plot(
        total_x,
        np.abs(total_y - total_initial) / abs(total_initial),
        color=total_colour,
        label="Total Energy",
        linewidth=2,
        markersize=8,
    )

    # ax.set_yscale('log')
    te_ax.set_xlabel(r"Time")
    pe_ax.set_ylabel(
        r"$(E_{\mathcal{U}} - E(0)) / E(0)$, $E_{\mathcal{U}} = \frac{1}{2}\sum_{i} \phi(\vec{r}_i)q_i$",
        color=potential_colour,
    )
    ke_ax.set_ylabel(
        r"$(E_{\mathcal{K}} - E(0)) / E(0)$, $E_{\mathcal{K}} = \sum_{i} \frac{1}{2}m_i|\vec{v}_i|^2 $",
        color=kinetic_colour,
    )
    te_ax.set_ylabel(
        r"Total Energy ($E$) Rel. Error: $|E - E(0)|/|E(0)|$",
        color=total_colour,
    )

    fig.savefig("all_energy.pdf", bbox_inches="tight")


if __name__ == "__main__":

    if (len(sys.argv) < 3) or ("--help" in sys.argv) or ("-h" in sys.argv):
        print(
            """
Plots energies from electrostatic PIC. Call with:

python3 plot_two_stream_energy.py input.xml electrostatic_two_stream.h5

"""
        )
        quit()

    session = Session(sys.argv[1])
    two_stream = h5py.File(sys.argv[2], "r")

    gamma_energy = get_gamma_and_validate(session, two_stream)
    plot_figures(session, two_stream, gamma_energy)
