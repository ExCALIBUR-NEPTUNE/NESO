#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "revision.hpp"
#include "mesh.hpp"
#include "velocity.hpp"
#include "species.hpp"
#include "plasma.hpp"
#include "fft_wrappers.hpp"
#include "diagnostics.hpp"
#include "simulation.hpp"
#include "particle_utility/position_distribution.hpp"

namespace py = pybind11;

PYBIND11_MODULE(PyNESO, m) {
  m.attr("revision") = NESO::version::revision;
  m.attr("git_state") = NESO::version::git_state;

  // TODO: Provide access to other overloads
  py::class_<sycl::queue>(m, "Queue")
    .def(py::init<>());

  py::class_<Mesh>(m, "Mesh")
    .def(py::init<int, double, int>(), py::arg("nintervals")=10, py::arg("dt")=0.1, py::arg("nt")=1000)
    .def("evaluate_electric_field", &Mesh::evaluate_electric_field)
    .def("deposit", &Mesh::deposit)
    .def("get_electric_field", &Mesh::get_electric_field)
    .def("get_E_staggered_from_E", &Mesh::get_E_staggered_from_E)
    .def("set_initial_field", &Mesh::set_initial_field)
    .def("get_left_index", &Mesh::get_left_index)
    .def_readonly("t", &Mesh::t)
    .def_readonly("dt", &Mesh::dt)
    .def_readonly("nt", &Mesh::nt)
    .def_readonly("nintervals", &Mesh::nintervals)
    // TODO: Make it possible to convert between numpy arrays and std::vector
    .def_readonly("mesh", &Mesh::mesh);

  // TODO: Make it possible to convert between numpy arrays and std::vector
  py::class_<Velocity>(m, "Velocity")
    .def(py::init<std::vector<double>, std::vector<double>, std::vector<double>>())
    .def_readwrite("x", &Velocity::x)
    .def_readwrite("y", &Velocity::y)
    .def_readwrite("z", &Velocity::z);

  py::class_<Species>(m, "Species")
    .def(py::init<const Mesh &, bool, double, double, double, int>(),
         py::arg("mesh"), py::arg("kinetic")=true, py::arg("T")=1.0, py::arg("q")=1.0,
         py::arg("m")=1.0, py::arg("n")=10)
    .def("push", &Species::push)
    .def("set_array_dimensions", &Species::set_array_dimensions)
    // TODO: Make it possible to convert between numpy arrays and std::vector
    .def("set_initial_conditions", &Species::set_initial_conditions);

  py::class_<Plasma>(m, "Plasma")
    .def(py::init<std::vector<Species>>())
    .def("push", &Plasma::push);

  py::class_<FFT>(m, "FFT")
    .def(py::init<sycl::queue &, int>());

  // TODO: Make it possible to convert between numpy arrays and std::vector
  py::class_<Diagnostics>(m, "Diagnostics")
    .def(py::init())
    .def_readonly("time", &Diagnostics::time)
    .def_readonly("total_energy", &Diagnostics::total_energy)
    .def_readonly("particle_energy", &Diagnostics::particle_energy)
    .def_readonly("field_energy", &Diagnostics::field_energy)
    .def("store_time", &Diagnostics::store_time)
    .def("compute_total_energy", &Diagnostics::compute_total_energy)
    .def("compute_field_energy", &Diagnostics::compute_field_energy)
    .def("compute_particle_energy", &Diagnostics::compute_particle_energy);

  m.def("evolve", &evolve);

  m.def("sobol_within_extents", &NESO::sobol_within_extents);  // TODO: Provide proper wrappers for extents array; convert return-type to numpy array
  m.def("rsequence", py::overload_cast<const int, const double, const double>(&NESO::rsequence), py::arg("i"), py::arg("irrational"), py::arg("seed")=0.0);
  m.def("rsequence", py::overload_cast<const int, const int, const double>(&NESO::rsequence), py::arg("N"), py::arg("dim"), py::arg("seed")=0.0);
  m.def("rsequence_within_extents", &NESO::rsequence_within_extents, py::arg("N"), py::arg("ndim"), py::arg("extents"), py::arg("seed")=0); // TODO: Provide proper wrappers for extents array; convert return-type to numpy array

  // TODO: Write bindings for Nektar Interface?
}
