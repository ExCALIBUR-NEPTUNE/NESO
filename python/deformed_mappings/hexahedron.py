"""
Defines the X maps for Hexahedrons.
"""

from newton_generation import *


class LinearHex(LinearBase):
    def __init__(self):

        num_vertices = 8
        ndim = 3
        name = "linear_3d"
        namespace = "Hexahedron"
        x_description = """
X(xi) = (1/8) * v0 * (1 - xi_0) * (1 - xi_1) * (1 - xi_2) + 
        (1/8) * v1 * (1 + xi_0) * (1 - xi_1) * (1 - xi_2) + 
        (1/8) * v2 * (1 + xi_0) * (1 + xi_1) * (1 - xi_2) + 
        (1/8) * v3 * (1 - xi_0) * (1 + xi_1) * (1 - xi_2) + 
        (1/8) * v4 * (1 - xi_0) * (1 - xi_1) * (1 + xi_2) + 
        (1/8) * v5 * (1 + xi_0) * (1 - xi_1) * (1 + xi_2) + 
        (1/8) * v6 * (1 + xi_0) * (1 + xi_1) * (1 + xi_2) + 
        (1/8) * v7 * (1 - xi_0) * (1 + xi_1) * (1 + xi_2)
"""
        LinearBase.__init__(self, num_vertices, ndim, name, namespace, x_description)

    def get_x(self, xi):

        v = self.vertices

        x = (
            0.125 * v[0] * (1 - xi[0]) * (1 - xi[1]) * (1 - xi[2])
            + 0.125 * v[1] * (1 + xi[0]) * (1 - xi[1]) * (1 - xi[2])
            + 0.125 * v[2] * (1 + xi[0]) * (1 + xi[1]) * (1 - xi[2])
            + 0.125 * v[3] * (1 - xi[0]) * (1 + xi[1]) * (1 - xi[2])
            + 0.125 * v[4] * (1 - xi[0]) * (1 - xi[1]) * (1 + xi[2])
            + 0.125 * v[5] * (1 + xi[0]) * (1 - xi[1]) * (1 + xi[2])
            + 0.125 * v[6] * (1 + xi[0]) * (1 + xi[1]) * (1 + xi[2])
            + 0.125 * v[7] * (1 - xi[0]) * (1 + xi[1]) * (1 + xi[2])
        )

        return x


def self_test():

    hex_x = LinearHex()

    vertices_ref = (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    )
    hex_ref = LinearGeomEvaluate(hex_x, vertices_ref)

    for vx in vertices_ref:
        to_test = hex_ref.x(vx)
        correct = vx
        assert (
            np.linalg.norm(
                np.array(correct).ravel() - np.array(to_test).ravel(), np.inf
            )
            < 1.0e-15
        )
        to_test = hex_ref.f(vx, vx)
        assert np.linalg.norm(np.array(to_test).ravel(), np.inf) < 1.0e-15

    vertices_test = (
        (-3.0, -2.0, 2.0),
        (1.0, -1.0, 2.0),
        (2.0, 2.0, 2.5),
        (-1.0, 4.0, 4.5),
        (-3.0, -2.0, -0.1),
        (1.0, -1.0, -1.0),
        (2.0, 2.0, -2.0),
        (-1.0, 4.0, -1.5),
    )
    hex_test = LinearGeomEvaluate(hex_x, vertices_test)
    hex_newton = Newton(hex_x)

    hex_newton_evaluate = NewtonEvaluate(hex_newton, hex_test)

    xi_correct0 = -0.9
    xi_correct1 = 0.8
    xi_correct2 = 0.2
    xi_correct = (xi_correct0, xi_correct1, xi_correct2)
    phys = hex_test.x(xi_correct)
    residual, fv = hex_newton_evaluate.residual(xi_correct, phys)
    assert residual < 1.0e-15

    xi = [0.0, 0.0, 0.0]
    for stepx in range(5):
        residual, fv = hex_newton_evaluate.residual(xi, phys)
        xin = hex_newton_evaluate.step(xi, phys, fv)
        xi[0] = xin[0]
        xi[1] = xin[1]
        xi[2] = xin[2]

    assert abs(xi[0] - xi_correct[0]) < 1.0e-14
    assert abs(xi[1] - xi_correct[1]) < 1.0e-14

    vertices_test = (
        (-1, -1, -1),
        (-0.8, -1, -1),
        (-0.8, -0.8, -1),
        (-1, -0.8, -1),
        (-1, -1, -0.8),
        (-0.8, -1, -0.8),
        (-0.8, -0.8, -0.8),
        (-1, -0.8, -0.8),
    )
    hex_test = LinearGeomEvaluate(hex_x, vertices_test)
    hex_newton = Newton(hex_x)

    hex_newton_evaluate = NewtonEvaluate(hex_newton, hex_test)

    for vi, vx in enumerate(vertices_ref):
        to_test = hex_test.x(vx)
        correct = vertices_test[vi]
        assert (
            np.linalg.norm(
                np.array(correct).ravel() - np.array(to_test).ravel(), np.inf
            )
            < 1.0e-15
        )
        to_test = hex_test.f(vx, correct)
        assert np.linalg.norm(np.array(to_test).ravel(), np.inf) < 1.0e-15

    xi_correct0 = -0.6
    xi_correct1 = -0.7
    xi_correct2 = -0.9

    xi_correct = (xi_correct0, xi_correct1, xi_correct2)
    phys = hex_test.x(xi_correct)
    residual, fv = hex_newton_evaluate.residual(xi_correct, phys)
    assert residual < 1.0e-15
    phys_nektar = (-0.960000, -0.970000, -0.990000)

    assert abs(phys_nektar[0] - phys[0]) < 1.0e-8
    assert abs(phys_nektar[1] - phys[1]) < 1.0e-8
    assert abs(phys_nektar[2] - phys[2]) < 1.0e-8

    xi = [0.0, 0.0, 0.0]
    for stepx in range(5):
        residual, fv = hex_newton_evaluate.residual(xi, phys)
        xin = hex_newton_evaluate.step(xi, phys, fv)
        xi[0] = xin[0]
        xi[1] = xin[1]
        xi[2] = xin[2]

    assert abs(xi[0] - xi_correct[0]) < 1.0e-14
    assert abs(xi[1] - xi_correct[1]) < 1.0e-14
    assert abs(xi[2] - xi_correct[2]) < 1.0e-14


def get_geom_type():
    self_test()
    return LinearHex


if __name__ == "__main__":
    self_test()
