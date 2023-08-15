"""
Defines the X maps for Quadrilaterals embedded in 3D.
"""


from newton_generation import *


class LinearQuadEmbed3D(LinearBase):
    def __init__(self):
        # 5th "vertex" is penalty vector for the mapping and is orthogonal to
        # the plane the quad is embedded in
        num_vertices = 5
        ndim = 3
        name = "linear_2d"
        namespace = "QuadrilateralEmbed3D"
        x_description = """
X(xi) = 0.25 * v0 * (1 - xi_0) * (1 - xi_1) + 
        0.25 * v1 * (1 + xi_0) * (1 - xi_1) + 
        0.25 * v3 * (1 - xi_0) * (1 + xi_1) + 
        0.25 * v2 * (1 + xi_0) * (1 + xi_1) + V4 * xi_2
"""
        LinearBase.__init__(self, num_vertices, ndim, name, namespace, x_description)

    def get_x(self, xi):

        v = self.vertices
        x = (
            0.25 * v[0] * (1 - xi[0]) * (1 - xi[1])
            + 0.25 * v[1] * (1 + xi[0]) * (1 - xi[1])
            + 0.25 * v[3] * (1 - xi[0]) * (1 + xi[1])
            + 0.25 * v[2] * (1 + xi[0]) * (1 + xi[1])
            + xi[2] * v[4]
        )
        return x


def self_test():

    quad = LinearQuadEmbed3D()

    vertices_ref = (
        (-1.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    quad_ref = LinearGeomEvaluate(quad, vertices_ref)

    for vx in vertices_ref:
        to_test = quad_ref.x(vx)
        correct = vx
        assert (
            np.linalg.norm(
                np.array(correct).ravel() - np.array(to_test).ravel(), np.inf
            )
            < 1.0e-15
        )
        to_test = quad_ref.f(vx, vx)
        assert np.linalg.norm(np.array(to_test).ravel(), np.inf) < 1.0e-15

    vertices_test = (
        (-3.0, -2.0, 1.0),
        (1.0, -1.0, 1.0),
        (2.0, 2.0, 1.0),
        (-1.0, 4.0, 1.0),
        (0.0, 0.0, 1.0),
    )
    quad_test = LinearGeomEvaluate(quad, vertices_test)
    quad_newton = Newton(quad)

    quad_newton_evaluate = NewtonEvaluate(quad_newton, quad_test)

    xi_correct0 = -0.9
    xi_correct1 = 0.8
    xi_correct2 = 0.0
    xi_correct = (xi_correct0, xi_correct1, xi_correct2)
    phys = quad_test.x(xi_correct)
    residual, fv = quad_newton_evaluate.residual(xi_correct, phys)
    assert residual < 1.0e-15

    xi = [0.0, 0.0, 0.0]
    for stepx in range(5):
        residual, fv = quad_newton_evaluate.residual(xi, phys)
        xin = quad_newton_evaluate.step(xi, phys, fv)
        xi[0] = xin[0]
        xi[1] = xin[1]

    assert abs(xi[0] - xi_correct[0]) < 1.0e-14
    assert abs(xi[1] - xi_correct[1]) < 1.0e-14
    assert abs(xi[2]) < 1.0e-14


def get_geom_type():
    self_test()
    return LinearQuadEmbed3D


if __name__ == "__main__":
    self_test()
