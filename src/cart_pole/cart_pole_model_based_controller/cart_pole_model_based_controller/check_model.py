import numpy as np
import control as ctrl
from scipy.signal import lsim, lti
import matplotlib.pyplot as plt


def get_A(l, m, M, Ip):
    g = 9.81
    return np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, g * l**2 * m**2 / (-Ip * M - Ip * m - M * l**2 * m), 0, 0],
            [0, g * l * m * (-M - m) / (-Ip * M - Ip * m - M * l**2 * m), 0, 0],
        ]
    )


def get_B(l, m, M, Ip):
    return np.array(
        [
            [0],
            [0],
            [(-Ip - l**2 * m) / (-Ip * M - Ip * m - M * l**2 * m)],
            [l * m / (-Ip * M - Ip * m - M * l**2 * m)],
        ]
    )


def main():

    m = 0.455  # Mass of the pendulum [kg]
    M = 1.006  # Mass of the cart
    l = 0.435057  # Length to the pendulum's center of mass [m]
    Ip = 0.044096

    A = get_A(l, m, M, Ip)
    B = get_B(l, m, M, Ip)
    ctrb = ctrl.ctrb(A, B)
    print(f"ctrb: {np.linalg.matrix_rank(ctrb, tol=1.0e-10)}")
    print(f"eigs A: {np.linalg.eigvals(A)}")

    # Add feedback control
    K = ctrl.place(A, B, [-1, -2, -3, -4])  # Place the poles of the system
    print(f"K: {K}")

    A_cl = A - B @ K
    print(f"eigs A_cl: {np.linalg.eigvals(A_cl)}")
    ctrb = ctrl.ctrb(A_cl, B)
    print(f"ctrb: {np.linalg.matrix_rank(ctrb, tol=1.0e-10)}")
    C = np.array([0, 1, 0, 1])
    x = np.array([0.5, 0.1, 0, 0.1])

    print(f"State: {x}")
    print(f"C: {C}")
    print(f"KC: {K*C}")
    y = np.dot(K * C, x)
    print(f"y: {y}")

    D = 0.0

    system = lti(A_cl, B, C, D)
    t = np.linspace(0, 50, num=200)

    u = np.ones_like(t)
    _, y, x = lsim(system, u, t)

    plt.plot(t, y)
    plt.grid(alpha=0.3)
    plt.xlabel("t")
    plt.show()


if __name__ == "__main__":
    main()
