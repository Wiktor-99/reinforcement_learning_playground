import sympy as sp
from sympy import *
from sympy import init_printing


def main():
    t = sp.Symbol("t")  # time

    x1 = sp.Function("x1")(t)  # position of the cart
    x2 = sp.Function("x2")(t)  # angle of the pendulum
    x3 = sp.Function("x3")(t)  # velocity of the cart
    x4 = sp.Function("x4")(t)  # angular velocity of the pendulum

    # x3 = sp.diff(x1(t), t) # velocity of the cart
    # x4 = sp.diff(x2(t), t) # angular velocity of the pendulum

    l = sp.Symbol("l")  # length of the pendulum

    # Motion equations

    # Positions
    x_p = x1 + l * sin(x2)
    y_p = l * cos(x2)

    # Velocities
    dx_p = sp.diff(x_p, t)
    dy_p = sp.diff(y_p, t)

    dx_p = dx_p.subs(sp.Derivative(x2, t), x4).subs(sp.Derivative(x1, t), x3)
    dy_p = dy_p.subs(sp.Derivative(x2, t), x4).subs(sp.Derivative(x1, t), x3)

    print(f"dx_p =")
    pprint(dx_p)
    print(f"dy_p = ")
    pprint(dy_p)

    # Accelerations
    ddx_p = sp.diff(dx_p, t)
    ddy_p = sp.diff(dy_p, t)

    ddx_p = ddx_p.subs(sp.Derivative(x2, t), x4).subs(sp.Derivative(x1, t), x3)
    ddx_p = ddx_p.subs(sp.Derivative(x2, t), x4).subs(sp.Derivative(x1, t), x3)

    print(f"ddx_p")
    pprint(ddx_p)
    print(f"ddy_p = ")
    pprint(ddy_p)

    # Energies
    m = sp.Symbol("m")  # mass of the pendulum
    Ip = sp.Symbol("Ip")  # inertia of the pendulum
    M = sp.Symbol("M")  # mass of the cart
    g = sp.Symbol("g")  # gravity

    v = dx_p**2 + dy_p**2
    v = v.simplify()
    print(f"v = ")
    pprint(v)

    # Kinetic energy
    Tm = sp.Rational(1, 2) * m * v + sp.Rational(1, 2) * Ip * x4**2
    TM = sp.Rational(1, 2) * M * (x3**2)

    T = Matrix([Tm, TM])
    print(f"T = ")
    pprint(T)

    # Potential energy
    V = Matrix([m * g * l * cos(x2), 0])
    print(f"V = =")
    pprint(V)
    print()

    # Lagrangian
    L = T[0] + T[1] - V[0] - V[1]

    L = L.subs(x3, sp.Derivative(x1, t)).subs(x4, sp.Derivative(x2, t))
    print(f"L = {L}")

    # Euler-Lagrange equations
    q = Matrix([x1, x2])
    dq = Matrix([diff(x1, t), diff(x2, t)])

    dL_dx = L.diff(q)
    dL_ddx = L.diff(dq)
    dL_ddx_dt = diff(dL_ddx, t)

    print(f"dL_dx = {dL_dx}")

    E_Le = simplify(dL_ddx_dt - dL_dx).subs(sp.Derivative(x1, t), x3).subs(sp.Derivative(x2, t), x4)
    init_printing()
    print(f"\n\nE_Le = ")
    pprint(E_Le)
    QCD = E_Le

    print(f"\n\nQCD = ")
    pprint(QCD, use_unicode=True)

    print(f"shape(QCD) = {QCD.shape}")
    QCD = QCD.expand()

    ddq1_coeff = Matrix([QCD[0].coeff(x3.diff(t)), QCD[1].coeff(x3.diff(t))])
    ddq2_coeff = Matrix([QCD[0].coeff(x4.diff(t)), QCD[1].coeff(x4.diff(t))])
    Q = Matrix([ddq1_coeff.transpose(), ddq2_coeff.transpose()]).transpose()
    print(f"Q = ")
    pprint(Q)

    dq1_coeff = Matrix([QCD[0].coeff(x3**2), QCD[1].coeff(x3**2)])
    dq2_coeff = Matrix([QCD[0].coeff(x4**2), QCD[1].coeff(x4**2)])

    ddq = Matrix([x3.diff(t), x4.diff(t)])
    dq = Matrix([[x3], [x4]])
    q = Matrix([x1, x2])

    C = Matrix([dq1_coeff.transpose(), dq2_coeff.transpose()]).transpose() * dq
    C = Matrix([[sp.Rational(0), C[0]], [sp.Rational(0), C[1]]])
    print(f"C = ")
    pprint(C)

    Dq = (QCD - Q * ddq - C * dq).simplify()

    print(f"Dq = ")
    pprint(Dq)

    q_dot = Matrix([[x3], [x4]])

    uF = sp.Symbol("uF")  # force applied to the cart
    u = Matrix([[uF], [0]])

    # Compute the second part: -Inverse(Qq) * (Cqqp * q_dot + Dq) + Inverse(Qq) * Matrix([[uF], [0]])
    second_part = -Inverse(Q) * (C * q_dot + Dq) + Inverse(Q) * u

    # Join the two parts together
    fxu = Matrix.vstack(q_dot, second_part)

    print("f(x, u) = ")
    pprint(fxu)

    x = Matrix([x1, x2, x3, x4])
    A = fxu.jacobian(x).subs(uF, 0).subs(x1, 0).subs(x2, 0).subs(x3, 0).subs(x4, 0)

    print("A = ")
    pprint(A)

    B = fxu.diff(uF).subs(uF, 0).subs(x1, 0).subs(x2, 0).subs(x3, 0).subs(x4, 0)
    print("B = ")
    pprint(B)

    print(f"A = {sp.pycode(A)}")
    print(f"B = {sp.pycode(B)}")


if __name__ == "__main__":
    main()
