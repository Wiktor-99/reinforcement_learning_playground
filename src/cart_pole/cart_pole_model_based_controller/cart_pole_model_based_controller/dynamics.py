import sympy as sp
from sympy import *
from sympy import init_printing


def main():
    time = sp.Symbol("t")
    cart_position = sp.Function("x1")(time)
    pole_angle = sp.Function("x2")(time)
    cart_velocity = sp.Function("x3")(time)
    pole_angular_velocity = sp.Function("x4")(time)
    pendulums_length = sp.Symbol("l")

    x_position = cart_position + pendulums_length * sin(pole_angle)
    y_position = pendulums_length * cos(pole_angle)

    x_velocity = sp.diff(x_position, time)
    y_velocity = sp.diff(y_position, time)
    x_velocity = x_velocity.subs(sp.Derivative(pole_angle, time), pole_angular_velocity).subs(
        sp.Derivative(cart_position, time), cart_velocity
    )
    y_velocity = y_velocity.subs(sp.Derivative(pole_angle, time), pole_angular_velocity).subs(
        sp.Derivative(cart_position, time), cart_velocity
    )

    print(f"x_velocity={x_velocity}")
    print(f"y_velocity={y_velocity}")

    x_accelerations = sp.diff(x_velocity, time)
    y_accelerations = sp.diff(y_velocity, time)
    x_accelerations = x_accelerations.subs(sp.Derivative(pole_angle, time), pole_angular_velocity).subs(
        sp.Derivative(cart_position, time), cart_velocity
    )
    x_accelerations = x_accelerations.subs(sp.Derivative(pole_angle, time), pole_angular_velocity).subs(
        sp.Derivative(cart_position, time), cart_velocity
    )

    print(f"x_accelerations={x_accelerations}")
    print(f"y_accelerations={y_accelerations}")

    pendulum_mass_kg = sp.Symbol("m")
    pole_inertia = sp.Symbol("Ip")
    cart_mass_kg = sp.Symbol("M")
    g = sp.Symbol("g")

    velocity = (x_velocity**2 + y_velocity**2).simplify()
    pole_kinetic_energy = (
        sp.Rational(1, 2) * pendulum_mass_kg * velocity + sp.Rational(1, 2) * pole_inertia * pole_angular_velocity**2
    )
    cart_kinetic_energy = sp.Rational(1, 2) * cart_mass_kg * (cart_velocity**2)

    kinetic_energy = Matrix([pole_kinetic_energy, cart_kinetic_energy])
    print(f"Kinetic energy={kinetic_energy}")

    potential_energy = Matrix([pendulum_mass_kg * g * pendulums_length * cos(pole_angle), 0])
    print(f"Potential energy={potential_energy}")

    lagrangian = kinetic_energy[0] + kinetic_energy[1] - potential_energy[0] - potential_energy[1]
    lagrangian = lagrangian.subs(cart_velocity, sp.Derivative(cart_position, time)).subs(
        pole_angular_velocity, sp.Derivative(pole_angle, time)
    )
    print(f"Lagrangian = {lagrangian}")

    # Euler-Lagrange equations
    q = Matrix([cart_position, pole_angle])
    dq = Matrix([diff(cart_position, time), diff(pole_angle, time)])
    dlagrangian_dx = lagrangian.diff(q)
    dlagrangian_ddx = lagrangian.diff(dq)
    dlagrangian_ddx_dt = diff(dlagrangian_ddx, time)

    E_Le = (
        simplify(dlagrangian_ddx_dt - dlagrangian_dx)
        .subs(sp.Derivative(cart_position, time), cart_velocity)
        .subs(sp.Derivative(pole_angle, time), pole_angular_velocity)
    )

    print(f"\n\nE_Le = ")
    pprint(E_Le)
    QCD = E_Le

    print(f"\n\nQCD = ")
    pprint(QCD, use_unicode=True)

    print(f"shape(QCD) = {QCD.shape}")
    QCD = QCD.expand()

    ddq1_coeff = Matrix([QCD[0].coeff(cart_velocity.diff(time)), QCD[1].coeff(cart_velocity.diff(time))])
    ddq2_coeff = Matrix(
        [QCD[0].coeff(pole_angular_velocity.diff(time)), QCD[1].coeff(pole_angular_velocity.diff(time))]
    )
    Q = Matrix([ddq1_coeff.transpose(), ddq2_coeff.transpose()]).transpose()
    print(f"Q = ")
    pprint(Q)

    dq1_coeff = Matrix([QCD[0].coeff(cart_velocity**2), QCD[1].coeff(cart_velocity**2)])
    dq2_coeff = Matrix([QCD[0].coeff(pole_angular_velocity**2), QCD[1].coeff(pole_angular_velocity**2)])

    ddq = Matrix([cart_velocity.diff(time), pole_angular_velocity.diff(time)])
    dq = Matrix([[cart_velocity], [pole_angular_velocity]])
    q = Matrix([cart_position, pole_angle])

    C = Matrix([dq1_coeff.transpose(), dq2_coeff.transpose()]).transpose() * dq
    C = Matrix([[sp.Rational(0), C[0]], [sp.Rational(0), C[1]]])
    print(f"C = ")
    pprint(C)

    Dq = (QCD - Q * ddq - C * dq).simplify()

    print(f"Dq = ")
    pprint(Dq)

    q_dot = Matrix([[cart_velocity], [pole_angular_velocity]])

    uF = sp.Symbol("uF")  # force applied to the cart
    u = Matrix([[uF], [0]])

    # Compute the second part: -Inverse(Qq) * (Cqqp * q_dot + Dq) + Inverse(Qq) * Matrix([[uF], [0]])
    second_part = -Inverse(Q) * (C * q_dot + Dq) + Inverse(Q) * u

    # Join the two parts together
    fxu = Matrix.vstack(q_dot, second_part)

    print("f(x, u) = ")
    pprint(fxu)

    x = Matrix([cart_position, pole_angle, cart_velocity, pole_angular_velocity])
    A = (
        fxu.jacobian(x)
        .subs(uF, 0)
        .subs(cart_position, 0)
        .subs(pole_angle, 0)
        .subs(cart_velocity, 0)
        .subs(pole_angular_velocity, 0)
    )

    print("A = ")
    pprint(A)

    B = (
        fxu.diff(uF)
        .subs(uF, 0)
        .subs(cart_position, 0)
        .subs(pole_angle, 0)
        .subs(cart_velocity, 0)
        .subs(pole_angular_velocity, 0)
    )
    print("B = ")
    pprint(B)

    print(f"A = {sp.pycode(A)}")
    print(f"B = {sp.pycode(B)}")


if __name__ == "__main__":
    main()
