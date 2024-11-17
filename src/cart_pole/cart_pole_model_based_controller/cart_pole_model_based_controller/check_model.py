import numpy as np
import control
from scipy.signal import lsim, lti
import matplotlib.pyplot as plt


def get_state_matrix(length_to_pendulums_center_mass, pendulum_mass_kg, cart_mass_kg, pole_inertia):
    g = 9.81
    return np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [
                0,
                g
                * length_to_pendulums_center_mass**2
                * pendulum_mass_kg**2
                / (
                    -pole_inertia * cart_mass_kg
                    - pole_inertia * pendulum_mass_kg
                    - cart_mass_kg * length_to_pendulums_center_mass**2 * pendulum_mass_kg
                ),
                0,
                0,
            ],
            [
                0,
                g
                * length_to_pendulums_center_mass
                * pendulum_mass_kg
                * (-cart_mass_kg - pendulum_mass_kg)
                / (
                    -pole_inertia * cart_mass_kg
                    - pole_inertia * pendulum_mass_kg
                    - cart_mass_kg * length_to_pendulums_center_mass**2 * pendulum_mass_kg
                ),
                0,
                0,
            ],
        ]
    )


def get_input_matrix(length_to_pendulums_center_mass, pendulum_mass_kg, cart_mass_kg, pole_inertia):
    return np.array(
        [
            [0],
            [0],
            [
                (-pole_inertia - length_to_pendulums_center_mass**2 * pendulum_mass_kg)
                / (
                    -pole_inertia * cart_mass_kg
                    - pole_inertia * pendulum_mass_kg
                    - cart_mass_kg * length_to_pendulums_center_mass**2 * pendulum_mass_kg
                )
            ],
            [
                length_to_pendulums_center_mass
                * pendulum_mass_kg
                / (
                    -pole_inertia * cart_mass_kg
                    - pole_inertia * pendulum_mass_kg
                    - cart_mass_kg * length_to_pendulums_center_mass**2 * pendulum_mass_kg
                )
            ],
        ]
    )


def main():
    pendulum_mass_kg = 0.455
    cart_mass_kg = 1.006
    length_to_pendulums_center_mass = 0.435057
    pole_inertia = 0.044096

    state_matrix = get_state_matrix(length_to_pendulums_center_mass, pendulum_mass_kg, cart_mass_kg, pole_inertia)
    input_matrix = get_input_matrix(length_to_pendulums_center_mass, pendulum_mass_kg, cart_mass_kg, pole_inertia)
    controllability_matrix = control.ctrb(state_matrix, input_matrix)

    print(f"Controllability matrix's rank: {np.linalg.matrix_rank(controllability_matrix, tol=1.0e-10)}")
    print(f"Eigen values of state matrix (A): {np.linalg.eigvals(state_matrix)}")

    poles = [-1, -2, -3, -4]
    controller = control.place(state_matrix, input_matrix, poles)
    print(f"Controller (K): {controller}")

    state_matrix_new_eigen_values = state_matrix - input_matrix @ controller
    print(f"System with new eigen values after adding controller: {np.linalg.eigvals(state_matrix_new_eigen_values)}")

    controllability_matrix_with_new_eigen_values = control.ctrb(state_matrix_new_eigen_values, input_matrix)
    print(
        f"Controllability matrix's rank: {np.linalg.matrix_rank(controllability_matrix_with_new_eigen_values, tol=1.0e-10)}"
    )

    states_used_in_control_loop = np.array([0, 1, 0, 1])
    state_vector = np.array([0.5, 0.1, 0, 0.1])

    print(f"State: {state_vector}")
    print(f"States to use in controller: {states_used_in_control_loop}")
    print(f"Controller with certain states being used: {controller * states_used_in_control_loop}")
    print(f"Controller's: {np.dot(controller * states_used_in_control_loop, state_vector)}")

    D = 0.0
    system = lti(state_matrix_new_eigen_values, input_matrix, states_used_in_control_loop, D)
    time = np.linspace(0, 50, num=200)
    u = np.ones_like(time)
    _, y, x = lsim(system, u, time)
    plt.plot(time, y)
    plt.grid(alpha=0.3)
    plt.xlabel("time")
    plt.show()


if __name__ == "__main__":
    main()
