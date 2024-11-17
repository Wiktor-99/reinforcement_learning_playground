import rclpy
from cart_pole_reinforcement_learning.cart_pole_learning_control_node import CartPoleReinforcementLearning
from std_msgs.msg import Float64
import numpy as np
import control as ctrl


class LinearApproximationControlNode(CartPoleReinforcementLearning):
    def __init__(self, node_name="linear_approximation_control_node"):
        super().__init__(node_name)
        self.pendulum_mass_kg = 0.455
        self.cart_mass_kg = 1.006
        self.length_to_pendulums_center_mass = 0.435057
        self.pole_inertia = 0.044096

        state_matrix = self.get_state_matrix()
        input_matrix = self.get_input_matrix()
        self.controller = ctrl.place(state_matrix, input_matrix, [-3, -1, -4, -2])
        self.states_used_in_control_loop = np.array([1, 1, 1, 1])
        self.timer = self.create_timer(0.1, self.run)

    def get_state_matrix(self):
        g = 9.81
        return np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [
                    0,
                    g
                    * self.length_to_pendulums_center_mass**2
                    * self.pendulum_mass_kg**2
                    / (
                        -self.pole_inertia * self.cart_mass_kg
                        - self.pole_inertia * self.pendulum_mass_kg
                        - self.cart_mass_kg * self.length_to_pendulums_center_mass**2 * self.pendulum_mass_kg
                    ),
                    0,
                    0,
                ],
                [
                    0,
                    g
                    * self.length_to_pendulums_center_mass
                    * self.pendulum_mass_kg
                    * (-self.cart_mass_kg - self.pendulum_mass_kg)
                    / (
                        -self.pole_inertia * self.cart_mass_kg
                        - self.pole_inertia * self.pendulum_mass_kg
                        - self.cart_mass_kg * self.length_to_pendulums_center_mass**2 * self.pendulum_mass_kg
                    ),
                    0,
                    0,
                ],
            ]
        )

    def get_input_matrix(self):
        return np.array(
            [
                [0],
                [0],
                [
                    (-self.pole_inertia - self.length_to_pendulums_center_mass**2 * self.pendulum_mass_kg)
                    / (
                        -self.pole_inertia * self.cart_mass_kg
                        - self.pole_inertia * self.pendulum_mass_kg
                        - self.cart_mass_kg * self.length_to_pendulums_center_mass**2 * self.pendulum_mass_kg
                    )
                ],
                [
                    self.length_to_pendulums_center_mass
                    * self.pendulum_mass_kg
                    / (
                        -self.pole_inertia * self.cart_mass_kg
                        - self.pole_inertia * self.pendulum_mass_kg
                        - self.cart_mass_kg * self.length_to_pendulums_center_mass**2 * self.pendulum_mass_kg
                    )
                ],
            ]
        )

    def get_control_value(self):
        return -np.dot(self.controller * self.states_used_in_control_loop, self.get_cart_observations())[0]

    def run(self):
        if not self.is_simulation_ready():
            return

        if self.is_simulation_stopped():
            self.restart_learning_loop()
            self.get_logger().info("Simulation stopped. Restarting...")
            return

        self.take_action(Float64(data=self.get_control_value()))


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(LinearApproximationControlNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
