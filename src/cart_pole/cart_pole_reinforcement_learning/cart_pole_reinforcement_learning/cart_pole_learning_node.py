import rclpy
from rclpy.node import Node
from cart_pole_observation_interface.msg import CartPoleObservation
import numpy as np

UPPER_POLE_LIMIT = 1.4
LOWER_POLE_LIMIT = -1.4


class CartPoleReinforcementLearning(Node):
    def __init__(self):
        super().__init__("cart_pole_learning_node")
        self.observation_subscriber = self.create_subscription(
            CartPoleObservation, "observations", self.store_observation, 10
        )
        self.cart_observations = [0.0, 0.0, 0.0, 0.0]
        self.is_truncated = False

    def store_observation(self, cart_pole_observation: CartPoleObservation):
        self.cart_observations[0] = cart_pole_observation.cart_position
        self.cart_observations[1] = cart_pole_observation.cart_velocity
        self.cart_observations[2] = cart_pole_observation.pole_angle
        self.cart_observations[3] = cart_pole_observation.pole_angular_velocity

    def update_simulation_status(self):
        self.is_truncated = np.isclose(
            self.cart_observations[2], UPPER_POLE_LIMIT, rtol=1e-05, atol=1e-08, equal_nan=False
        ) or np.isclose(self.cart_observations[2], LOWER_POLE_LIMIT, rtol=1e-05, atol=1e-08, equal_nan=False)


def main(args=None):
    rclpy.init(args=args)
    cart_pole_reinforcement_learning = CartPoleReinforcementLearning()
    rclpy.spin(cart_pole_reinforcement_learning)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
