import rclpy
from rclpy.node import Node
from cart_pole_observation_interface.msg import CartPoleObservation
from std_msgs.msg import Float64
import numpy as np
from std_srvs.srv import Empty

UPPER_POLE_LIMIT = 1.4
LOWER_POLE_LIMIT = -1.4


class CartPoleReinforcementLearning(Node):
    def __init__(self):
        super().__init__("cart_pole_learning_node")
        self.observation_subscriber = self.create_subscription(
            CartPoleObservation, "observations", self.store_observation, 10
        )
        self.simulation_reset_service_client = self.create_client(Empty, "restart_sim_service")
        self.effort_command_publisher = self.create_publisher(Float64, "effort_cmd", 10)
        self.cart_observations = [0.0, 0.0, 0.0, 0.0]
        self.is_truncated = False

    def store_observation(self, cart_pole_observation: CartPoleObservation):
        self.cart_observations[0] = cart_pole_observation.cart_position
        self.cart_observations[1] = cart_pole_observation.cart_velocity
        self.cart_observations[2] = cart_pole_observation.pole_angle
        self.cart_observations[3] = cart_pole_observation.pole_angular_velocity
        self.update_simulation_status()

    def update_simulation_status(self):
        self.is_truncated = np.isclose(
            self.cart_observations[2], UPPER_POLE_LIMIT, rtol=1e-05, atol=1e-08, equal_nan=False
        ) or np.isclose(self.cart_observations[2], LOWER_POLE_LIMIT, rtol=1e-05, atol=1e-08, equal_nan=False)

    def restart_simulation(self):
        while not self.simulation_reset_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("restart_sim_service not available, waiting again...")

        return self.simulation_reset_service_client.call_async(Empty.Request())


def main(args=None):
    rclpy.init(args=args)
    cart_pole_reinforcement_learning = CartPoleReinforcementLearning()
    rclpy.spin(cart_pole_reinforcement_learning)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
