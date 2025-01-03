from rclpy.node import Node
from cart_pole_observation_interface.msg import CartPoleObservation
from std_msgs.msg import Float64
import numpy as np
from std_srvs.srv import Empty
from rclpy.task import Future
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from rclpy.client import Client


class CartPoleReinforcementLearning(Node):
    def __init__(self, node_name="cart_pole_learning_node"):
        super().__init__(node_name)
        self.observation_subscriber: Subscription = self.create_subscription(
            CartPoleObservation, "observations", self.store_observation, 10
        )
        self.simulation_reset_service_client: Client = self.create_client(Empty, "restart_sim_service")
        self.effort_command_publisher: Publisher = self.create_publisher(Float64, "effort_cmd", 10)
        self.cart_observations: list[float] = [0.0, 0.0, 0.0, 0.0]
        self.is_truncated: bool = False
        self.UPPER_POLE_LIMIT: float = 1.4
        self.LOWER_POLE_LIMIT: float = -1.4
        self.restarting_future: Future = None

    def store_observation(self, cart_pole_observation: CartPoleObservation):
        self.cart_observations[0] = cart_pole_observation.cart_position
        self.cart_observations[1] = cart_pole_observation.pole_angle
        self.cart_observations[2] = cart_pole_observation.cart_velocity
        self.cart_observations[3] = cart_pole_observation.pole_angular_velocity
        self.update_simulation_status()

    def get_cart_observations(self) -> list[float]:
        return [obs for obs in self.cart_observations]

    def is_simulation_stopped(self) -> bool:
        return self.is_truncated

    def take_action(self, action: Float64):
        self.effort_command_publisher.publish(action)

    def reset_observation(self):
        self.cart_observations = [0.0, 0.0, 0.0, 0.0]
        self.is_truncated = False

    def update_simulation_status(self):
        self.is_truncated = np.isclose(
            self.cart_observations[1],
            self.UPPER_POLE_LIMIT,
            rtol=1e-05,
            atol=1e-08,
            equal_nan=False,
        ) or np.isclose(
            self.cart_observations[1],
            self.LOWER_POLE_LIMIT,
            rtol=1e-05,
            atol=1e-08,
            equal_nan=False,
        )

    def restart_simulation(self):
        while not self.simulation_reset_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("restart_sim_service not available, waiting again...")
        self.restarting_future = self.simulation_reset_service_client.call_async(Empty.Request())

    def is_simulation_ready(self) -> bool:
        if self.restarting_future is None:
            return True
        try:
            return self.restarting_future.done()
        except:
            return False

    def restart_learning_loop(self):
        self.restart_simulation()
        self.reset_observation()
