import rclpy
import rclpy.context
from rclpy.node import Node
from cart_pole_observation_interface.msg import CartPoleObservation
from std_msgs.msg import Float64
import numpy as np
from std_srvs.srv import Empty
import time

UPPER_POLE_LIMIT = 1.4
LOWER_POLE_LIMIT = -1.4
MAX_STEPS_IN_EPISODE = 500
MAX_EPISODE = 100
MAX_EFFORT_COMMAND = 5.0
TIME_BETWEEN_COMMANDS = 0.02


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

    def reset_observation(self):
        self.cart_observations = [0.0, 0.0, 0.0, 0.0]
        self.is_truncated = False

    def update_simulation_status(self):
        self.is_truncated = np.isclose(
            self.cart_observations[2], UPPER_POLE_LIMIT, rtol=1e-05, atol=1e-08, equal_nan=False
        ) or np.isclose(self.cart_observations[2], LOWER_POLE_LIMIT, rtol=1e-05, atol=1e-08, equal_nan=False)

    def restart_simulation(self):
        while not self.simulation_reset_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("restart_sim_service not available, waiting again...")
        future = self.simulation_reset_service_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)

    def basic_policy(self):
        return Float64(data=MAX_EFFORT_COMMAND) if self.cart_observations[2] > 0 else Float64(data=-MAX_EFFORT_COMMAND)

    def take_basic_action(self):
        action = self.basic_policy()
        self.effort_command_publisher.publish(action)

    def run_single_episode(self, episode):
        episode_rewards = 0
        for _ in range(MAX_STEPS_IN_EPISODE):
            self.take_basic_action()
            if self.is_truncated:
                break
            episode_rewards += 1
            rclpy.spin_once(self)
            time.sleep(TIME_BETWEEN_COMMANDS)

        self.get_logger().info(f"Ended episode: {episode} with score: {episode_rewards}")
        return episode_rewards

    def run_basic_policy(self):
        rewards = []
        for episode in range(MAX_EPISODE):
            reward = self.run_single_episode(episode)
            rewards.append(reward)
            self.restart_simulation()
            self.reset_observation()

        self.get_logger().info(f"Rewards {rewards}")


def main(args=None):
    rclpy.init(args=args)
    cart_pole_reinforcement_learning = CartPoleReinforcementLearning()
    cart_pole_reinforcement_learning.run_basic_policy()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
