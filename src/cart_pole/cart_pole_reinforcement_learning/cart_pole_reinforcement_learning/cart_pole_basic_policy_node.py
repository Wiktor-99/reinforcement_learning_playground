import rclpy
import time
from std_msgs.msg import Float64
from cart_pole_reinforcement_learning.cart_pole_learning_control_node import CartPoleReinforcementLearning


class CartPoleReinforcementBasicPolicy:
    def __init__(self):
        self.MAX_STEPS_IN_EPISODE = 500
        self.MAX_EPISODE = 100
        self.MAX_EFFORT_COMMAND = 5.0
        self.TIME_BETWEEN_COMMANDS = 0.02
        self.learning_control_node = CartPoleReinforcementLearning("cart_pole_basic_policy_node")

    def basic_policy(self):
        return (
            Float64(data=self.MAX_EFFORT_COMMAND)
            if self.learning_control_node.get_cart_observations()[2] > 0
            else Float64(data=-self.MAX_EFFORT_COMMAND)
        )

    def run_single_episode(self, episode):
        episode_rewards = 0
        for _ in range(self.MAX_STEPS_IN_EPISODE):
            self.learning_control_node.take_action(self.basic_policy())
            if self.learning_control_node.is_simulation_stopped():
                break
            episode_rewards += 1
            rclpy.spin_once(self.learning_control_node)
            time.sleep(self.TIME_BETWEEN_COMMANDS)

        self.learning_control_node.get_logger().info(f"Ended episode: {episode} with score: {episode_rewards}")
        return episode_rewards

    def run_basic_policy(self):
        rewards = []
        for episode in range(self.MAX_EPISODE):
            reward = self.run_single_episode(episode)
            rewards.append(reward)
            self.learning_control_node.restart_simulation()
            self.learning_control_node.reset_observation()

        self.learning_control_node.get_logger().info(f"Rewards {rewards}")


def main(args=None):
    rclpy.init(args=args)
    cart_pole_reinforcement_learning = CartPoleReinforcementBasicPolicy()
    cart_pole_reinforcement_learning.run_basic_policy()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
