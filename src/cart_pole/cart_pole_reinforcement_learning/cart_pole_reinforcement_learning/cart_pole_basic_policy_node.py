import rclpy
from std_msgs.msg import Float64
from cart_pole_reinforcement_learning.cart_pole_learning_control_node import CartPoleReinforcementLearning


class CartPoleReinforcementBasicPolicy:
    def __init__(self):
        self.MAX_STEPS_IN_EPISODE = 500
        self.MAX_EPISODES = 100
        self.MAX_EFFORT_COMMAND = 5.0
        self.episode = 0
        self.steps = 0
        self.learning_control_node = CartPoleReinforcementLearning("cart_pole_basic_policy_node")
        self.learning_control_node.create_timer(0.05, self.run)

    def create_action(self):
        return (
            Float64(data=self.MAX_EFFORT_COMMAND)
            if self.learning_control_node.get_cart_observations()[2] > 0
            else Float64(data=-self.MAX_EFFORT_COMMAND)
        )

    def run_one_step(self):
        self.learning_control_node.take_action(self.create_action())
        self.steps += 1

    def is_episode_ended(self):
        return self.steps == self.MAX_STEPS_IN_EPISODE

    def run(self):
        if self.episode == self.MAX_EPISODES:
            quit()

        if not self.learning_control_node.is_simulation_ready():
            return

        if self.is_episode_ended() or self.learning_control_node.is_simulation_stopped():
            self.learning_control_node.get_logger().info(f"Ended episode: {self.episode} with score: {self.steps}")
            self.episode += 1
            self.steps = 0
            self.learning_control_node.restart_learning_loop()

        self.run_one_step()


def main(args=None):
    rclpy.init(args=args)
    cart_pole_reinforcement_learning = CartPoleReinforcementBasicPolicy()
    rclpy.spin(cart_pole_reinforcement_learning.learning_control_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
