import rclpy
from std_msgs.msg import Float64
from cart_pole_reinforcement_learning.cart_pole_learning_control_node import CartPoleReinforcementLearning


class CartPoleReinforcementBasicPolicy(CartPoleReinforcementLearning):
    def __init__(self):
        super().__init__("cart_pole_basic_policy_node")
        self.MAX_STEPS_IN_EPISODE = 500
        self.MAX_EPISODES = 100
        self.MAX_EFFORT_COMMAND = 5.0
        self.episode = 0
        self.steps = 0
        self.create_timer(0.05, self.run)

    def create_action(self):
        return (
            Float64(data=self.MAX_EFFORT_COMMAND)
            if self.get_cart_observations()[2] > 0
            else Float64(data=-self.MAX_EFFORT_COMMAND)
        )

    def run_one_step(self):
        self.take_action(self.create_action())
        self.steps += 1

    def is_episode_ended(self):
        return self.steps == self.MAX_STEPS_IN_EPISODE

    def run(self):
        if self.episode == self.MAX_EPISODES:
            quit()

        if not self.is_simulation_ready():
            return

        if self.is_episode_ended() or self.is_simulation_stopped():
            self.get_logger().info(f"Ended episode: {self.episode} with score: {self.steps}")
            self.episode += 1
            self.steps = 0
            self.restart_learning_loop()

        self.run_one_step()


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(CartPoleReinforcementBasicPolicy())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
