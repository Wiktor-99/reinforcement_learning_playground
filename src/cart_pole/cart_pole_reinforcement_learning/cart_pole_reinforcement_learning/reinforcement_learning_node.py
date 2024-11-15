from std_msgs.msg import Float64
from cart_pole_reinforcement_learning.cart_pole_learning_control_node import CartPoleReinforcementLearning


class ReinforcementLearningNode(CartPoleReinforcementLearning):
    def __init__(
        self,
        name,
        max_number_of_episodes=100,
        max_number_of_steps=200,
        max_effort_command=5.0,
        discount_factor=0.95,
        reward=1,
        optimizer=None,
        loss_function=None,
        model=None,
    ):
        super().__init__(name)
        self.max_number_of_episodes = max_number_of_episodes
        self.max_number_of_steps = max_number_of_steps
        self.max_effort_command = max_effort_command
        self.discount_factor = discount_factor
        self.reward = reward
        self.episode = 0
        self.step = 0
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.model = model

    def is_episode_ended(self):
        return self.step == self.max_number_of_steps

    def create_command(self, action):
        return Float64(data=self.max_effort_command) if action == 0 else Float64(data=-self.max_effort_command)

    def stop_run_when_learning_ended(self):
        if self.episode == self.max_number_of_episodes:
            quit()

    def advance_episode_when_finished(self, clean_up_function=None):
        if self.is_episode_ended() or self.is_simulation_stopped():
            self.get_logger().info(f"Ended episode: {self.episode} with score: {self.step}")
            self.episode += 1
            self.step = 0
            self.restart_learning_loop()
            clean_up_function and clean_up_function()
