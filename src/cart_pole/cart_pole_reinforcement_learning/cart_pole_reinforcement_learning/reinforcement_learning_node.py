from std_msgs.msg import Float64
from collections.abc import Callable
from cart_pole_reinforcement_learning.cart_pole_learning_control_node import CartPoleReinforcementLearning
from cart_pole_reinforcement_learning.cart_pole_reinforcement_learning_params import (
    reinforcement_learning_node_parameters,
)


class ReinforcementLearningNode(CartPoleReinforcementLearning):
    def __init__(
        self,
        name,
        max_number_of_episodes=None,
        max_number_of_steps=None,
        max_effort_command=None,
        discount_factor=None,
        reward=None,
        optimizer=None,
        loss_function=None,
        model=None,
    ):
        super().__init__(name)

        self.max_number_of_episodes = (
            reinforcement_learning_node_parameters.Params.max_number_of_episodes
            if max_number_of_episodes is None
            else max_number_of_episodes
        )
        self.max_number_of_steps = (
            reinforcement_learning_node_parameters.Params.max_number_of_steps
            if max_number_of_steps is None
            else max_number_of_steps
        )
        self.max_effort_command = (
            reinforcement_learning_node_parameters.Params.max_effort_command
            if max_effort_command is None
            else max_effort_command
        )
        self.discount_factor = (
            reinforcement_learning_node_parameters.Params.discount_factor
            if discount_factor is None
            else discount_factor
        )
        self.reward = reinforcement_learning_node_parameters.Params.reward if max_number_of_steps is None else reward
        self.episode = 0
        self.step = 0
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.model = model

    def is_episode_ended(self) -> bool:
        return self.step == self.max_number_of_steps

    def create_command(self, action: int) -> Float64:
        return Float64(data=self.max_effort_command) if action == 0 else Float64(data=-self.max_effort_command)

    def stop_run_when_learning_ended(self):
        if self.episode == self.max_number_of_episodes:
            quit()

    def advance_episode_when_finished(self, clean_up_function: Callable[[], None] = None):
        if self.is_episode_ended() or self.is_simulation_stopped():
            self.get_logger().info(f"Ended episode: {self.episode} with score: {self.step}")
            self.episode += 1
            self.step = 0
            self.restart_learning_loop()
            clean_up_function and clean_up_function()
