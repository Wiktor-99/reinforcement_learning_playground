import rclpy
import time
from std_msgs.msg import Float64
from cart_pole_reinforcement_learning.cart_pole_learning_control_node import CartPoleReinforcementLearning
import tensorflow as tf
import copy
import numpy as np


class CartPoleReinforcementNeuralNetworkPolicy:
    def __init__(self):
        self.MAX_STEPS_IN_EPISODE = 500
        self.MAX_ITERATION = 10
        self.MAX_EPISODE_PER_UPDATE = 10
        self.MAX_EFFORT_COMMAND = 5.0
        self.step = 0
        self.episode = 0
        self.iteration = 0
        self.learning_control_node = CartPoleReinforcementLearning("cart_pole_neural_network_node")
        self.loss_fn = tf.keras.losses.binary_crossentropy
        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01)
        self.discount_factor = 0.97
        self.rewards = []
        self.gradients = []
        self.episode_rewards = []
        self.episode_gradients = []
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(15, activation="elu"),
                tf.keras.layers.Dense(20, activation="elu"),
                tf.keras.layers.Dense(10, activation="elu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.learning_control_node.create_timer(0.01, self.run)

    def create_command(self, action):
        return Float64(data=self.MAX_EFFORT_COMMAND) if action == 0 else Float64(data=-self.MAX_EFFORT_COMMAND)

    def run_one_step(self):
        observations = np.array(copy.deepcopy(self.learning_control_node.get_cart_observations()))
        with tf.GradientTape() as tape:
            left_proba = self.model(observations[np.newaxis])
            action = tf.random.uniform([1, 1]) > left_proba
            y_target = tf.constant([[1.0]]) - tf.cast(action, tf.float32)
            loss = tf.reduce_mean(self.loss_fn(y_target, left_proba))

        self.gradients.append(tape.gradient(loss, self.model.trainable_variables))
        self.rewards.append(1.0)
        self.learning_control_node.take_action(self.create_command(int(action)))
        self.step += 1

    def discount_rewards(self, rewards):
        discounted = np.array(rewards)
        for step in range(len(rewards) - 2, -1, -1):
            discounted[step] += discounted[step + 1] * self.discount_factor
        return discounted

    def discount_and_normalize_rewards(self):
        all_discounted_rewards = [self.discount_rewards(rewards) for rewards in self.episode_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

    def is_iteration_ended(self):
        return self.episode == self.MAX_EPISODE_PER_UPDATE

    def is_episode_ended(self):
        return self.step == self.MAX_STEPS_IN_EPISODE

    def run(self):
        if not self.learning_control_node.is_simulation_ready():
            return

        if self.iteration == self.MAX_ITERATION:
            quit()

        if self.is_episode_ended() or self.learning_control_node.is_simulation_stopped():
            self.episode_rewards.append(self.rewards)
            self.episode_gradients.append(self.gradients)
            self.rewards = []
            self.gradients = []
            self.episode += 1
            self.learning_control_node.get_logger().info(f"Ended episode: {self.episode} with score: {self.step}")
            self.learning_control_node.restart_learning_loop()
            self.step = 0

        if self.is_iteration_ended():
            self.iteration += 1
            all_final_rewards = self.discount_and_normalize_rewards()
            all_mean_grads = self.calculate_mean_grads(all_final_rewards)
            self.optimizer.apply_gradients(zip(all_mean_grads, self.model.trainable_variables))
            self.learning_control_node.get_logger().info(
                f"Ended {self.iteration} iteration with max final {max([sum(rewards) for rewards in self.episode_rewards])}"
            )
            self.episode_gradients = []
            self.episode_rewards = []
            self.episode = 0

        self.run_one_step()

    def calculate_mean_grads(self, all_final_rewards):
        all_mean_grads = []
        for var_index in range(len(self.model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [
                    final_reward * self.episode_gradients[episode_index][step][var_index]
                    for episode_index, final_rewards in enumerate(all_final_rewards)
                    for step, final_reward in enumerate(final_rewards)
                ],
                axis=0,
            )
            all_mean_grads.append(mean_grads)
        return all_mean_grads


def main(args=None):
    rclpy.init(args=args)
    cart_pole_reinforcement_learning = CartPoleReinforcementNeuralNetworkPolicy()
    rclpy.spin(cart_pole_reinforcement_learning.learning_control_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
