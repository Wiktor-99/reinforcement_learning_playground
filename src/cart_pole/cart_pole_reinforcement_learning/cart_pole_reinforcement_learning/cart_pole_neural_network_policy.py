import rclpy
from cart_pole_reinforcement_learning.reinforcement_learning_node import ReinforcementLearningNode
import tensorflow as tf
import copy
import numpy as np


class CartPoleReinforcementNeuralNetworkPolicy(ReinforcementLearningNode):
    def __init__(self):
        super().__init__(
            name="cart_pole_neural_network_node",
            optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3),
            loss_function=tf.keras.losses.binary_crossentropy,
            model=self.create_model(),
        )

        self.iteration = 0
        self.max_number_of_iterations = 100
        self.max_number_of_episodes_per_iteration = 20
        self.rewards = []
        self.gradients = []
        self.episode_rewards = []
        self.episode_gradients = []
        self.create_timer(0.05, self.run)

    def create_model(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="elu"),
                tf.keras.layers.Dense(256, activation="elu"),
                tf.keras.layers.Dense(256, activation="elu"),
                tf.keras.layers.Dense(256, activation="elu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

    def run_one_step(self):
        observations = np.array(copy.deepcopy(self.get_cart_observations()))
        with tf.GradientTape() as tape:
            left_proba = self.model(observations[np.newaxis])
            action = tf.random.uniform([1, 1]) > left_proba
            y_target = tf.constant([[1.0]]) - tf.cast(action, tf.float32)
            loss = tf.reduce_mean(self.loss_function(y_target, left_proba))

        self.gradients.append(tape.gradient(loss, self.model.trainable_variables))
        self.rewards.append(1.0)
        self.take_action(self.create_command(int(action)))
        self.step += self.reward

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
        return self.episode == self.max_number_of_episodes_per_iteration

    def clean_up_at_end_of_episode(self):
        self.episode_rewards.append(self.rewards)
        self.episode_gradients.append(self.gradients)
        self.rewards = []
        self.gradients = []

    def run(self):
        if not self.is_simulation_ready():
            return
        elif self.iteration == self.max_number_of_iterations:
            quit()

        self.advance_episode_when_finished(self.clean_up_at_end_of_episode)
        self.advance_iteration_when_finished()
        self.run_one_step()

    def advance_iteration_when_finished(self):
        if self.is_iteration_ended():
            self.get_logger().info(
                f"Ended {self.iteration} iteration with max final {max([sum(rewards) for rewards in self.episode_rewards])}"
            )
            self.apply_gradients()
            self.episode_gradients = []
            self.episode_rewards = []
            self.episode = 0
            self.iteration += 1

    def apply_gradients(self):
        all_final_rewards = self.discount_and_normalize_rewards()
        all_mean_grads = self.calculate_mean_grads(all_final_rewards)
        self.optimizer.apply_gradients(zip(all_mean_grads, self.model.trainable_variables))

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
    rclpy.spin(CartPoleReinforcementNeuralNetworkPolicy())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
