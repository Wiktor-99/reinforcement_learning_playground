import rclpy
from cart_pole_reinforcement_learning.reinforcement_learning_node import ReinforcementLearningNode
import tensorflow as tf
import copy
import numpy as np
from collections import deque


class CartPoleReinforcementDeepQLearningPolicy(ReinforcementLearningNode):
    def __init__(self):
        self.BATCH_SIZE = 32
        self.NUMBER_OF_OUTPUTS = 2
        self.NUMBER_OF_EXPERIMENT_RESULTS = 6
        self.NUMBER_OF_EPISODES_BEFORE_LEARNING = 50
        super().__init__(
            name="cart_pole_deep_q_learning_policy_node",
            optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3),
            loss_function=tf.keras.losses.mse,
            model=self.create_model(),
        )
        self.state = self.get_cart_observations()
        self.replay_buffer = deque(maxlen=2000)
        self.create_timer(0.05, self.run)

    def create_model(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(4,)),
                tf.keras.layers.Dense(32, activation="elu"),
                tf.keras.layers.Dense(32, activation="elu"),
                tf.keras.layers.Dense(32, activation="elu"),
                tf.keras.layers.Dense(32, activation="elu"),
                tf.keras.layers.Dense(self.NUMBER_OF_OUTPUTS),
            ]
        )

    def epsilon_greedy_policy(self, epsilon, state):
        if np.random.rand() < epsilon:
            return np.random.randint(self.NUMBER_OF_OUTPUTS)

        return self.model.predict(state[np.newaxis], verbose=0)[0].argmax()

    def sample_experiences(self):
        indices = np.random.randint(len(self.replay_buffer), size=self.BATCH_SIZE)
        batch = [self.replay_buffer[index] for index in indices]
        return [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(self.NUMBER_OF_EXPERIMENT_RESULTS)
        ]

    def run_one_step(self, epsilon):
        self.action = self.epsilon_greedy_policy(epsilon, np.array(self.get_cart_observations()))
        self.take_action(self.create_command(self.action))
        self.step += 1

    def training_step(self):
        states, actions, rewards, next_states, dones, truncateds = self.sample_experiences()
        target_q_values = self.calculate_target_q_values(rewards, next_states, dones, truncateds)
        grads = self.calculate_gradient(states, actions, target_q_values)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def calculate_gradient(self, states, actions, target_q_values):
        mask = tf.one_hot(actions, self.NUMBER_OF_OUTPUTS)
        with tf.GradientTape() as tape:
            all_q_values = self.model(states)
            q_values = tf.reduce_sum(all_q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_function(target_q_values, q_values))

        grads = tape.gradient(loss, self.model.trainable_variables)
        return grads

    def calculate_target_q_values(self, rewards, next_states, dones, truncateds):
        max_next_q_values = self.model.predict(next_states, verbose=0).max(axis=1)
        runs = 1.0 - (dones | truncateds)
        target_q_values = rewards + runs * self.discount_factor * max_next_q_values
        target_q_values = target_q_values.reshape(-1, 1)
        return target_q_values

    def learn_neural_network(self):
        if self.episode > self.NUMBER_OF_EPISODES_BEFORE_LEARNING and (
            self.is_episode_ended() or self.is_simulation_stopped()
        ):
            self.training_step()

    def run(self):
        if not self.is_simulation_ready():
            return
        self.learn_neural_network()
        self.stop_run_when_learning_ended()
        self.advance_episode_when_finished()
        self.run_one_step(epsilon=max(1 - self.episode / self.max_number_of_episodes, 0.01))
        self.append_data_from_last_step()

    def append_data_from_last_step(self):
        self.replay_buffer.append(
            (
                self.state,
                self.action,
                self.reward,
                self.get_cart_observations(),
                self.is_episode_ended(),
                self.is_simulation_stopped(),
            )
        )
        self.state = self.get_cart_observations()


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(CartPoleReinforcementDeepQLearningPolicy())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
