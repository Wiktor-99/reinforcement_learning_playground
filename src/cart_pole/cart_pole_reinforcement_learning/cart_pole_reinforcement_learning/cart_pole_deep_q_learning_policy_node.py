import rclpy
from std_msgs.msg import Float64
from cart_pole_reinforcement_learning.cart_pole_learning_control_node import CartPoleReinforcementLearning
import tensorflow as tf
import copy
import numpy as np
from collections import deque


class CartPoleReinforcementDeepQLearningPolicy:
    def __init__(self):
        self.MAX_EPISODES = 600
        self.MAX_STEPS = 300
        self.MAX_EFFORT_COMMAND = 5.0
        self.BATCH_SIZE = 32
        self.NUMBER_OF_OUTPUTS = 2
        self.DISCOUNT_FACTOR = 0.95
        self.NUMBER_OF_EXPERIMENT_RESULTS = 6
        self.REWARD = 1
        self.NUMBER_OF_EPISODES_BEFORE_LEARNING = 50

        self.episode = 0
        self.steps = 0
        self.learning_control_node = CartPoleReinforcementLearning("cart_pole_deep_q_learning_policy_node")
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32, activation="elu", input_shape=[4]),
                tf.keras.layers.Dense(32, activation="elu"),
                tf.keras.layers.Dense(32, activation="elu"),
                tf.keras.layers.Dense(32, activation="elu"),
                tf.keras.layers.Dense(self.NUMBER_OF_OUTPUTS),
            ]
        )
        self.replay_buffer = deque(maxlen=2000)
        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-3)
        self.loss_fn = tf.keras.losses.mse
        self.learning_control_node.create_timer(0.05, self.run)

    def is_episode_ended(self):
        return self.steps == self.MAX_STEPS

    def create_command(self, action):
        return Float64(data=self.MAX_EFFORT_COMMAND) if action == 0 else Float64(data=-self.MAX_EFFORT_COMMAND)

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

    def play_one_step(self, epsilon):
        state = np.array(copy.deepcopy(self.learning_control_node.get_cart_observations()))
        action = self.epsilon_greedy_policy(epsilon, state)
        self.learning_control_node.take_action(self.create_command(action))
        next_state = np.array(copy.deepcopy(self.learning_control_node.get_cart_observations()))
        self.replay_buffer.append(
            (
                state,
                action,
                self.REWARD,
                next_state,
                self.is_episode_ended(),
                self.learning_control_node.is_simulation_stopped(),
            )
        )
        self.steps += 1

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
            loss = tf.reduce_mean(self.loss_fn(target_q_values, q_values))

        grads = tape.gradient(loss, self.model.trainable_variables)
        return grads

    def calculate_target_q_values(self, rewards, next_states, dones, truncateds):
        max_next_q_values = self.model.predict(next_states, verbose=0).max(axis=1)
        runs = 1.0 - (dones | truncateds)
        target_q_values = rewards + runs * self.DISCOUNT_FACTOR * max_next_q_values
        target_q_values = target_q_values.reshape(-1, 1)
        return target_q_values

    def learn_neural_network(self):
        if self.episode > self.NUMBER_OF_EPISODES_BEFORE_LEARNING:
            self.training_step()

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

        self.run_single_episode()

    def run_single_episode(self):
        self.play_one_step(epsilon=max(1 - self.episode / self.MAX_EPISODES, 0.01))
        self.learn_neural_network()


def main(args=None):
    rclpy.init(args=args)
    cart_pole_reinforcement_learning = CartPoleReinforcementDeepQLearningPolicy()
    rclpy.spin(cart_pole_reinforcement_learning.learning_control_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
