import random
from copy import deepcopy
from decimal import Decimal

import matplotlib.pyplot as plt

from ml.create_test_data import TestDataTypes, near_optimal_policy
from td3 import TD3
from replay_buffer import ReplayBuffer

from stonks_env import *
from charts.volume import generate_volume
from database import *


class TD3Runner:
    def __init__(self):
        self.state_dim = StonksState.token_data_state_dim
        self.hidden_dim_1 = 128
        self.hidden_dim_2 = 64
        self.action_dim = 1  # -1 to 1 representing buy, sell, hold
        self.max_action = 1  # max upper bound for action
        self.policy_noise = 0.2  # Noise added to target policy during critic update
        self.noise_clip = 0.4  # Range to clip target policy noise
        self.batch_size = 128  # How many timesteps for each training session for the actor and critic
        # BATCH_SIZE = 1024

        self.start_training_note = 'new token, testing learning rate with: '

        self.explore_noise = 0.2  # Std of Gaussian exploration noise
        self.random_policy_steps = 5_000  # Time steps that initial random policy is used

        self.learning_rate = 1e-4

        self.max_train_timesteps = 5_000_000_000
        self.eval_interval = 5000

        self.run_sim = True

        self.save_policy_reward_threshold = 2e4
        self.validation_ratio = 0.2  # ratio of data to reserve for validation

        # self.eval_only = False
        # self.load_file = ''
        self.eval_only = True
        self.load_file = 'models/saved/model_307_2.1e4'

        self.figure_size = (7, 5)
        self.eval_figure_num = 3
        self.validation_figure_num = 4
        self.unseen_data_figure_num = 5

        self.connection = create_connection()
        cursor = self.connection.cursor()
        self.token = altlayer

        if self.load_file == '':
            self.model_name = 'model'
            self.model_name = f'{self.model_name}_{reserve_agent_num(self.connection)}'

            print(f'save_file_name: {self.model_name}')
        else:
            self.model_name = self.load_file

        # load balances
        balances_rows = load_balances_table(cursor, self.token.address)

        # calculate sum of diffs and add to chart output
        timestamps, percentiles = generate_volume(balances_rows, cursor, self.token.address,
                                                  granularity=datetime.timedelta(minutes=60))
        self.percentile_volume = TimestampData(percentiles, timestamps)

        self.training_data_before = '2024-03-12 00:00:00'
        self.training_data_after = '2024-01-26 00:00:00' if self.token == altlayer else None

        price_data = load_structured_prices(cursor, self.token.address,
                                            after_timestamp=self.training_data_after,
                                            before_timestamp=self.training_data_before)
        validation_split = round(len(price_data.data) * (1 - self.validation_ratio))
        training_data = TimestampData(price_data.data[:validation_split], price_data.timestamps[:validation_split])
        validation_data = TimestampData(price_data.data[validation_split:], price_data.timestamps[validation_split:])

        # training envs. eval_env runs the model on training data
        self.train_env = StonksEnv(training_data, percentile_volume=self.percentile_volume,
                                   show_price_map=False)
        self.eval_env = StonksEnv(training_data, percentile_volume=self.percentile_volume)
        self.sim_env = StonksEnv(training_data, percentile_volume=self.percentile_volume)
        # validation env that runs the model on validation data that was never explicitly trained on
        self.validation_env = StonksEnv(validation_data, percentile_volume=self.percentile_volume)

        self.policy = TD3(state_dim=self.state_dim, action_dim=self.action_dim,
                          hidden_dim_1=self.hidden_dim_1, hidden_dim_2=self.hidden_dim_2,
                          max_action=self.max_action, policy_noise=self.policy_noise, noise_clip=self.noise_clip,
                          learning_rate=self.learning_rate)
        if self.load_file != '':
            self.policy.load(f"./{self.load_file}")
            print(f'loaded policy: {self.load_file}')

        self.training_loss = self.TrainingLoss(1)

        plt.ion()

    class TrainingLoss:
        def __init__(self, random_policy_steps):
            self.actor_loss = [0] * random_policy_steps
            self.critic_loss = [0] * random_policy_steps

        def append_losses(self, actor_loss, critic_loss):
            self.actor_loss.append(actor_loss)
            self.critic_loss.append(critic_loss)

    def run(self):
        print(f'''{self.start_training_note}
              token:        {self.token.name}
              explore_noise: {self.explore_noise}
              learning_rate: {self.learning_rate}
              run_sim:      {self.run_sim}
              policy_noise: {self.policy_noise}
              noise_clip:   {self.noise_clip}
              batch_size:   {self.batch_size}
              hidden_dims:  {self.hidden_dim_1, self.hidden_dim_2}
              random_policy_steps: {self.random_policy_steps}
              ''')

        state = self.train_env.reset()
        train_rewards = []
        eval_rewards = []
        validation_rewards = []
        max_validation_reward = -10_000
        max_train_reward = -10_000
        episode_reward = 1
        episode_timesteps = 0
        episode_num = 0

        replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)

        eval_reward, validation_reward = self.test_policy()
        if self.eval_only:
            self.test_policy_on_unseen_data()
            # self.test_policy()
            print(f'total training eval reward: {eval_reward} and total validation reward: {validation_reward}')
            input('hit enter to close graphs')
            return

        start = time.time()

        for t in range(1, self.max_train_timesteps + 1):
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < self.random_policy_steps:
                action = self.train_env.action_space.sample()
            else:
                action = (
                        self.policy.select_action(np.array(state))
                        + np.random.normal(0, self.max_action * self.explore_noise, size=self.action_dim)
                ).clip(-self.max_action, self.max_action)

            if self.run_sim:
                # simulate opposite extreme action
                sim_action = [1] if action[0] < 0 else [-1]
                self.sim_env.copy(self.train_env)
                sim_next_state, sim_reward, sim_terminated, sim_truncated, sim_info = self.sim_env.step(sim_action)
                sim_done = float(sim_terminated or sim_truncated)
                replay_buffer.add(state, sim_action, sim_next_state, sim_reward, sim_done)

            # Perform action
            next_state, reward, terminated, truncated, info = self.train_env.step(action)
            done = float(terminated or truncated)

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done)

            state = next_state
            episode_reward *= reward

            # Train agent after collecting sufficient data
            if t >= self.random_policy_steps:
                actor_loss, critic_loss = self.policy.train(replay_buffer, self.batch_size)
                if t % self.eval_interval == 0:
                    self.training_loss.append_losses(actor_loss.item(), critic_loss.item())

            if done:
                print(
                    f"Total steps: {t + 1}, "
                    f"Episode Num: {episode_num + 1}, "
                    f"Episode steps: {episode_timesteps}, "
                    f"Reward: {Decimal(episode_reward):.2E}, ")
                state, done = self.train_env.reset(), 0
                if episode_reward > max_train_reward:
                    max_train_reward = episode_reward
                # append to both train and eval to keep them with the same number
                train_rewards.append(episode_reward)
                eval_rewards.append(eval_reward)
                validation_rewards.append(validation_reward)
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                episode_reward = 1
                episode_timesteps = 0
                episode_num += 1
                self.update_loss_plot()
                self.update_training_plot(eval_rewards=eval_rewards, train_rewards=train_rewards,
                                          validation_rewards=validation_rewards,
                                          timestep=t)

            if t % self.eval_interval == 0:
                # append to both train and eval to keep them at the index on the chart
                eval_reward, validation_reward = self.test_policy()
                train_rewards.append(episode_reward)
                eval_rewards.append(eval_reward)
                validation_rewards.append(validation_reward)
                print(f'steps/sec: {self.eval_interval / (time.time() - start)}, '
                      f'eval reward: {Decimal(eval_reward):.2E}')

                if validation_reward > max_validation_reward:
                    max_validation_reward = validation_reward
                    if max_validation_reward > self.save_policy_reward_threshold:
                        self.policy.save(f"./models/{self.model_name}")
                        print(f"saved model {self.model_name}")

                self.update_training_plot(eval_rewards=eval_rewards, train_rewards=train_rewards,
                                          validation_rewards=validation_rewards,
                                          timestep=t)

                start = time.time()

        input("Training complete. Press enter to continue: ")
        plt.ioff()
        plt.show()
        self.train_env.close()
        self.eval_env.close()
        self.sim_env.close()
        print(f'total time: {time.time() - start}')

    def update_training_plot(self, eval_rewards, train_rewards, validation_rewards, timestep,
                             show_result=False):
        fig = plt.figure(1, figsize=self.figure_size)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title(f'Training {self.model_name}... dim: {self.hidden_dim_1}x{self.hidden_dim_2}, '
                      f'batch: {self.batch_size}, '
                      f'eval interval: {self.eval_interval}, \n'
                      f'explore_noise: {self.explore_noise} '
                      f'learning_rate: {self.learning_rate}')

        next_eval = abs((timestep % self.eval_interval) - self.eval_interval)
        plt.xlabel(f'Episode ({len(train_rewards)}), '
                   f'next eval: {next_eval}', fontsize=15)
        plt.ylabel(f'Reward (max train: {Decimal(max(eval_rewards)):.2E}, '
                   f'max vali: {Decimal(max(validation_rewards)):.2E})', fontsize=11)
        # plt.plot(train_rewards, label='Train Reward')
        plt.plot(eval_rewards, label='Training Reward')
        plt.plot(validation_rewards, label='Validation Reward')
        plt.yscale("log")
        # plt.hlines(REWARD_THRESHOLD, 0, len(test_rewards), color='r')
        fig.legend()

        fig.canvas.start_event_loop(0.001)  # this updates the plot and doesn't steal window focus

    def update_loss_plot(self):
        fig = plt.figure(2, figsize=self.figure_size)
        plt.clf()
        plt.title(f'Training loss {self.model_name}')

        ax1 = fig.get_axes()[0]
        ax2 = ax1.twinx()

        ax2.plot(self.training_loss.actor_loss, color='b', label='actor_loss')
        ax2.set_ylabel(f'Actor loss {Decimal(self.training_loss.actor_loss[-1]):.2E}', color='b')
        ax1.plot(self.training_loss.critic_loss, color='r', label='critic_loss')
        ax1.set_ylabel(f'Critic loss {Decimal(self.training_loss.critic_loss[-1]):.2E}', color='r')

        ax1.set_xlabel('evals')
        fig.legend()
        fig.canvas.start_event_loop(0.001)  # this updates the plot and doesn't steal window focus

    def test_policy(self):
        eval_reward = self.test_policy_on_env(self.eval_env, self.eval_figure_num)
        validation_reward = self.test_policy_on_env(self.validation_env, self.validation_figure_num)
        return eval_reward, validation_reward

    def test_policy_on_unseen_data(self):
        cursor = create_connection().cursor()
        price_data = load_structured_prices(cursor, self.token.address, after_timestamp=self.training_data_before)
        unseen_data = TimestampData(price_data.data, price_data.timestamps)
        unseen_data_env = StonksEnv(unseen_data, percentile_volume=self.percentile_volume,
                                    show_price_map=True, env_name='unseen data')
        self.test_policy_on_env(unseen_data_env, self.unseen_data_figure_num)

    def test_policy_on_env(self, env, figure_num):
        done = False
        state = env.reset()
        total_reward = 1
        total_balance = 0
        actions = []
        rewards = []
        timestamps = []

        while not done:
            action = self.policy.select_action(np.array(state))
            state, reward, done, truncated, info = env.step(action)
            total_balance = info['stonks_state'].total_balance
            total_reward *= reward
            actions.append(action)
            rewards.append(total_balance)
            timestamps.append(info['stonks_state'].timestamp)
            done = done or truncated

        self.show_eval_chart(actions, rewards, timestamps, figure_num)

        return total_balance

    def show_eval_chart(self, actions, rewards, timestamps, figure_num):
        fig = plt.figure(figure_num, figsize=self.figure_size)
        plt.clf()
        env_type = 'Training'
        if figure_num == self.validation_figure_num:
            env_type = 'Validation'
        if figure_num == self.unseen_data_figure_num:
            env_type = 'Testing unseen data'
        plt.title(f'{self.model_name} {env_type}')

        ax1 = fig.get_axes()[0]
        ax2 = ax1.twinx()

        ax2.plot(timestamps, rewards, color='b', label='reward')
        ax2.set_ylabel(f'{env_type} Rewards {Decimal(rewards[-1]):.2E}', color='b')
        ax1.plot(timestamps, actions, color='r', label='action')
        ax1.set_ylabel(f'{env_type} Actions', color='r')

        ax1.set_xlabel('timestamp')

        fig.legend()
        fig.canvas.draw()
        fig.canvas.start_event_loop(0.001)  # this updates the plot and doesn't steal window focus


if __name__ == "__main__":
    TD3Runner().run()
