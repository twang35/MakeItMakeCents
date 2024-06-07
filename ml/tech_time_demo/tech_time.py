import random
from copy import deepcopy
from decimal import Decimal

import matplotlib.pyplot as plt

from ml.create_test_data import TestDataTypes, human_policy
from ml.td3 import TD3
from ml.replay_buffer import ReplayBuffer

from tech_time_env import *
from database import *


class TechTime:
    def __init__(self, eval_only=False, load_file=''):
        # self.eval_only = False
        # self.ml_agent = True
        # self.load_file = load_file

        self.eval_only = True
        self.ml_agent = False
        self.load_file = 'models/model_462'  # 2.86E16
        # self.load_file = 'models/model_457'  # 3.57E16

        if self.eval_only:
            if self.ml_agent is False:
                print('running eval on human agent')
            else:
                print(f'running eval on ml agent: {self.load_file}')


        self.state_dim = TechTimeState.state_dim
        self.hidden_dim_1 = 64
        self.hidden_dim_2 = 32
        self.action_dim = 1  # -1 to 1 representing buy, sell, hold
        self.max_action = 1  # max upper bound for action
        self.policy_noise = 0.1  # Noise added to target policy during critic update
        self.noise_clip = 0.2  # Range to clip target policy noise
        self.batch_size = 128  # How many timesteps for each training session for the actor and critic
        # BATCH_SIZE = 1024

        self.start_training_note = 'higher explore noise, no sim training, larger context window with:'

        self.explore_noise = 0.1  # Std of Gaussian exploration noise
        self.random_policy_steps = 5_000  # Time steps that initial random policy is used

        self.learning_rate = 4e-5

        self.max_train_timesteps = 5_000_000_000
        self.eval_interval = 5000

        self.run_sim = True

        self.figure_size = (8, 5)

        conn = create_connection()
        cursor = conn.cursor()

        self.model_name = 'model'
        self.model_name = f'{self.model_name}_{reserve_agent_num(conn)}'

        if self.eval_only is False:
            print(f'save_file_name: {self.model_name}')

        self.train_env = TechTimeEnv(load_structured_test_data_prices(cursor, TestDataTypes.easy_horizontal),
                                     show_price_map=False)
        self.eval_env = TechTimeEnv(load_structured_test_data_prices(cursor, TestDataTypes.easy_horizontal),
                                    show_price_map=False)
        self.sim_env = TechTimeEnv(load_structured_test_data_prices(cursor, TestDataTypes.easy_horizontal))

        self.policy = TD3(state_dim=self.state_dim, action_dim=self.action_dim,
                          hidden_dim_1=self.hidden_dim_1, hidden_dim_2=self.hidden_dim_2,
                          max_action=self.max_action, policy_noise=self.policy_noise, noise_clip=self.noise_clip,
                          learning_rate=self.learning_rate)
        if self.load_file != '':
            self.policy.load(f"./{self.load_file}")

        self.training_loss = self.TrainingLoss(self.random_policy_steps)

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
        max_eval_reward = -10_000
        max_train_reward = -10_000
        episode_reward = 1
        episode_timesteps = 0
        episode_num = 0

        replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)

        eval_reward = self.eval_policy()
        eval_rewards.append(eval_reward)
        if self.eval_only:
            eval_reward = self.eval_policy(show_chart=True, ml_agent=self.ml_agent, show_plotly=True)
            print(f'total reward: {eval_reward}')
            print(f'total training eval reward: {eval_reward}')
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
                self.training_loss.append_losses(actor_loss.item(), critic_loss.item())

            if done:
                print(
                    f"Total steps: {t + 1}, "
                    f"Episode Num: {episode_num + 1}, "
                    f"Episode steps: {episode_timesteps}, "
                    f"Reward: {Decimal(episode_reward):.2E}, ")
                    # f"Actor loss: {Decimal(self.training_loss.actor_loss[-1]):.2E}, "
                    # f"Critic loss: {Decimal(self.training_loss.critic_loss[1]):.2E}, ")
                state, done = self.train_env.reset(), 0
                if episode_reward > max_train_reward:
                    max_train_reward = episode_reward
                # append to both train and eval to keep them with the same number
                train_rewards.append(episode_reward)
                eval_rewards.append(eval_reward)
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                episode_reward = 1
                episode_timesteps = 0
                episode_num += 1
                self.update_training_plot(test_rewards=eval_rewards, train_rewards=train_rewards,
                                          timestep=t, max_training_reward=max_train_reward)
                self.update_loss_plot()

            if t % self.eval_interval == 0:
                # append to both train and eval to keep them at the index on the chart
                eval_reward = self.eval_policy(show_chart=True)
                train_rewards.append(episode_reward)
                eval_rewards.append(eval_reward)
                print(f'steps/sec: {self.eval_interval / (time.time() - start)}, '
                      f'eval reward: {Decimal(eval_reward):.2E}')

                if eval_reward > max_eval_reward:
                    max_eval_reward = eval_reward
                    if max_eval_reward > 1e14:
                        self.policy.save(f"./models/{self.model_name}")
                        print(f"saved model {self.model_name}")

                self.update_training_plot(test_rewards=eval_rewards, train_rewards=train_rewards,
                                          timestep=t, max_training_reward=max_train_reward)

                start = time.time()

        input("Training complete. Press enter to continue: ")
        plt.ioff()
        plt.show()
        self.train_env.close()
        self.eval_env.close()
        print(f'total time: {time.time() - start}')

    def eval_policy(self, show_chart=False, ml_agent=False, show_plotly=False):
        done = False
        state = self.eval_env.reset()
        total_reward = 1
        actions = []
        rewards = []
        timestamps = []

        while not done:
            action = self.policy.select_action(np.array(state))
            state, reward, done, truncated, info = self.eval_env.step(action)
            total_balance = info['stonks_state'].total_balance
            total_reward *= reward
            actions.append(action)
            rewards.append(total_balance)
            timestamps.append(info['stonks_state'].timestamp)
            done = done or truncated

        if show_chart:
            self.show_eval_chart(actions, rewards, timestamps)
        if show_plotly:
            self.eval_custom_policy(ml_agent=ml_agent)

        return total_reward

    def eval_custom_policy(self, ml_agent):
        done = False
        state = self.eval_env.reset()
        total_reward = 1
        actions = []
        rewards = []
        timestamps = []
        previous_action = [-1]

        while not done:
            if ml_agent:
                action = self.policy.select_action(np.array(state))
            else:
                action = human_policy(np.array(state), previous_action)
            state, reward, done, truncated, info = self.eval_env.step(action)
            total_balance = info['stonks_state'].total_balance
            total_reward *= reward
            actions.append(action)
            rewards.append(total_balance)
            timestamps.append(info['stonks_state'].timestamp)
            previous_action = action
            done = done or truncated

        self.show_plotly_eval_chart(actions, rewards, timestamps)

        return total_reward

    def show_plotly_eval_chart(self, actions, rewards, timestamps):
        fig = make_subplots()
        fig.update_layout(title=dict(text=f'eval results', font=dict(size=25)))

        rewards = [reward/TechTimeState.starting_cash for reward in rewards]

        fig.add_trace(go.Scatter(x=timestamps, y=rewards, name='total rewards'))

        fig.update_yaxes(title_text=f'reward: {rewards[-1]:.2E}', showspikes=True)
        fig.update_layout(hovermode='x unified')
        fig.show()

    def show_eval_chart(self, actions, rewards, timestamps):
        fig = plt.figure(2, figsize=self.figure_size)
        plt.clf()
        plt.title(f'{self.model_name} Eval')

        ax1 = fig.get_axes()[0]
        ax2 = ax1.twinx()

        rewards = [reward/TechTimeState.starting_cash for reward in rewards]
        ax2.plot(timestamps, rewards, color='b', label='reward')
        ax2.set_ylabel(f'Eval Rewards {Decimal(rewards[-1]):.2E}', color='b')
        ax1.plot(timestamps, actions, color='r', label='action')
        ax1.set_ylabel('Eval Actions', color='r')

        ax1.set_xlabel('timestamp')

        # add a second chart for buy/sell/hold behavior with actual action num and lines at 0.5 and -0.5

        fig.legend()
        fig.canvas.start_event_loop(0.001)  # this updates the plot and doesn't steal window focus

    def update_training_plot(self, test_rewards, train_rewards, timestep, max_training_reward,
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
        plt.ylabel(f'Reward (max eval: {Decimal(max(test_rewards)):.2E}, '
                   f'max train: {Decimal(max_training_reward):.2E})', fontsize=13)
        plt.plot(train_rewards, label='Train Reward')
        plt.plot(test_rewards, label='Eval Reward')
        # plt.hlines(REWARD_THRESHOLD, 0, len(test_rewards), color='r')
        plt.legend(loc='upper left')

        fig.canvas.start_event_loop(0.001)  # this updates the plot and doesn't steal window focus

    def update_loss_plot(self):
        fig = plt.figure(3, figsize=self.figure_size)
        plt.clf()
        plt.title(f'Training loss {self.model_name}')

        ax1 = fig.get_axes()[0]
        ax2 = ax1.twinx()

        ax2.plot(self.training_loss.actor_loss, color='b', label='actor_loss')
        ax2.set_ylabel(f'Actor loss {Decimal(self.training_loss.actor_loss[-1]):.2E}', color='b')
        ax1.plot(self.training_loss.critic_loss, color='r', label='critic_loss')
        ax1.set_ylabel(f'Critic loss {Decimal(self.training_loss.critic_loss[-1]):.2E}', color='r')

        ax1.set_xlabel('episode')
        fig.legend()
        fig.canvas.start_event_loop(0.001)  # this updates the plot and doesn't steal window focus


if __name__ == "__main__":
    TechTime().run()