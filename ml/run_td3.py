import matplotlib.pyplot as plt

from td3 import TD3
from replay_buffer import ReplayBuffer

from stonks_env import *
from database import *
from plotly.subplots import make_subplots


class TD3Runner:
    def __init__(self, eval_only=False, load_file=''):
        # 24 latest price, remaining cash, token balance, total balance
        self.state_dim = 27
        self.hidden_dim_1 = 256
        self.hidden_dim_2 = 128
        self.action_dim = 1  # -1 to 1 representing buy, sell, hold
        self.max_action = 1  # max upper bound for action
        self.policy_noise = 0.1  # Noise added to target policy during critic update
        self.noise_clip = 0.3  # Range to clip target policy noise
        self.batch_size = 512  # How many timesteps for each training session for the actor and critic
        # BATCH_SIZE = 1024

        self.explore_noise = 1.0  # Std of Gaussian exploration noise
        self.random_policy_steps = 5000  # Time steps that initial random policy is used

        self.max_train_timesteps = 5_000_000_000
        self.eval_interval = 5000

        self.eval_only = eval_only
        self.load_file = load_file
        # self.eval_only = True
        # self.load_file = 'models/saved/term3_overfit'
        # self.load_file = 'models/default_model'

        conn = create_connection()
        cursor = conn.cursor()
        token = pepefork

        self.model_name = 'model'
        self.model_name = f'{self.model_name}_{reserve_agent_num(conn)}'

        print(f'save_file_name: {self.model_name}')

        self.train_env = StonksEnv(load_structured_prices(cursor, token.address), show_price_map=False)
        self.eval_env = StonksEnv(load_structured_prices(cursor, token.address))

        self.policy = TD3(state_dim=self.state_dim, action_dim=self.action_dim,
                          hidden_dim_1=self.hidden_dim_1, hidden_dim_2=self.hidden_dim_2,
                          max_action=self.max_action, policy_noise=self.policy_noise, noise_clip=self.noise_clip)
        if self.load_file != '':
            self.policy.load(f"./{self.load_file}")

        plt.ion()

    def run(self):
        print(f'start training with explore_noise: {self.explore_noise}')

        state = self.train_env.reset()
        train_rewards = []
        eval_rewards = []
        max_eval_reward = 0
        max_train_reward = 0
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)

        eval_fig = make_subplots(specs=[[{"secondary_y": True}]])
        eval_fig.add_trace(go.Scatter(x=self.eval_env.token_prices.timestamps,
                                      y=self.eval_env.token_prices.data,
                                      name='hourly prices'),
                           secondary_y=True)
        eval_fig.add_trace(go.Scatter(x=self.eval_env.token_prices.timestamps,
                                      y=self.eval_env.token_prices.data,
                                      name='rewards'),
                           secondary_y=False)

        eval_reward = self.eval_policy(eval_fig)
        eval_rewards.append(eval_reward)
        if self.eval_only:
            print(f'total reward: {eval_reward}')
            self.eval_policy(eval_fig, show_plotly_chart=True)  # ANOTHER!!
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

            # Perform action
            next_state, reward, terminated, truncated, info = self.train_env.step(action)
            done = float(terminated or truncated)

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= self.random_policy_steps:
                self.policy.train(replay_buffer, self.batch_size)

            if done:
                print(
                    f"Total steps: {t + 1}, "
                    f"Episode Num: {episode_num + 1}, "
                    f"Episode steps: {episode_timesteps}, "
                    f"Reward: {episode_reward:.3f}")
                state, done = self.train_env.reset(), 0
                if episode_reward > max_train_reward:
                    max_train_reward = episode_reward
                # append to both train and eval to keep them with the same number
                train_rewards.append(episode_reward)
                eval_rewards.append(eval_reward)
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                self.plot_durations(test_rewards=eval_rewards, train_rewards=train_rewards,
                                    timestep=t, max_training_reward=max_train_reward)

            if t % self.eval_interval == 0:
                print(f'steps/sec: {self.eval_interval / (time.time() - start)}')
                # append to both train and eval to keep them at the index on the chart
                eval_reward = self.eval_policy(eval_fig)
                train_rewards.append(episode_reward)
                eval_rewards.append(eval_reward)
                print(f'eval reward: {eval_reward}')

                if eval_reward > max_eval_reward:
                    max_eval_reward = eval_reward
                    if max_eval_reward > 20_000:
                        self.policy.save(f"./models/{self.model_name}")
                        print(f"saved model {self.model_name}")

                self.plot_durations(test_rewards=eval_rewards, train_rewards=train_rewards,
                                    timestep=t, max_training_reward=max_train_reward)

                start = time.time()

        input("Training complete. Press enter to continue: ")
        plt.ioff()
        plt.show()
        self.train_env.close()
        self.eval_env.close()
        print(f'total time: {time.time() - start}')

    def eval_policy(self, fig, show_plotly_chart=False):
        done = False
        state = self.eval_env.reset()
        total_reward = 0
        rewards = []
        timestamps = []

        while not done:
            action = self.policy.select_action(np.array(state))
            state, reward, done, truncated, info = self.eval_env.step(action)
            total_reward += reward
            rewards.append(total_reward)
            timestamps.append(info['stonks_state'].timestamp)
            done = done or truncated

        if show_plotly_chart:
            self.show_plotly_eval_chart(fig, rewards, timestamps)

        return total_reward

    @staticmethod
    def show_plotly_eval_chart(fig, rewards, timestamps):
        fig.update_layout(title=dict(text=f'Training rewards', font=dict(size=25)))

        # replace rewards data
        reward_trace = fig.data[1]
        reward_trace.x = timestamps
        reward_trace.y = rewards

        # add a second chart for buy/sell/hold behavior with actual action num and lines at 0.5 and -0.5

        # Set y-axes titles
        fig.update_yaxes(title_text="price", showspikes=True)
        fig.update_layout(hovermode="x unified")

        fig.show()

    def plot_durations(self, test_rewards, train_rewards, timestep, max_training_reward,
                       show_result=False):
        fig = plt.figure(1, figsize=(9, 6))
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title(f'Training {self.model_name}... dim: {self.hidden_dim_1}x{self.hidden_dim_2}, '
                      f'batch: {self.batch_size}, '
                      f'eval interval: {self.eval_interval}, '
                      f'explore_noise: {self.explore_noise}')

        next_eval = abs((timestep % self.eval_interval) - self.eval_interval)
        plt.xlabel(f'Episode ({len(train_rewards)}), '
                   f'next eval: {next_eval}', fontsize=20)
        plt.ylabel(f'Reward (max eval: {max(test_rewards):5.1f}'
                   f', max train: {max_training_reward:5.1f})', fontsize=15)
        plt.plot(train_rewards, label='Train Reward')
        plt.plot(test_rewards, label='Test Reward')
        # plt.hlines(REWARD_THRESHOLD, 0, len(test_rewards), color='r')
        plt.legend(loc='upper left')

        fig.canvas.start_event_loop(0.001)  # this updates the plot and doesn't steal window focus


if __name__ == "__main__":
    TD3Runner().run()
