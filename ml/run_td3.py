import multiprocessing
import random
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from stonks_env import StonksEnv
from td3 import TD3
from replay_buffer import ReplayBuffer

from charts.shared_charts import *
from database import *
from plotly.subplots import make_subplots

import dash
from dash import dcc
from dash import html
from threading import Thread


# 24 latest price, remaining cash, token balance, total balance
STATE_DIM = 27
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 256
ACTION_DIM = 1      # -1 to 1 representing buy, sell, hold
MAX_ACTION = 1      # max upper bound for action
POLICY_NOISE = 0.2  # Noise added to target policy during critic update
NOISE_CLIP = 0.5    # Range to clip target policy noise
BATCH_SIZE = 512    # How many timesteps for each training session for the actor and critic
# BATCH_SIZE = 1024

EXPLORE_NOISE = 0.1         # Std of Gaussian exploration noise
RANDOM_POLICY_STEPS = 5000  # Time steps that initial random policy is used

MAX_TRAIN_TIMESTEPS = 5_000_000_000
EVAL_INTERVAL = 1000

TRACK_SEED = 123
# TRACK_SEED = random.randint(1, 100000)
# TRACK_SEED = 12147  # first corner sharp high speed

EVAL_ONLY = False
LOAD_FILE = ''
# EVAL_ONLY = True
# LOAD_FILE = 'models/saved/term3_overfit'
# LOAD_FILE = 'models/default_model'

if LOAD_FILE == 'models/saved/term3' or LOAD_FILE == 'models/term3':
    HIDDEN_DIM_1 = 512
    HIDDEN_DIM_2 = 256

plt.ion()
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


def run_td3_bot(argv):
    save_file_name = 'default_model'
    if len(argv) == 2 and argv[0] == '-name':
        save_file_name = argv[1]

    conn = create_connection()
    cursor = conn.cursor()
    token = pepefork

    train_env = StonksEnv(load_structured_prices(cursor, token.address), show_price_map=False)
    eval_env = StonksEnv(load_structured_prices(cursor, token.address))

    policy = TD3(state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 hidden_dim_1=HIDDEN_DIM_1, hidden_dim_2=HIDDEN_DIM_2,
                 max_action=MAX_ACTION, policy_noise=POLICY_NOISE, noise_clip=NOISE_CLIP)
    if LOAD_FILE != '':
        policy.load(f"./{LOAD_FILE}")

    print(f'track_seed: {TRACK_SEED}')
    state = train_env.reset(seed=TRACK_SEED)
    train_rewards = []
    eval_rewards = []
    max_eval_reward = 0
    max_train_reward = 0
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    replay_buffer = ReplayBuffer(STATE_DIM, ACTION_DIM)

    eval_fig = make_subplots(specs=[[{"secondary_y": True}]])
    eval_fig.add_trace(go.Scatter(x=eval_env.token_prices.timestamps, y=eval_env.token_prices.data, name='hourly prices'),
                   secondary_y=True)
    eval_fig.add_trace(go.Scatter(x=eval_env.token_prices.timestamps, y=eval_env.token_prices.data, name='rewards'),
                   secondary_y=False)

    app = dash.Dash()
    app.layout = html.Div([
        dcc.Graph(figure=eval_fig)
    ])

    server_process = multiprocessing.Process(target=run_dash_app, args=[app])
    server_process.start()

    eval_reward = eval_policy(policy, eval_env, eval_fig, app)
    eval_rewards.append(eval_reward)
    if EVAL_ONLY:
        print(f'total reward: {eval_reward}')
        eval_policy(policy, eval_env, eval_fig, app)  # ANOTHER!!
        return

    start = time.time()

    for t in range(1, MAX_TRAIN_TIMESTEPS + 1):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < RANDOM_POLICY_STEPS:
            action = train_env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, MAX_ACTION * EXPLORE_NOISE, size=ACTION_DIM)
            ).clip(-MAX_ACTION, MAX_ACTION)

        # Perform action
        next_state, reward, terminated, truncated, info = train_env.step(action)
        done = float(terminated or truncated)

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= RANDOM_POLICY_STEPS:
            policy.train(replay_buffer, BATCH_SIZE)

        if done:
            print(
                f"Total steps: {t + 1}, "
                f"Episode Num: {episode_num + 1}, "
                f"Episode steps: {episode_timesteps}, "
                f"Reward: {episode_reward:.3f}")
            state, done = train_env.reset(), 0
            if episode_reward > max_train_reward:
                max_train_reward = episode_reward
            # append to both train and eval to keep them with the same number
            train_rewards.append(episode_reward)
            eval_rewards.append(eval_reward)
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            plot_durations(test_rewards=eval_rewards, train_rewards=train_rewards,
                           timestep=t, max_training_reward=max_train_reward)

        if t % EVAL_INTERVAL == 0:
            print(f'steps/sec: {EVAL_INTERVAL / (time.time() - start)}')
            # append to both train and eval to keep them at the index on the chart
            eval_reward = eval_policy(policy, eval_env, eval_fig, app)
            train_rewards.append(episode_reward)
            eval_rewards.append(eval_reward)
            print(f'eval reward: {eval_reward}')

            if eval_reward > max_eval_reward:
                max_eval_reward = eval_reward
                if max_eval_reward > 900:
                    policy.save(f"./models/{save_file_name}")
                    print(f"saved model {save_file_name}")

            plot_durations(test_rewards=eval_rewards, train_rewards=train_rewards,
                           timestep=t, max_training_reward=max_train_reward)

            start = time.time()

    plot_durations(test_rewards=eval_rewards, train_rewards=train_rewards, show_result=True,
                   timestep=MAX_TRAIN_TIMESTEPS+1, max_training_reward=max_train_reward)
    plt.ioff()
    plt.show()
    train_env.close()
    eval_env.close()
    print(f'total time: {time.time() - start}')


def eval_policy(policy, env, fig, app):
    done = False
    state = env.reset()
    total_reward = 0
    rewards = []
    timestamps = []

    while not done:
        action = policy.select_action(np.array(state))
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        rewards.append(total_reward)
        timestamps.append(info['stonks_state'].timestamp)
        done = done or truncated

    show_eval_chart(fig, rewards, timestamps, app)

    return total_reward


def run_dash_app(app):
    app.run_server(debug=False)


def show_eval_chart(fig, rewards, timestamps, app):
    fig.update_layout(title=dict(text=f'Training rewards', font=dict(size=25)))

    # replace rewards data
    reward_trace = fig.data[1]
    reward_trace.x = timestamps
    reward_trace.y = rewards

    # add a second chart for buy/sell/hold behavior with actual action num and lines at 0.5 and -0.5

    # Set y-axes titles
    fig.update_yaxes(title_text="price", showspikes=True)
    fig.update_layout(hovermode="x unified")
    # fig.show()

    app.layout = html.Div([
        dcc.Graph(figure=fig)
    ])


def plot_durations(test_rewards, train_rewards, timestep, max_training_reward,
                   show_result=False):
    fig = plt.figure(1, figsize=(9, 6))
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title(f'Training... dim: {HIDDEN_DIM_1}x{HIDDEN_DIM_2}, '
                  f'batch: {BATCH_SIZE}, '
                  f'eval interval: {EVAL_INTERVAL}')

    next_eval = abs((timestep % EVAL_INTERVAL) - EVAL_INTERVAL)
    plt.xlabel(f'Episode ({len(train_rewards)}), '
               f'next eval: {next_eval}', fontsize=20)
    plt.ylabel(f'Reward (max eval: {max(test_rewards):5.1f}'
               f', max train: {max_training_reward:5.1f})', fontsize=15)
    plt.plot(train_rewards, label='Train Reward')
    plt.plot(test_rewards, label='Test Reward')
    # plt.hlines(REWARD_THRESHOLD, 0, len(test_rewards), color='r')
    plt.legend(loc='upper left')

    fig.canvas.start_event_loop(0.001)  # this updates the plot and doesn't steal window focus
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


if __name__ == "__main__":
    run_td3_bot(sys.argv[1:])
