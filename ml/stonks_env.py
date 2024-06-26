from __future__ import annotations

import copy
import datetime
import math
from typing import Optional, Union

from sklearn import preprocessing
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from enum import Enum

from charts.shared_charts import *
from plotly.subplots import make_subplots


class StonkAction(Enum):
    BUY = 1
    HOLD = 2
    SELL = 3


class StonksState:
    context_window = 24
    test_data_state_dim = context_window + 0
    # context_window * 6: 5 percentiles + 1 price
    token_data_state_dim = context_window * 6 + 1
    starting_cash = 10000

    def __init__(self, state, previous_action, total_balance, timestamp):
        # context_window (24) latest prices, remaining cash, token balance, total balance
        self.price_state = state[0:self.context_window]
        self.previous_action = previous_action
        # self.remaining_cash = state[self.context_window]
        # self.token_balance = state[self.context_window + 1]
        # self.total_balance = state[self.context_window + 2]
        self.total_balance = total_balance
        self.timestamp = timestamp


class StonksEnv(gym.Env):
    metadata = {
        "render_modes": [
            "human",
            "none",
        ],
    }

    def __init__(
            self,
            token_prices: TimestampData,
            percentile_volume: TimestampData = None,
            show_price_map=False,
            env_name='',
            render_mode: Optional[str] = None,
            verbose: bool = False,
            txn_cost=20,
            txn_percent=0.4,
            granularity = datetime.timedelta(minutes=60),
    ):
        self.token_prices = self.convert_to_hourly_average(copy.deepcopy(token_prices), granularity)
        self.env_name = env_name
        if show_price_map:
            self.show_price_map()
        self.percentile_volume = self.align_timestamps(copy.deepcopy(percentile_volume), self.token_prices) \
            if percentile_volume is not None else percentile_volume
        self.txn_cost = txn_cost
        self.txn_percent = txn_percent
        self.starting_cash = StonksState.starting_cash
        self.remaining_cash = self.starting_cash
        self.previous_total_balance = self.starting_cash
        self.token_balance = 0
        self.previous_action = StonkAction.SELL

        self.context_window = StonksState.context_window
        # always start the env with at least enough data to populate the full context_window
        self.i = StonksState.context_window
        self.granularity = granularity

        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose

        self.action_space = spaces.Box(
            np.array([-1]).astype(np.float32),
            np.array([+1]).astype(np.float32),
        )  # -1 is all cash, 1 is all token

        # not sure if this is used for anything
        self.observation_space = spaces.Box(
            low=-100_000, high=100_000, shape=(27, 1), dtype=np.uint8
        )

        self.price_scaler = preprocessing.MinMaxScaler()
        min_price = min(self.token_prices.data)
        max_price = max(self.token_prices.data)
        self.price_scaler.fit(np.array([min_price, max_price]).reshape(-1, 1))
        self.volume_scaler = preprocessing.MinMaxScaler()
        self.volume_scaler.fit(np.array([-100, 100]).reshape(-1, 1))
        self.cash_scaler = preprocessing.MinMaxScaler()
        self.cash_scaler.fit(np.array([0, 100_000]).reshape(-1, 1))
        self.token_scaler = preprocessing.MinMaxScaler()
        self.token_scaler.fit(np.array([0, 1000]).reshape(-1, 1))

        self.render_mode = render_mode

    def copy(self, env: StonksEnv):
        # note: make sure each variable being copied is a primitive so that the copy doesn't change the actual value
        self.txn_cost = env.txn_cost
        self.starting_cash = env.starting_cash
        self.remaining_cash = env.remaining_cash
        self.previous_total_balance = env.previous_total_balance
        self.token_balance = env.token_balance
        self.previous_action = env.previous_action

        self.context_window = env.context_window
        self.i = env.i
        self.granularity = env.granularity

        self.reward = env.reward
        self.prev_reward = env.prev_reward
        self.verbose = env.verbose

    def get_total_balance(self):
        return self.remaining_cash + self.token_balance * self.get_current_price()

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset()
        self.remaining_cash = self.starting_cash
        self.token_balance = 0

        self.i = self.context_window

        self.reward = 0.0
        self.prev_reward = 0.0

        zero_step = self.step([-1])  # start with 0 tokens
        return zero_step[0]

    def step(self, action):
        terminated = False
        truncated = False

        self.reward = self.process_discrete_action(action)

        step_reward = self.reward

        if self.get_total_balance() < self.txn_cost:
            terminated = True

        if self.i == len(self.token_prices.data) - 1:
            # Truncation due to reaching the end of pricing data
            # This should not be treated as a failure
            truncated = True

        state = []
        # price state
        price_state = self.get_price_state()
        # price_state = self.price_scaler.transform(np.array(price_state).reshape(-1, 1))
        # state.extend(price_state.reshape(1, -1)[0])
        state.extend(price_state)
        # percentile volume
        if self.percentile_volume is not None:
            percentile_volumes = self.get_unaltered_percentile_volumes()
            for volumes in percentile_volumes:
                unnormalized_volumes = [volume.percent_buy_sell for volume in volumes]
                normalized_volumes = self.volume_scaler.transform(np.array(unnormalized_volumes).reshape(-1, 1))
                state.extend(normalized_volumes.reshape(1, -1)[0])
        # balances
        # state.append(self.cash_scaler.transform(np.array([[self.remaining_cash]]))[0][0])
        # state.append(self.token_scaler.transform(np.array([[self.token_balance]]))[0][0])
        # state.append(self.cash_scaler.transform(np.array([[self.get_total_balance()]]))[0][0])
        # previous_action so that model can know if it will be charged to continue with BUY action
        state.append(self.previous_action)

        return (state, step_reward, terminated, truncated,
                {'stonks_state': StonksState(state, self.previous_action, self.get_total_balance(),
                                             self.token_prices.timestamps[self.i])})

    # converts continuous -1 to 1 input to BUY, SELL, HOLD enum
    @staticmethod
    def get_action(action: [float]):
        if action[0] < 0:
            return StonkAction.SELL
        else:
            return StonkAction.BUY

    def get_current_price(self):
        return self.token_prices.data[self.i]

    def process_discrete_action(self, action):
        # get_current_price returns the last average price so trades are done on the last price data seen by the
        # agent. This should be a good approximation, but it may be more accurate to compute balances based on the
        # current real price instead of acting on a historical average.

        action = self.get_action(action)

        # buy if cash is available
        if action == StonkAction.BUY and self.remaining_cash > 0:
            # transfer all cash to token
            self.remaining_cash *= (1 - self.txn_percent/100)
            self.token_balance = self.remaining_cash / self.get_current_price()
            self.remaining_cash = 0
        # sell if tokens are available
        elif action == StonkAction.SELL and self.token_balance > 0:
            # transfer all token to cash
            self.remaining_cash = self.token_balance * self.get_current_price()
            # self.remaining_cash -= self.txn_cost
            self.token_balance = 0
        # do nothing on HOLD

        self.i += 1

        percent_change = self.get_total_balance() / self.previous_total_balance
        self.previous_total_balance = self.get_total_balance()
        self.previous_action = -1 if action == StonkAction.SELL else 1
        return percent_change

    # -1: only cash, 0: 50/50 split, 1: only hold tokens
    def process_action(self, token_ratio):
        # get_current_price returns the last average price so trades are done on the last price data seen by the
        # agent. This should be a good approximation, but it may be more accurate to compute balances based on the
        # current real price instead of acting on a historical average.

        # rebalance cash/token holdings based on ratio from action
        token_ratio = (token_ratio[0] + 1) / 2  # token_ratio from -1 to 1 because the Actor is designed this way
        total_balance = self.get_total_balance()
        cash_target = total_balance * (1 - token_ratio)

        token_change = (self.remaining_cash - cash_target) / self.get_current_price()

        self.token_balance += token_change
        self.remaining_cash -= token_change * self.get_current_price()

        self.i += 1

        percent_change = self.get_total_balance() / self.previous_total_balance
        self.previous_total_balance = self.get_total_balance()
        return percent_change

    def get_price_state(self):
        price_state = self.token_prices.data[self.i - self.context_window + 1: self.i + 1]
        latest_price = price_state[-1]
        for i in range(len(price_state)):
            price_state[i] = price_state[i] / latest_price

        return price_state

    def get_unaltered_price_state(self):
        return self.token_prices.data[self.i - self.context_window + 1: self.i + 1]

    def get_unaltered_percentile_volumes(self):
        result = []
        for key in self.percentile_volume.data.keys():
            result.append(self.percentile_volume.data[key][self.i - self.context_window + 1: self.i + 1])
        return result

    def render(self):
        # nothing to render
        pass

    @staticmethod
    def convert_to_hourly_average(token_price: TimestampData, granularity: datetime.timedelta):
        current_hour = token_price.first_hour
        before_time_str = str(current_hour + granularity)
        i = 0
        hourly_prices = []
        hourly_timestamps = []

        current_prices = []

        while i < len(token_price.data):
            # append previous hour data into hourly arrays
            if token_price.timestamps[i] >= before_time_str:
                avg_price = 0
                for num in current_prices:
                    avg_price += num
                avg_price /= len(current_prices)

                hourly_prices.append(avg_price)
                hourly_timestamps.append(current_hour)

                # reset temp vars
                current_prices = []
                current_hour += granularity
                before_time_str = str(current_hour + granularity)

            current_prices.append(token_price.data[i])
            i += 1

        return TimestampData(hourly_prices, hourly_timestamps)

    def show_price_map(self):
        fig = make_subplots()
        fig.update_layout(title=dict(text=f'{self.env_name} hourly price', font=dict(size=25)))

        fig.add_trace(go.Scatter(x=self.token_prices.timestamps, y=self.token_prices.data, name='hourly prices'))

        # Set y-axes titles
        fig.update_yaxes(title_text="price", showspikes=True)
        fig.update_layout(hovermode="x unified")
        fig.show()
        print('done with chart')

    @staticmethod
    def align_timestamps(source_data: TimestampData, align_to: TimestampData):
        start = align_to.timestamps[0]
        end = align_to.timestamps[-1]

        start_i = source_data.timestamps.index(str(start))
        end_i = source_data.timestamps.index(str(end))

        for key in source_data.data.keys():
            source_data.data[key] = source_data.data[key][start_i: end_i + 1]

        source_data.timestamps = source_data.timestamps[start_i: end_i + 1]
        source_data.first_hour = datetime.datetime.fromisoformat(source_data.timestamps[0])
        return source_data
