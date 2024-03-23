from __future__ import annotations

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
            show_price_map = False,
            render_mode: Optional[str] = None,
            verbose: bool = False,
            txn_cost=20,
            starting_cash=10000,
            context_window=24,
            granularity = datetime.timedelta(minutes=60),
    ):
        self.token_prices = self.convert_to_hourly_average(token_prices, granularity)
        if show_price_map:
            self.show_price_map()
        self.txn_cost = txn_cost
        self.starting_cash = starting_cash
        self.remaining_cash = starting_cash
        self.token_balance = 0

        self.context_window = context_window
        # always start the env with at least enough data to populate the full context_window
        self.i = context_window
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
        self.price_scaler.fit(np.array([80, 120]).reshape(-1, 1))
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
        self.token_balance = env.token_balance

        self.context_window = env.context_window
        # always start the env with at least enough data to populate the full context_window
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

    class StonksState:
        def __init__(self, state, context_window, timestamp):
            self.price_state = state[0:context_window]
            self.remaining_cash = state[context_window]
            self.token_balance = state[context_window + 1]
            self.total_balance = state[context_window + 2]
            self.timestamp = timestamp

    def step(self, action):
        terminated = False
        truncated = False

        self.reward = self.process_action(action)

        step_reward = self.reward - self.prev_reward
        self.prev_reward = self.reward

        if self.get_total_balance() < self.txn_cost:
            terminated = True

        if self.i == len(self.token_prices.data) - 1:
            # Truncation due to reaching the end of pricing data
            # This should not be treated as a failure
            truncated = True

        state = []
        # price state
        price_state = self.get_unaltered_price_state()
        price_state = self.price_scaler.transform(np.array(price_state).reshape(-1, 1))
        state.extend(price_state.reshape(1, -1)[0])
        # balances
        state.append(self.cash_scaler.transform(np.array([[self.remaining_cash]]))[0][0])
        state.append(self.token_scaler.transform(np.array([[self.token_balance]]))[0][0])
        state.append(self.cash_scaler.transform(np.array([[self.get_total_balance()]]))[0][0])

        return (state, step_reward, terminated, truncated,
                {'stonks_state': self.StonksState(state, self.context_window, self.token_prices.timestamps[self.i])})

    # converts continuous -1 to 1 input to BUY, SELL, HOLD enum
    @staticmethod
    def get_action(action: float):
        if action < 0:
            return StonkAction.SELL
        else:
            return StonkAction.BUY

    def get_current_price(self):
        return self.token_prices.data[self.i]

    def process_stonk_action(self, action: StonkAction):
        # get_current_price returns the last average price so trades are done on the last price data seen by the
        # agent. This should be a good approximation, but it may be more accurate to compute balances based on the
        # current real price instead of acting on a historical average.

        # buy if cash is available
        if action == StonkAction.BUY and self.remaining_cash > 0:
            # transfer all cash to token
            self.remaining_cash -= self.txn_cost
            self.token_balance = self.remaining_cash / self.get_current_price()
            self.remaining_cash = 0
        # sell if tokens are available
        elif action == StonkAction.SELL and self.token_balance > 0:
            # transfer all token to cash
            self.remaining_cash = self.token_balance * self.get_current_price()
            self.remaining_cash -= self.txn_cost
            self.token_balance = 0
        # do nothing on HOLD

        self.i += 1

        return self.get_total_balance()

    # -1: only cash, 0: 50/50 split, 1: only hold tokens
    def process_action(self, token_ratio):
        # get_current_price returns the last average price so trades are done on the last price data seen by the
        # agent. This should be a good approximation, but it may be more accurate to compute balances based on the
        # current real price instead of acting on a historical average.

        # rebalance cash/token holdings based on ratio from action
        token_ratio = (token_ratio[0] + 1) / 2  # token_ratio starts from -1 to 1 because the Actor is designed this way
        total_balance = self.get_total_balance()
        cash_target = total_balance * (1 - token_ratio)

        token_change = (self.remaining_cash - cash_target) / self.get_current_price()

        self.token_balance += token_change
        self.remaining_cash -= token_change * self.get_current_price()

        self.i += 1

        return self.get_total_balance()

    def get_price_state(self):
        price_state = []
        price_i = self.i - self.context_window + 1
        starting_price = self.token_prices.data[price_i]
        price_state.append(starting_price)
        price_i += 1
        while price_i <= self.i:
            price_state.append(self.token_prices.data[price_i] / starting_price)
            price_i += 1

        return price_state

    def get_unaltered_price_state(self):
        return self.token_prices.data[self.i - self.context_window + 1: self.i + 1]

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
        fig.update_layout(title=dict(text=f'hourly price', font=dict(size=25)))

        fig.add_trace(go.Scatter(x=self.token_prices.timestamps, y=self.token_prices.data, name='hourly prices'))

        # Set y-axes titles
        fig.update_yaxes(title_text="price", showspikes=True)
        fig.update_layout(hovermode="x unified")
        fig.show()
        print('done with chart')
