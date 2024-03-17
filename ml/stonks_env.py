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
            render_mode: Optional[str] = None,
            verbose: bool = False,
            txn_cost=20,
            starting_cash=10000,
            context_window=24,
            granularity = datetime.timedelta(minutes=60),
    ):
        self.token_prices = self.convert_to_hourly_average(token_prices, granularity)
        # debug
        self.check_price_maps(token_prices)
        self.txn_cost = txn_cost
        self.starting_cash = starting_cash
        self.remaining_cash = starting_cash
        self.token_balance = 0

        self.context_window = context_window
        # always start the env with at least enough data to populate the full context_window
        self.cur_time = context_window
        self.granularity = granularity

        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose

        self.action_space = spaces.Box(
            np.array([-1]).astype(np.float32),
            np.array([+1]).astype(np.float32),
        )  # < -0.5: sell, > 0.5: buy, else hold

        # not sure if this is used for anything
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(96, 96, 3), dtype=np.uint8
        )

        self.render_mode = render_mode

    def get_total_balance(self):
        return self.remaining_cash + self.token_balance * self.get_current_price()

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        self.remaining_cash = self.starting_cash
        self.token_balance = 0

        self.cur_time = self.context_window

        super().reset(seed=seed)
        self.reward = 0.0
        self.prev_reward = 0.0

        zero_step = self.step(0)
        return zero_step[0]

    def step(self, action: float):
        stonk_action = self.get_action(action)

        step_reward = 0
        terminated = False
        truncated = False

        self.reward = self.compute_action(stonk_action)

        step_reward = self.reward - self.prev_reward
        self.prev_reward = self.reward

        if self.get_total_balance() < self.txn_cost:
            terminated = True

        if self.cur_time == len(self.token_prices):
            # Truncation due to reaching the end of pricing data
            # This should not be treated as a failure
            truncated = True

        state = []
        # distance to grass
        distance_to_grass = self.dist_scaler.transform(np.array(distance_to_grass).reshape(-1, 1))
        state.extend(distance_to_grass.reshape(1, -1)[0])
        # road segment angles ahead
        angles_ahead = self.angle_scaler.transform(np.array(angles_ahead).reshape(-1, 1))
        state.extend(angles_ahead.reshape(1, -1)[0])
        # speed
        speed = self.speed_scaler.transform(np.array([[self.get_speed(self.car)]]))
        state.append(speed[0][0])
        # wheel angle
        state.append(self.car.wheels[0].joint.angle)  # range -0.42 to 0.42 on front wheels

        self.cur_time += 1.0

        return state, step_reward, terminated, truncated, {}

    # converts continuous -1 to 1 input to BUY, SELL, HOLD enum
    @staticmethod
    def get_action(action: float):
        if action < -0.5:
            return StonkAction.SELL
        if action > 0.5:
            return StonkAction.BUY
        return StonkAction.HOLD

    def get_current_price(self):
        return self.token_prices[self.cur_time]

    def compute_action(self, action: StonkAction):
        if action == StonkAction.BUY:
            # transfer all cash to token
            self.remaining_cash -= self.txn_cost
            self.token_balance = self.remaining_cash / self.get_current_price()
            self.remaining_cash = 0
        elif action == StonkAction.SELL:
            # transfer all token to cash
            self.remaining_cash = self.token_balance * self.get_current_price()
            self.remaining_cash -= self.txn_cost
            self.token_balance = 0
        # do nothing on HOLD

        self.cur_time += 1

        return self.get_total_balance()

    def get_state_price(self):
        price_state = []
        price_i = self.cur_time - self.context_window + 1
        starting_price = self.token_prices[price_i]
        price_state.append(starting_price)
        price_i += 1
        while price_i <= self.cur_time:
            price_state.append(self.token_prices[price_i] / starting_price)
            price_i += 1

        return price_state

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self._render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        elif mode == "none":
            return None
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()

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

    def check_price_maps(self, token_prices):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.update_layout(
            title=dict(text=f'price comparison', font=dict(size=25))
        )


        fig.add_trace(
            go.Scatter(x=self.token_prices.timestamps, y=self.token_prices.data, name='hourly prices'),
            secondary_y=True,
        )
        # fig.update_yaxes(type="log")

        fig.add_trace(
            go.Scatter(x=token_prices.timestamps, y=token_prices.data, name="all prices", line=dict(color='black')),
            secondary_y=True,
        )

        # Set y-axes titles
        fig.update_yaxes(title_text="price", showspikes=True, secondary_y=True)
        fig.update_layout(hovermode="x unified")
        fig.show()
        print('done with chart')
