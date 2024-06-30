import random

from database import *
import numpy as np


class TestDataTypes:
    # centered around 100 with 10 amplitude and 0 slope
    easy_horizontal = 'easy_horizontal'



def create_test_data():
    # create_test_data_table()

    cycles = 200                # how many sine cycles
    resolution = cycles * 10    # how many datapoints to generate

    length = np.pi * 2 * cycles
    wave = np.sin(np.arange(0, length, length / resolution))

    wave = [point * 10 + 100 + np.random.normal(0, 1) for point in wave]

    print(wave)

    save_to_test_data_table(wave, data_type=TestDataTypes.easy_horizontal)

    print('done create_test_data')


def save_to_test_data_table(wave, data_type, start_date='2024-01-02 03:00:00'):
    conn = create_connection()

    cur_date = datetime.datetime.fromisoformat(start_date)

    for point in wave:
        row = (point, data_type, str(cur_date))
        insert_test_data(conn, row)
        cur_date += datetime.timedelta(minutes=60)


def human_policy(state, previous_action):
    # eval reward to beat: 1.82E+16
    return josh_2_policy(state, previous_action)


# version 1: 3.89e16
#   buy below 0.3
#   sell above 0.7
def chloe_and_katelyn_policy(state, previous_action):
    current_price = state[-1]

    if current_price > 0.7:
        return [-1]
    if current_price < 0.3:
        return [1]

    return previous_action


# eval reward: 1.14E+17 eval reward
def tony_policy(state, previous_action):
    # catch outliers
    if state[-1] > 0.75:
        return [-1]
    if state[-1] < 0.25:
        return [1]

    # last high before the dip
    if state[-3] > 0.58 and state[-1] > 0.65:
        # sell all tokens
        return [-1]
    # last low before the rise
    elif state[-3] < 0.45 and state[-1] < 0.35:
        # buy all tokens
        return [1]
    return previous_action


# 3.89E+16
def josh_policy(state, previous_action):
    current_price = state[-1]

    # sell when between .7 and .8
    if current_price > 0.7:
        return [-1]

    # buy when between .3
    if current_price < .3:
        return [1]

    return previous_action


# 2.17E+10: buy below .4, sell above .6
# 4.75E+12: buy one after lowest, sell one after highest
# 1.10E+15: buy 3 after midpoint, sell above .7
# 2.68E+16: buy 3 after midpoint, sell above .7 and bug fixed
def josh_2_policy(state, previous_action):
    current_price = state[-1]

    # 3.89E+16

    # sell when between .7 and .8
    if current_price > 0.7:
        return [-1]

    # buy when between .3
    if current_price < .295:
        return [1]

    return previous_action


RAYQUAZA = None

def peter_policy(state, previous_action):
    print('hello peter')
    return previous_action


if __name__ == "__main__":
    create_test_data()