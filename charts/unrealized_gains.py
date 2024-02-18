import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from database import *
from balance import *
import time
from charts.shared_charts import *


def run_unrealized_gains():
    token = 'pepefork'

    conn = create_connection()
    cursor = conn.cursor()

    # load balances
    token_address = token_addresses[token]
    balances_rows = load_balances_table(cursor, token_address)

    # load prices
    time_to_price, first_price_timestamp = get_price_map(cursor, token_address)

    # calculate URG
    timestamps, gain_percentages = generate_unrealized_gains_by_holdings(balances_rows,
                                                                         time_to_price, first_price_timestamp)

    # generate hourly graph
    prices = load_prices(cursor, token_address)
    create_unrealized_gains_by_holdings_graph(prices, gain_percentages, timestamps, token)


def create_unrealized_gains_by_holdings_graph(prices, gain_percentages, timestamps, token):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        title=dict(text=token, font=dict(size=30))
    )

    for percentage in gain_percentages.keys():
        # address: [realized_gains, timestamps]
        fig.add_trace(
            go.Scatter(x=timestamps, y=gain_percentages[percentage], name=percentage),
            secondary_y=False,
        )

    add_price_trace(prices, fig)

    # Set y-axes titles
    fig.update_yaxes(title_text='total_value', secondary_y=False)
    fig.update_yaxes(title_text="price", secondary_y=True)
    fig.show()


def generate_unrealized_gains_by_holdings(balances_rows, time_to_price, first_price_timestamp):
    print('generate_unrealized_gains_by_holdings')

    start = time.time()
    # wallet_address: [balance, total_cost_basis, remaining_cost_basis, realized_gains, unrealized_gains]
    wallets = {}
    timestamps = []
    gain_percentages = {  # balances are bucketed down
        -100: [],
        0: [],
        25: [],
        50: [],
        75: [],
        100: [],
        200: [],
        500: [],
        1000: [],
        10000: [],
    }

    # get first hour
    current_hour = datetime.datetime.fromisoformat(balances_rows[0][2][:-5] + '00:00')

    i = 0
    print_interval = datetime.timedelta(days=7)
    print_time = current_hour

    # while not end of balances
    while i < len(balances_rows):
        if current_hour >= print_time:
            print(f'current_hour: {current_hour}')
            print_time += print_interval

        # get one hour of balances_rows
        i, to_process_rows = get_balances_changes(i=i, balances_rows=balances_rows,
                                                  before_timestamp=current_hour+datetime.timedelta(minutes=60))

        # process changes to wallets map
        update_wallets(wallets, to_process_rows, current_hour, time_to_price, first_price_timestamp)

        # for all wallets, recalculate all holdings and percentile info
        update_percentiles(gain_percentages, wallets)
        timestamps.append(str(current_hour))

        current_hour += datetime.timedelta(minutes=60)

    print(f'completed generate_unrealized_gains_by_holdings: {time.time() - start}')
    return timestamps, gain_percentages


def get_balances_changes(i, balances_rows, before_timestamp):
    before_str = str(before_timestamp)
    output = []

    # get timestamps before next_hour
    while i < len(balances_rows) and balances_rows[i][2] < before_str:
        output.append(balances_rows[i])
        i += 1

    return i, output


def update_wallets(wallets, to_process_rows, current_hour, time_to_price, first_price_timestamp):
    # row: 0 wallet_address, 1 token_address, 2 timestamp, 3 block, 4 balance, 5 total_cost_basis,
    #   6 remaining_cost_basis, 7 realized_gains
    for row in to_process_rows:
        # wallet_address: [balance, total_cost_basis, remaining_cost_basis, realized_gains, unrealized_gains]
        wallets[row[0]] = [row[4], row[5], row[6], row[7], 0]

    price = get_price(time_to_price, first_price_timestamp, str(current_hour + datetime.timedelta(minutes=60)))

    for address in wallets.keys():
        # unrealized_gains = balance * price
        wallets[address][4] = wallets[address][0] * price


def update_percentiles(gain_percentages, wallets):
    # add new row for percentiles
    for key in gain_percentages.keys():
        gain_percentages[key].append(0)

    for balance, total_cost_basis, remaining_cost_basis, realized_gains, unrealized_gains in wallets.values():
        if balance < 1e-6:
            # don't count if value is too small
            continue
        if remaining_cost_basis == 0:
            # holders before the token is on the market and SQLite REAL precision loss can have 0 remaining_cost_basis
            percentage = 1000000
        else:
            percentage = (unrealized_gains / remaining_cost_basis * 100) - 100

        add_to_percentiles(gain_percentages, percentage, balance)


def add_to_percentiles(gain_percentages, percentage, balance):
    keys = list(gain_percentages.keys())
    for i in range(len(keys)):
        if i == len(keys)-1 or keys[i] <= percentage < keys[i+1]:
            gain_percentages[keys[i]][-1] += balance
            return


def load_balances_table(cursor, token_address):
    query = f"""
        SELECT * FROM balances
        WHERE token_address = '{token_address}'
        ORDER BY block_number;
        """
    cursor.execute(query)
    rows = cursor.fetchall()
    print("Total transactions rows are: ", len(rows))
    return rows


if __name__ == "__main__":
    run_unrealized_gains()
