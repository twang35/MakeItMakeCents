from dataclasses import dataclass

from plotly.subplots import make_subplots
from charts.shared_charts import *
from database import *


# get buy and sell activity for all users
# then split by all-time high balance percentile
# then add exchange addresses as a category for balance percentile
def run_volume():
    print('run_volume')
    token = pepefork

    conn = create_connection()
    cursor = conn.cursor()

    # load balances
    balances_rows = load_balances_table(cursor, token.address)
    prices = load_prices(cursor, token.address)

    # calculate sum of diffs and add to chart output
    timestamps, percentiles = generate_volume(balances_rows, cursor, token.address)

    # generate hourly graph
    create_volume_graph(prices, percentiles, timestamps, token, left_offset=3)


@dataclass
class Volume:
    buy: float
    sell: float


def generate_volume(balances_rows, cursor, token_address):
    print(f'running generate_volume')

    start = time.time()
    # wallet_address: balance
    wallet_balances = {}
    timestamps = []
    percentiles = {  # balances are bucketed down
        0.1: [],
        1: [],
        10: [],
        50: [],
        100: [],
    }
    wallet_percentiles = generate_wallet_percentiles(cursor, percentiles, token_address)

    # get first hour
    current_hour = datetime.datetime.fromisoformat(balances_rows[0][BalancesColumns.timestamp][:-5] + '00:00')

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
                                                  before_timestamp=current_hour + datetime.timedelta(minutes=60))

        # process changes to wallets map
        wallet_diffs = process_balances_changes(wallet_balances, to_process_rows)

        # update percentiles
        update_balance_percentiles(percentiles, wallet_diffs, wallet_percentiles)
        timestamps.append(str(current_hour))

        current_hour += datetime.timedelta(minutes=60)

    print(f'completed generate_percentiles: {time.time() - start}')
    return timestamps, percentiles


def generate_wallet_percentiles(cursor, percentiles, token_address):
    wallet_percentiles = {}

    # load the existing wallets from the balances table
    balances_rows = get_largest_alltime_wallet_balances(cursor, token_address)

    # balances: [(balance, wallet_address), ()...]
    balances = [(row[1], row[0]) for row in balances_rows]
    balances.sort(reverse=True)

    i = 0
    percentile_i = -1
    for key in percentiles.keys():
        # update percentile_i to the location of the end of the percentile key
        if i > percentile_i:
            percentile_i = (key / 100 * len(balances)) - 1

        # add the percentile key mapping to all addresses under that percentile_i
        while i < len(balances) and i <= percentile_i:
            wallet_percentiles[balances[i][1]] = key
            i += 1

    return wallet_percentiles


def build_wallets(wallets, balances_rows):
    # wallet_address: [balance, total_cost_basis, remaining_cost_basis, realized_gains]
    for row in balances_rows:
        wallet_address, token_balance, total_cost_basis, remaining_cost_basis, realized_gains, _ = row
        wallets[wallet_address] = [token_balance, total_cost_basis, remaining_cost_basis, realized_gains]

    return wallets


# get all diffs from balances_rows and update the wallets
def process_balances_changes(wallet_balances, to_process_rows):
    # wallet_address: volume
    diffs = {}

    for row in to_process_rows:
        wallet_address = row[BalancesColumns.wallet_address]
        if wallet_address not in wallet_balances:
            wallet_balances[wallet_address] = 0

        if wallet_address not in diffs:
            diffs[wallet_address] = Volume(0, 0)

        new_balance = row[BalancesColumns.balance]
        diff = new_balance - wallet_balances[wallet_address]

        if diff > 0:
            diffs[wallet_address].buy += diff
        else:
            diffs[wallet_address].sell += diff

        wallet_balances[wallet_address] = new_balance

    return diffs


# add volume from wallet_diffs to percentiles
def update_balance_percentiles(percentiles, wallet_diffs, wallet_percentiles):
    for key in percentiles:
        percentiles[key].append(Volume(0, 0))

    for wallet_address, volume in wallet_diffs.items():
        percentiles[wallet_percentiles[wallet_address]][-1].buy += volume.buy
        percentiles[wallet_percentiles[wallet_address]][-1].sell += volume.sell


def create_volume_graph(prices, percentiles, timestamps, token, left_offset=0):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        title=dict(text=f'{token.name} volume', font=dict(size=25))
    )

    for percentile in percentiles.keys():
        buy_data = [element.buy for element in percentiles[percentile]]
        sell_data = [element.sell for element in percentiles[percentile]]
        fig.add_trace(
            go.Scatter(x=timestamps[left_offset:], y=buy_data[left_offset:], name=f'{percentile} buy'),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=timestamps[left_offset:], y=sell_data[left_offset:], name=f'{percentile} sell'),
            secondary_y=False,
        )
    # fig.update_yaxes(type="log")

    add_price_trace(prices, fig, left_offset=100)

    fig.update_layout(legend_title_text='percentiles')
    # Set y-axes titles
    fig.update_yaxes(title_text='token amount', range=[-10000000000000, 10000000000000], secondary_y=False)
    fig.update_yaxes(title_text="price", showspikes=True, secondary_y=True)
    fig.update_layout(hovermode="x unified")
    fig.show()


if __name__ == "__main__":
    run_volume()
