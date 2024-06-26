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
    timestamps, percentiles = generate_volume(balances_rows, cursor, token.address,
                                              granularity=datetime.timedelta(minutes=60))

    # generate hourly graph
    # create_volume_graph(prices, percentiles, timestamps, token, left_offset=1, view_date_start='2024-03-02 00:00:00')
    create_volume_graph(prices, percentiles, timestamps, token, left_offset=1, view_date_start=None)


@dataclass
class Volume:
    buy: float
    sell: float
    percent_buy_sell: float
    total_volume: float


def generate_volume(balances_rows, cursor, token_address, granularity=datetime.timedelta(minutes=60)):
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
    wallet_percentiles, _ = generate_wallet_percentiles(cursor, percentiles, token_address)

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
        i, to_process_rows = get_next_rows(i=i, table_rows=balances_rows,
                                           timestamp_column_num=BalancesColumns.timestamp,
                                           before_timestamp=current_hour + granularity)

        # process changes to wallets map
        volume_totals = process_balances_changes(wallet_balances, to_process_rows)

        # update percentiles
        update_balance_percentiles(percentiles, volume_totals, wallet_percentiles)
        timestamps.append(str(current_hour))

        current_hour += granularity

    print(f'completed generate_percentiles: {time.time() - start}')
    return timestamps, percentiles


def build_wallets(wallets, balances_rows):
    # wallet_address: [balance, total_cost_basis, remaining_cost_basis, realized_gains]
    for row in balances_rows:
        wallet_address, token_balance, total_cost_basis, remaining_cost_basis, realized_gains, _ = row
        wallets[wallet_address] = [token_balance, total_cost_basis, remaining_cost_basis, realized_gains]

    return wallets


# get all diffs from balances_rows and update the wallets
def process_balances_changes(wallet_balances, to_process_rows):
    # wallet_address: volume
    volume_totals = {}

    for row in to_process_rows:
        wallet_address = row[BalancesColumns.wallet_address]
        if wallet_address not in wallet_balances:
            wallet_balances[wallet_address] = 0

        new_balance = row[BalancesColumns.balance]

        diff = new_balance - wallet_balances[wallet_address]
        update_volume(volume_totals, wallet_address, diff)

        wallet_balances[wallet_address] = new_balance

    return volume_totals


def update_volume(volume_totals, wallet_address, value):
    if wallet_address not in volume_totals:
        volume_totals[wallet_address] = Volume(0, 0, 0, 0)

    if value > 0:
        volume_totals[wallet_address].buy += value
    else:
        volume_totals[wallet_address].sell += value


# add volume from wallet_diffs to percentiles
def update_balance_percentiles(percentiles, volume_totals, wallet_percentiles):
    for percentile in percentiles:
        percentiles[percentile].append(Volume(0, 0, 0, 0))

    for wallet_address, volume in volume_totals.items():
        percentiles[wallet_percentiles[wallet_address]][-1].buy += volume.buy
        percentiles[wallet_percentiles[wallet_address]][-1].sell += volume.sell

    # calculate percent_buy_sell and total_volume
    for percentile in percentiles:
        volume = percentiles[percentile][-1]
        volume.total_volume = volume.buy - volume.sell  # sell is negative
        volume.percent_buy_sell = 0 if volume.total_volume == 0 \
            else (volume.buy + volume.sell) / volume.total_volume * 100


def create_volume_graph(prices, percentiles, timestamps, token, left_offset=0, alt_title=None, view_date_start=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        specs=[[{"secondary_y": True}], [{"secondary_y": True}]], )
    fig.update_layout(
        title=dict(text=f'{token.name} percent buy/sell and volume' if alt_title is None else alt_title,
                   font=dict(size=25))
    )

    price_left_offset = 10
    if view_date_start is not None:
        i = 0
        while i < len(prices):
            if prices[i][1] > view_date_start:
                break
            i += 1
        price_left_offset = i
        i = 0
        while i < len(prices):
            if timestamps[i] > view_date_start:
                break
            i += 1
        left_offset = i

    for percentile in percentiles.keys():
        buy_sell_percentage_data = [element.percent_buy_sell for element in percentiles[percentile]]
        total_volume = [element.total_volume for element in percentiles[percentile]]
        fig.add_trace(
            go.Scatter(x=timestamps[left_offset:],
                       y=buy_sell_percentage_data[left_offset:],
                       name=f'{percentile} buy/sell percent'),
            secondary_y=False,
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=timestamps[left_offset:], y=total_volume[left_offset:], name=f'{percentile} total_volume'),
            secondary_y=False,
            row=2, col=1,
        )

    add_price_trace(prices, fig, left_offset=price_left_offset, row=1)
    add_price_trace(prices, fig, left_offset=price_left_offset, row=2)

    fig.update_layout(legend_title_text='percentiles')
    # Set y-axes titles
    fig.update_yaxes(type="log", row=2, secondary_y=False)
    fig.update_yaxes(title_text='percent buy/sell', row=1, secondary_y=False)
    fig.update_yaxes(title_text='token volume', row=2, secondary_y=False)
    fig.update_yaxes(title_text="price", showspikes=True, secondary_y=True)

    fig.update_layout(hovermode="x unified", xaxis_showticklabels=True)
    fig.update_traces(xaxis="x1")
    fig.update_xaxes(spikemode='across+marker')
    fig.show()


if __name__ == "__main__":
    run_volume()
