from dataclasses import dataclass

from plotly.subplots import make_subplots
from charts.shared_charts import *
from database import *


# get buy and sell activity for all users
# then split by all-time high balance percentile
# then add exchange addresses as a category for balance percentile
def volume():
    print('volume')
    token = pepefork

    conn = create_connection()
    cursor = conn.cursor()

    # load balances
    balances_rows = load_balances_table(cursor, token.address)
    prices = load_prices(cursor, token.address)

    # calculate sum of diffs and add to chart output
    timestamps, percentages = generate_volume(balances_rows)

    # generate hourly graph
    create_volume_graph(prices, percentages, timestamps, token)


def generate_volume(balances_rows):
    print(f'running generate_volume')

    start = time.time()
    # wallet_address: [balance, total_cost_basis, remaining_cost_basis, realized_gains, unrealized_gains]
    wallets = {}
    timestamps = []
    percentiles = {  # balances are bucketed down
        # 0.1: [],
        # 0.5: [],
        # 1: [],
        # 5: [],
        # 10: [],
        # 25: [],
        # 50: [],
        100: [],
    }

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
        wallet_diffs = process_balances_changes(wallets, to_process_rows)

        # update percentiles
        update_balance_percentiles(percentiles, wallet_diffs)
        timestamps.append(str(current_hour))

        current_hour += datetime.timedelta(minutes=60)

    print(f'completed generate_percentiles: {time.time() - start}')
    return timestamps, percentiles


def create_volume_graph(prices, percentages, timestamps, token):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        title=dict(text=f'{token.name} volume', font=dict(size=25))
    )

    for percentage in percentages.keys():
        fig.add_trace(
            go.Scatter(x=timestamps, y=percentages[percentage], name=percentage),
            secondary_y=False,
        )
    # fig.update_yaxes(type="log")

    add_price_trace(prices, fig)

    fig.update_layout(legend_title_text='percentiles')
    # Set y-axes titles
    fig.update_yaxes(title_text='token amount', secondary_y=False)
    fig.update_yaxes(title_text="price", showspikes=True, secondary_y=True)
    fig.update_layout(hovermode="x unified")
    fig.show()


if __name__ == "__main__":
    volume()
