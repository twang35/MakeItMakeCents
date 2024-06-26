import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

from charts.balance_rows import *
from charts.unrealized_gains import *
from database import *


def charts():
    # manual settings #################
    token = pepefork
    # token = altlayer
    # token = xcad
    # token = mubi

    # balance graph settings
    # 'balance', 'remaining_cost_basis', 'realized_gains'
    balances_column = 'balance'

    # unrealized gains graph settings
    chart_type = urg_percent_by_holdings
    # chart_type = holdings_by_urg_percent

    # generate graphs #################
    conn = create_connection()
    cursor = conn.cursor()

    prices = load_prices(cursor, token.address)

    # create_balances_graph(token, balances_column, prices, cursor)
    # create_balances_by_percentile_graph(token, balances_column, prices, cursor)
    create_urg_chart(token, chart_type, prices, cursor)


# displays largest 15 wallets for the selected balances_column from the balances table
def create_balances_graph(token, balances_column, prices, cursor):
    largest_balances_rows = get_largest_balances(cursor, token.address, balances_column)
    balances_map = compute_balances_map(largest_balances_rows, balances_column)

    create_balances_and_price_graph(prices, balances_map, balances_column, token, left_offset=4)


# displays largest 15 wallets for the selected balances_column from the balances table
def create_balances_by_percentile_graph(token, balances_column, prices, cursor):
    percentile = 0.1
    addresses = get_percentile_addresses(cursor, token.address, percentile)
    largest_balances_rows = get_largest_balances_for_addresses(cursor, token.address, balances_column, addresses)
    balances_map = compute_balances_map(largest_balances_rows, balances_column)

    create_balances_and_price_graph(prices, balances_map, balances_column, token, left_offset=4,
                                    alt_title=f'{token.name} {balances_column} {percentile} top addresses')


# display unrealized gains data depending on the chart_type, ie urg_percent_by_holdings shows how much urg percent for
# each percentile of token balances (top 0.1%, top 1%, top 10%, etc.)
def create_urg_chart(token, chart_type, prices, cursor):
    # load balances
    balances_rows = load_balances_table(cursor, token.address)

    # load prices
    time_to_price, first_price_timestamp = get_price_map(cursor, token.address)

    # calculate URG
    timestamps, percentages = generate_percentiles(chart_type, balances_rows, time_to_price, first_price_timestamp)

    # generate hourly graph
    create_unrealized_gains_graph(prices, percentages, timestamps, token, chart_type)


if __name__ == "__main__":
    charts()
