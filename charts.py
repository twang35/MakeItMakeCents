# Using graph_objects
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

from charts.balance_rows import *
from database import *
import time


def charts():
    token = 'pepefork'
    # token = 'altlayer'
    # 'balance', 'remaining_cost_basis', 'realized_gains'
    balances_column = 'balance'

    conn = create_connection()
    cursor = conn.cursor()

    token_address = token_addresses[token]
    balances_rows = get_largest_balances(cursor, token_address, balances_column, after_timestamp='2024-02-11 10:00:00')
    balances_map = compute_balances_map(balances_rows, balances_column)
    prices = load_prices(cursor, token_address)

    create_balances_and_price_graph(prices, balances_map, balances_column, token)


charts()
