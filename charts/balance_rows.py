import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from database import *
import time

filter_out_addresses = [
    '0x3aF2ACB662A241da4ef4310C7AB226f552B42115', # Altlayer Airdrop Safe Smart Account
    '0x9c7F0628ceE619953Fc395Cd7cF0576DCe1F505E', # PepeFork Airdrop contract
]


def create_balances_and_price_graph(prices, balances_map, balances_column, token):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        title=dict(text=token, font=dict(size=30))
    )

    for address in balances_map.keys():
        # address: [realized_gains, timestamps]
        fig.add_trace(
            go.Scatter(x=balances_map[address][1], y=balances_map[address][0], name=address),
            secondary_y=False,
        )

    add_price_trace(prices, fig)

    # Set y-axes titles
    fig.update_yaxes(title_text=balances_column, secondary_y=False)
    fig.update_yaxes(title_text="price", secondary_y=True)
    fig.show()


def add_price_trace(prices, fig):
    # 0 price, 1 timestamp
    prices_y = [row[0] for row in prices]
    prices_timestamp = [row[1] for row in prices]

    fig.add_trace(
        go.Scatter(x=prices_timestamp, y=prices_y, name="price"),
        secondary_y=True,
    )


def compute_balances_map(balances_rows, balances_column):
    # 0 wallet_address, 1 timestamp, 2 balance, 3 total_cost_basis, 4 remaining_cost_basis, 5 realized_gains
    balances_map = {}
    for row in balances_rows:
        if row[0] not in balances_map:
            balances_map[row[0]] = [[], []]
        column_value = ""
        if balances_column == 'realized_gains':
            column_value = row[5]
        elif balances_column == 'remaining_cost_basis':
            column_value = row[4]
        elif balances_column == 'total_cost_basis':
            column_value = row[3]
        elif balances_column == 'balance':
            column_value = row[2]
        balances_map[row[0]][0].append(column_value)
        balances_map[row[0]][1].append(row[1])  # timestamp

    return balances_map


def load_prices(cursor, token_address):
    start_time = time.time()
    query = f"""
        SELECT 
            price,
            timestamp
        FROM 
            prices
        WHERE
            token_address='{token_address}'
        ORDER BY
            timestamp;
        """
    cursor.execute(query)
    prices = cursor.fetchall()
    print("Total prices rows are:  ", len(prices))
    print(f'load_prices time: {time.time() - start_time}')
    return prices


def get_largest_balances(cursor, token_address, balances_column, after_timestamp=None):
    largest_wallets = get_largest_column_wallets(cursor, token_address, balances_column, after_timestamp)
    return load_balances(cursor, token_address, largest_wallets)


def get_largest_column_wallets(cursor, token_address, balances_column, after_timestamp=None):
    start_time = time.time()
    after_timestamp_string = '' if after_timestamp is None else f"AND timestamp >= '{after_timestamp}'"
    find_largest_wallets_query = f"""
        SELECT * 
        FROM (
            SELECT 
              wallet_address, 
              timestamp, 
              balance, 
              total_cost_basis, 
              remaining_cost_basis, 
              realized_gains, 
              ROW_NUMBER() OVER (
                PARTITION BY wallet_address 
                ORDER BY 
                  {balances_column} DESC
              ) AS row_num 
            FROM 
              balances 
            WHERE 
              token_address = '{token_address}'
              AND wallet_address NOT IN ({to_sql_collection_string(filter_out_addresses)})
              {after_timestamp_string}
        ) AS ranked 
        WHERE 
            row_num = 1 
        ORDER BY 
          {balances_column} DESC
        LIMIT 15;
        """
    cursor.execute(find_largest_wallets_query)
    largest_wallets_rows = cursor.fetchall()
    print(f'get_largest_realized_gains_wallets time: {time.time() - start_time}')
    return [row[0] for row in largest_wallets_rows]


def load_balances(cursor, token_address, largest_wallets):
    start_time = time.time()
    wallet_addresses = to_sql_collection_string(largest_wallets)
    load_wallets_balances_query = f"""
        SELECT 
            wallet_address, 
            timestamp,
            balance, 
            total_cost_basis,
            remaining_cost_basis,
            realized_gains
        FROM 
            balances
        WHERE 
            wallet_address IN ({wallet_addresses})
            AND token_address='{token_address}'
        ORDER BY
            block_number;
        """
    cursor.execute(load_wallets_balances_query)
    balances_rows = cursor.fetchall()
    print(f'load_wallets_balances time: {time.time() - start_time}')
    print("Total get_largest_balances rows:  ", len(balances_rows))
    return balances_rows


def to_sql_collection_string(items):
    result = [f"'{item.lower()}',\n" for item in items]
    result[-1] = result[-1][:-2]  # remove last \n and , from last wallet_address
    return " ".join(result)