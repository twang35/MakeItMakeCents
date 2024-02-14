# Using graph_objects
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from database import *
import time


def charts():
    conn = create_connection()

    balances = load_balances(conn, altlayer_token_address)
    balances_map = compute_balances_map(balances)
    prices = load_prices(conn, altlayer_token_address)

    # 0 price, 1 timestamp
    prices_y = [row[0] for row in prices]
    prices_timestamp = [row[1] for row in prices]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for address in balances_map.keys():
        # address: [realized_gains, timestamps]
        fig.add_trace(
            go.Scatter(x=balances_map[address][1], y=balances_map[address][0], name=address),
            secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(x=prices_timestamp, y=prices_y, name="price"),
        secondary_y=True,
    )

    # Set y-axes titles
    fig.update_yaxes(title_text="realized gains", secondary_y=False)
    fig.update_yaxes(title_text="price", secondary_y=True)
    fig.show()


def load_balances(conn, token_address):
    cursor = conn.cursor()

    start_time = time.time()
    query = f"""
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
            wallet_address in ('0xbd513f67ed9bccc8364176e8b97bb93a5030e777',
                                '0x28c6c06298d514db089934071355e5743bf21d60',
                                '0xdfd5293d8e347dfe59e90efd55b2956a1343963d',
                                '0x21a31ee1afc51d94c2efccaa2092ad1028285549',
                                '0x0d4d42969105e5a144a2538aeb24a2e16be40e34',
                                '0xc882b111a75c0c657fc507c04fbfcd2cc984f071',
                                '0xecd3b4b7fd0bc9f52335f29a3e007453de07d2b9',
                                '0x58edf78281334335effa23101bbe3371b6a36a51',
                                '0x0d0707963952f2fba59dd06f2b425ace40b492fe',
                                '0xfcc43eb62b4da4bc5d2260721a497c4693e59a70')
            AND token_address='{token_address}'
        ORDER BY
            timestamp;
        """
    cursor.execute(query)
    balances = cursor.fetchall()
    print("Total balances rows are:  ", len(balances))
    print(f'query time: {time.time() - start_time}')
    return balances


def compute_balances_map(balances):
    # 0 wallet_address, 1 timestamp, 2 balance, 3 total_cost_basis, 4 remaining_cost_basis, 5 realized_gains
    balances_map = {}
    for row in balances:
        if row[0] not in balances_map:
            balances_map[row[0]] = [[], []]
        balances_map[row[0]][0].append(row[5])  # realized_gains
        balances_map[row[0]][1].append(row[1])  # timestamp

    return balances_map


def load_prices(conn, token_address):
    cursor = conn.cursor()

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
    print(f'query time: {time.time() - start_time}')
    return prices


charts()
