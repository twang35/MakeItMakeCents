import time
import plotly.graph_objects as go

from database import *


def add_price_trace(prices, fig):
    # 0 price, 1 timestamp
    prices_y = [row[0] for row in prices]
    prices_timestamp = [row[1] for row in prices]

    fig.add_trace(
        go.Scatter(x=prices_timestamp, y=prices_y, name="price", line=dict(color='black')),
        secondary_y=True,
    )


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


def get_price_map(cursor, token_address):
    query = f"""
            SELECT * FROM prices
            where token_address='{token_address}'
            ORDER by timestamp;
            """
    cursor.execute(query)
    prices_rows = cursor.fetchall()
    print("Total prices rows are: ", len(prices_rows))
    return to_prices_map(prices_rows)


def to_prices_map(prices_rows):
    time_to_price = {}
    for row in prices_rows:
        time_to_price[row[PricesColumns.timestamp]] = row[PricesColumns.price]

    cur_time = datetime.datetime.fromisoformat(prices_rows[0][PricesColumns.timestamp])
    # some charts grab the price at the end of the hour. If the end of the hour is in the future, there is no price.
    # Adding extra 1 day buffer in case of day granularity for future charts.
    last_time = datetime.datetime.utcnow() + datetime.timedelta(days=1)

    last_price = time_to_price[str(cur_time)]

    while cur_time < last_time:
        # replace missing times with last price
        if str(cur_time) not in time_to_price:
            time_to_price[str(cur_time)] = last_price
        else:
            last_price = time_to_price[str(cur_time)]
        cur_time += datetime.timedelta(seconds=60)

    # time_to_price map, first timestamp that has price data
    return time_to_price, prices_rows[0][PricesColumns.timestamp]


def get_price(time_to_price, first_price_timestamp, timestamp):
    return time_to_price[timestamp] if timestamp >= first_price_timestamp else 0
