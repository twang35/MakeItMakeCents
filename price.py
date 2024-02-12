import requests
from pprint import pprint
import time
from database import *
import datetime

# https://app.transpose.io/account/keys contains key as of Feb 5, 2024
# https://app.transpose.io/playground to play with SQL queries
with open('transpose_key.txt', 'r') as f:
    transpose_key = f.read()

# sql query can only return 3000 rows at a time (50 hrs)
# add 48 hr blocks to queue (2880 rows)
batch_price_date_increment = datetime.timedelta(days=2)


def price():
    conn = create_connection()
    # add_items_to_queue(
    #     start=datetime.datetime.fromisoformat('2024-02-03 20:26:00'),
    #     end=datetime.datetime.utcnow(),
    #     increment=batch_price_date_increment
    # )
    prices = get_prices(
        token_address='0x8457CA5040ad67fdebbCC8EdCE889A335Bc0fbFB',
        # start='2024-01-24 21:46:00', end='2024-02-15 09:15:36')
        start=datetime.datetime.fromisoformat('2024-02-03 20:26:00'),
        end=datetime.datetime.fromisoformat('2024-02-03 21:15:36'))
    # process_prices(prices, conn)


def process_price_dates(token_address, queue_name):
    q = attach_queue(queue_name)
    conn = create_connection()
    i = 0
    start = time.time()
    print_interval = 1

    # get date and query
    while q.qsize() > 0:
        if i % print_interval == 0:
            end = time.time()
            q_size = q.qsize()
            velocity = print_interval / (end - start)
            estimate = str(datetime.timedelta(seconds=q_size / velocity))
            print(f'queue size: {q_size}, velocity: {"%.4f" % velocity} elements/second,'
                  f' completion estimate: {estimate}')
            start = time.time()
        i += 1

        item = q.get()
        prices = get_prices(token_address, item, item + batch_price_date_increment)
        process_prices(prices, conn)

        q.ack(item)


def process_prices(prices, conn):
    i = 0

    while i < len(prices):
        price = prices[i]
        # token_address, timestamp, token_symbol, price, volume
        row = (price['token_address'],
               datetime.datetime.fromisoformat(price['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
               price['token_symbol'], price['average_price'], price['volume'])
        insert_price(conn, row)
        i += 1


def get_prices(token_address, start, end):
    url = "https://api.transpose.io/sql"
    sql_query = f"""
    SELECT token_address, timestamp, token_symbol, average_price, volume
    FROM ethereum.token_prices_ohlc_1m
    WHERE token_address = '{token_address}'
        AND timestamp BETWEEN '{start}' AND '{end}'
    ORDER BY timestamp;
    """

    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': transpose_key,
    }
    response = requests.post(url, headers=headers, json={'sql': sql_query})

    if response.status_code != 200:
        # retry once
        response = requests.post(url, headers=headers, json={'sql': sql_query})

    prices = response.json()['results']
    print('Transpose credits charged:', response.headers.get('X-Credits-Charged', None))
    return prices


if __name__ == "__main__":
    price()
