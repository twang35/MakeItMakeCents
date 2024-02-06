import requests
from pprint import pprint
from database import *
import datetime

# https://app.transpose.io/account/keys contains key as of Feb 5, 2024
# https://app.transpose.io/playground to play with SQL queries
with open('transpose_key.txt', 'r') as f:
    transpose_key = f.read()


def main():
    # sql query can only return 3000 rows at a time (50 hrs)
    # add 48 hr blocks to queue and query for 48 hrs at a time
    prices = get_prices(
        token_address='0x8457CA5040ad67fdebbCC8EdCE889A335Bc0fbFB',
        # start='2024-01-24 21:46:00', end='2024-02-15 09:15:36')
        start='2024-01-27 20:26:00', end='2024-02-15 09:15:36')
    process_prices(prices)


def process_prices(prices):
    conn = create_connection()
    i = 0

    while i < len(prices):
        if i % 10 == 0:
            print(f'prices remaining: {len(prices) - i}')
        price = prices[i]
        # token_address, timestamp, token_symbol, price, volume
        row = (price['token_address'],
               datetime.datetime.fromisoformat(price['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
               price['token_symbol'], price['average_price'], price['volume'])
        insert_price(conn, row)
        i += 1

    print('completed processing prices successfully')


def get_prices(token_address, start, end):
    url = "https://api.transpose.io/sql"
    sql_query = f"""
    SELECT *
    FROM ethereum.token_prices_ohlc_1m
    WHERE token_address = '{token_address}'
        AND timestamp BETWEEN '{start}' AND '{end}'
    ORDER BY timestamp;
    """

    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': transpose_key,
    }
    response = requests.post(url,
        headers=headers,
        json={
            'sql': sql_query,
        },
    )

    prices = response.json()['results']
    print('Transpose credits charged:', response.headers.get('X-Credits-Charged', None))
    return prices


main()
