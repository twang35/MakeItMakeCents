import requests
from pprint import pprint
from database import *

# https://app.transpose.io/account/keys contains key as of Feb 5, 2024
with open('transpose_key.txt', 'r') as f:
    transpose_key = f.read()


def main():
    get_prices()


def get_prices(token_address='0x8457CA5040ad67fdebbCC8EdCE889A335Bc0fbFB', start='2024-01-24 21:46:00', end='2024-01-25 09:15:36'):
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
    print('Credits charged:', response.headers.get('X-Credits-Charged', None))
    return prices


main()
