import time
import timeit
from time import sleep
import json

from eth_typing import BlockNumber
from web3 import Web3, HTTPProvider, AsyncWeb3
from database import *
import datetime
from eth_abi import decode
from concurrent.futures import ThreadPoolExecutor, wait


def main():
    search_term = 'binance'

    # create_known_addresses_table()
    conn = create_connection()

    # time a write
    timestamp = datetime.datetime.utcnow()
    epoch_time = int(timestamp.timestamp())
    timestamp_string = timestamp.strftime('%Y-%m-%d %H:%M:%S')

    row = (2, timestamp_string, epoch_time)

    start = time.time()
    insert_block_time(conn, row)
    print(f'write time: {time.time() - start}')

    # time a read
    cursor = conn.cursor()
    query = f"""
        SELECT * 
        FROM block_times
        WHERE block_number = 2;
        """
    start = time.time()
    cursor.execute(query)
    result = cursor.fetchall()
    print(f'read time: {time.time() - start}')
    print(result)
    print('done')


if __name__ == "__main__":
    main()
