import time
from pprint import pprint
import datetime
from database import *


def main():
    conn = create_connection()
    cursor = conn.cursor()

    # read first txn
    # 0 block_number, 1 transaction_index, 2 log_index, 3 sender, 4 recipient, 5 token_id, 6 value
    # use limit 1000 and https://stackoverflow.com/questions/14468586
    query = """
        SELECT * FROM transactions
        ORDER BY block_number, log_index;
        """
    cursor.execute(query)
    txns = cursor.fetchall()
    print("Total transactions rows are:  ", len(txns))

    query = """
        SELECT * FROM block_times;
        """
    cursor.execute(query)
    block_times_rows = cursor.fetchall()
    print("Total block_times rows are:  ", len(block_times_rows))
    block_times = to_block_times_map(block_times_rows)

    query = """
        SELECT * FROM prices
        where token_address='0x8457CA5040ad67fdebbCC8EdCE889A335Bc0fbFB';
        """
    cursor.execute(query)
    prices_rows = cursor.fetchall()
    print("Total prices rows are:  ", len(prices_rows))
    time_to_price, first_price_time_str = to_prices_map(prices_rows)

    # write first row to balances
    #   PRIMARY(wallet_address, token_address, epoch_seconds),
    #   timestamp, balance, average_cost_basis, realized_gains, unrealized_gains


def to_block_times_map(block_times_rows):
    output = {}
    # block_number, timestamp, epoch_seconds
    for row in block_times_rows:
        # block_num -> timestamp -> round down to nearest minute
        output[row[0]] = row[1][:-2] + '00'

    return output


def to_prices_map(prices_rows):
    time_to_price = {}
    # 0 token_address, 1 timestamp, 2 token_symbol, 3 price, 4 volume
    for row in prices_rows:
        time_to_price[row[1]] = row[3]

    # replace missing times with last price  #######################
    cur_time = datetime.datetime.fromisoformat(prices_rows[0][1])
    last_time = datetime.datetime.fromisoformat(prices_rows[-1][1])

    last_price = time_to_price[str(cur_time)]

    while cur_time < last_time:
        if str(cur_time) not in time_to_price:
            time_to_price[str(cur_time)] = last_price
        else:
            last_price = time_to_price[str(cur_time)]
        cur_time += datetime.timedelta(seconds=60)
    return time_to_price, prices_rows[0][1]


main()
