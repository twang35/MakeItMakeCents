import time
from pprint import pprint
import datetime
from database import *


def main():
    conn = create_connection()
    cursor = conn.cursor()

    # compute_balances(conn)

    query = """
        SELECT 
            DISTINCT b.wallet_address, 
            b.balance, 
            b.timestamp
        FROM 
            balances b
        WHERE
            b.timestamp = (
                SELECT MAX(timestamp)
                FROM balances
                WHERE timestamp <= '2024-02-08 13:46:00'
                AND wallet_address = b.wallet_address
            )
        ORDER BY 
            b.balance DESC
        LIMIT 300;
        """
    cursor.execute(query)
    latest_balances = cursor.fetchall()
    print("Total unique balances rows are:  ", len(latest_balances))
    for row in latest_balances:
        print(row)


def compute_balances(conn):
    cursor = conn.cursor()

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

    # compute balances table

    wallet_to_balance = {}  # wallet_address: [balance, average_cost_basis, realized_gains, unrealized_gains]
    null_address = '0x0000000000000000000000000000000000000000'

    i = 0
    # 0 block_number, 1 transaction_index, 2 log_index, 3 sender, 4 recipient, 5 token_id, 6 value
    for txn in txns:
        if i % 10000 == 0:
            print(f'remaining transactions to process: {len(txns) - i}')
        i += 1

        sender = txn[3]
        recipient = txn[4]
        value = txn[6]

        if value == 0:
            continue

        if sender == null_address:
            # do not subtract
            print(f'null address sender: {value}')
        else:
            wallet_to_balance[sender] -= value

        if recipient == null_address:
            # do not subtract
            print(f'null address recipient: {value}')
        else:
            if recipient not in wallet_to_balance:
                wallet_to_balance[recipient] = 0
            wallet_to_balance[recipient] += value

        if recipient != null_address:
            row = (recipient, '_', block_times[txn[0]], wallet_to_balance[recipient], 0, 0, 0)
            insert_balance(conn, row)
        if sender != null_address:
            row = (sender, '_', block_times[txn[0]],  wallet_to_balance[sender], 0, 0, 0)
            insert_balance(conn, row)

        #   PRIMARY(wallet_address, token_address, timestamp),
        #    balance, average_cost_basis, realized_gains, unrealized_gains


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
