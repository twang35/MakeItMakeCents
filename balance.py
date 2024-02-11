import time
from pprint import pprint
import datetime
from database import *

null_address = '0x0000000000000000000000000000000000000000'
smallest_balance = 1e-12


def main():
    conn = create_connection()

    # compute_balances(conn)

    balances = get_balances_before(conn, '2024-02-04 13:00:00')
    for i in range(100):
        print(balances[i])


def get_balances_before(conn, timestamp):
    cursor = conn.cursor()

    start_time = time.time()
    query = f"""
        SELECT 
            *
        FROM (
            SELECT 
                wallet_address, 
                token_address,
                timestamp,
                balance, 
                total_cost_basis,
                remaining_cost_basis,
                realized_gains,
                ROW_NUMBER() OVER (PARTITION BY wallet_address ORDER BY timestamp DESC) AS row_num
            FROM 
                balances
            WHERE 
                timestamp <= '{timestamp}'
        ) AS ranked
        WHERE 
            row_num = 1
        ORDER BY 
            realized_gains DESC;
        """
    cursor.execute(query)
    balances = cursor.fetchall()
    print("Total unique balances rows are:  ", len(balances))
    print(f'query time: {time.time() - start_time}')
    return balances


def compute_balances(conn):
    cursor = conn.cursor()
    txns, block_times, time_to_price, first_price_timestamp = load_data(cursor)

    # wallet_address: [balance, total_cost_basis, remaining_cost_basis, realized_gains]
    wallets = {}

    i = 0
    for txn in txns:
        if i % 10000 == 0:
            print(f'remaining transactions to process: {len(txns) - i}')
        i += 1

        # 0 block_number, 1 transaction_index, 2 log_index, 3 sender, 4 recipient, 5 token_id, 6 value
        block, _, _, sender, recipient, token_id, value = txn
        price = time_to_price[block_times[block]] if block_times[block] >= first_price_timestamp else 0

        if value < smallest_balance:
            continue

        update_sender(wallets, sender, value, price)
        update_recipient(wallets, recipient, value, price)

        # row: wallet_address, token_address, timestamp, balance, total_cost_basis, remaining_cost_basis, realized_gains
        if sender != null_address:
            wallet = wallets[sender]
            row = (sender, token_id, block_times[block], wallet[0], wallet[1], wallet[2], wallet[3])
            insert_balance(conn, row)
        if recipient != null_address:
            wallet = wallets[recipient]
            row = (recipient, token_id, block_times[block], wallet[0], wallet[1], wallet[2], wallet[3])
            insert_balance(conn, row)


def update_sender(wallets, sender, value, price):
    if sender == null_address:
        # do not subtract
        print(f'null address generated: {value}')
        return

    # wallet_address: [balance, total_cost_basis, remaining_cost_basis, realized_gains]

    # update remaining cost basis
    cost_sent = value / wallets[sender][0] * wallets[sender][2]
    wallets[sender][2] -= cost_sent
    # update balance
    wallets[sender][0] -= value
    # update realized gain
    wallets[sender][3] += price * value - cost_sent


def update_recipient(wallets, recipient, value, price):
    if recipient == null_address:
        # do not subtract
        print(f'null address burned: {value}')
        return

    # wallet_address: [balance, total_cost_basis, remaining_cost_basis, realized_gains]
    if recipient not in wallets:
        wallets[recipient] = [0, 0, 0, 0]

    # update balance
    wallets[recipient][0] += value
    # update total cost basis
    wallets[recipient][1] += price * value
    # update remaining cost basis
    wallets[recipient][2] += price * value


def load_data(cursor):
    query = """
        SELECT * FROM transactions
        WHERE block_number < 19171614
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
        where token_address='0x8457CA5040ad67fdebbCC8EdCE889A335Bc0fbFB'
        ORDER by timestamp;
        """
    cursor.execute(query)
    prices_rows = cursor.fetchall()
    print("Total prices rows are:  ", len(prices_rows))
    time_to_price, first_price_timestamp = to_prices_map(prices_rows)

    return txns, block_times, time_to_price, first_price_timestamp


def to_block_times_map(block_times_rows):
    output = {}
    # block_number, timestamp, epoch_seconds
    for row in block_times_rows:
        # block_num -> timestamp -> round down to nearest minute
        # rounds seconds down: '2024-02-01 13:34:12' -> '2024-02-01 13:34:00'
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
