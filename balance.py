import time
from pprint import pprint
import datetime
from database import *

null_address = '0x0000000000000000000000000000000000000000'
smallest_balance = 1e-12


def run_balance():
    conn = create_connection()

    compute_balances(conn, token_addresses['altlayer'])

    # balances = get_balances_before(conn, '2024-02-14 13:00:00', altlayer_token_address)
    balances = get_balances_before(conn, datetime.datetime.utcnow(), token_addresses['altlayer'])
    for i in range(100):
        print(balances[i])


def get_balances_before(conn, timestamp, token_address):
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
                ROW_NUMBER() OVER (PARTITION BY wallet_address ORDER BY block_number DESC) AS row_num
            FROM 
                balances
            WHERE 
                timestamp <= '{timestamp}'
                AND token_address = '{token_address}'
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


def compute_balances(conn, token_address):
    # compute from the latest block in the balances table or start computing from the earliest txn
    latest_block = get_latest_balances_block(conn, token_address)
    latest_block = 0 if latest_block is None else latest_block

    cursor = conn.cursor()
    txns, block_times, time_to_price, first_price_timestamp = load_data(cursor=cursor,
                                                                        token_address=token_address,
                                                                        latest_block=latest_block)

    # wallet_address: [balance, total_cost_basis, remaining_cost_basis, realized_gains]
    wallets = {}
    if latest_block != 0:
        # load the existing wallets from the balances table
        wallets = build_wallets(get_latest_wallet_balances(conn, token_address))

    start = time.time()
    i = 0
    print_interval = 10000
    for txn in txns:
        if i % print_interval == 0:
            end = time.time()
            velocity = print_interval / (end - start)
            estimate = str(datetime.timedelta(seconds=(len(txns) - i) / velocity))
            print(f'remaining transactions to process: {len(txns) - i}, velocity: {"%.4f" % velocity} elements/second,'
                  f' completion estimate: {estimate}')
            start = time.time()
        i += 1

        # 0 block_number, 1 transaction_index, 2 log_index, 3 timestamp, 4 sender, 5 recipient, 6 token_id, 7 value
        block, _, _, _, sender, recipient, token_id, value = txn
        price = get_price(time_to_price, first_price_timestamp, block_times[block])

        if value < smallest_balance or (wallets[sender][0] == 0 and value < 1):
            # don't count if value is too small
            # or if wallet has 0 balance due to SQLite REAL number precision being too low
            continue

        update_sender(wallets, sender, value, price, txn)
        update_recipient(wallets, recipient, value, price)

        # row: wallet_address, token_address, timestamp, block, balance, total_cost_basis, remaining_cost_basis,
        #   realized_gains
        if sender != null_address:
            balance, total_cost_basis, remaining_cost_basis, realized_gains = wallets[sender]
            row = (sender, token_id, block_times[block], block,
                   balance, total_cost_basis, remaining_cost_basis, realized_gains)
            insert_balance(conn, row)
        if recipient != null_address:
            balance, total_cost_basis, remaining_cost_basis, realized_gains = wallets[recipient]
            row = (recipient, token_id, block_times[block], block,
                   balance, total_cost_basis, remaining_cost_basis, realized_gains)
            insert_balance(conn, row)


def build_wallets(balances_rows):
    # wallet_address: [balance, total_cost_basis, remaining_cost_basis, realized_gains]
    wallets = {}
    for row in balances_rows:
        wallet_address, token_balance, total_cost_basis, remaining_cost_basis, realized_gains, _ = row
        wallets[wallet_address] = [token_balance, total_cost_basis, remaining_cost_basis, realized_gains]

    return wallets


def update_sender(wallets, sender, value, price, txn):
    if sender == null_address:
        # do not subtract
        print(f'null address generated: {value}')
        return

    # wallet_address: [balance, total_cost_basis, remaining_cost_basis, realized_gains]

    try:
        cost_sent = value / wallets[sender][0] * wallets[sender][2]
        # update remaining cost basis
        wallets[sender][2] -= cost_sent
        # update balance
        wallets[sender][0] -= value
        # update realized gain
        wallets[sender][3] += price * value - cost_sent
    except:
        print(f'\ncaught an exception:\n'
              f'txn row: block_number, transaction_index, log_index, timestamp, sender, recipient, token_address, value'
              f'\n{txn}\n')
        raise


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


def load_data(cursor, token_address, latest_block):
    print(f"Loading for token_address {token_address} after block {latest_block}")

    start = time.time()
    query = f"""
        SELECT * FROM transactions
        WHERE block_number > {latest_block} and token_address = '{token_address}'
        ORDER BY block_number, log_index;
        """
    cursor.execute(query)
    txns = cursor.fetchall()
    print("Total transactions rows are: ", len(txns))

    query = """
        SELECT * FROM block_times;
        """
    cursor.execute(query)
    block_times_rows = cursor.fetchall()
    print("Total block_times rows are: ", len(block_times_rows))
    block_times = to_block_times_map(block_times_rows)

    time_to_price, first_price_timestamp = get_price_map(cursor, token_address)

    print(f'load_data time: {time.time() - start}s')
    return txns, block_times, time_to_price, first_price_timestamp


def to_block_times_map(block_times_rows):
    output = {}
    # block_number, timestamp, epoch_seconds
    for row in block_times_rows:
        # block_num -> timestamp -> round down to nearest minute
        # rounds seconds down: '2024-02-01 13:34:12' -> '2024-02-01 13:34:00'
        output[row[0]] = row[1][:-2] + '00'

    return output


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
    # 0 token_address, 1 timestamp, 2 token_symbol, 3 price, 4 volume
    for row in prices_rows:
        time_to_price[row[1]] = row[3]

    cur_time = datetime.datetime.fromisoformat(prices_rows[0][1])
    last_time = datetime.datetime.utcnow()

    last_price = time_to_price[str(cur_time)]

    while cur_time < last_time:
        # replace missing times with last price
        if str(cur_time) not in time_to_price:
            time_to_price[str(cur_time)] = last_price
        else:
            last_price = time_to_price[str(cur_time)]
        cur_time += datetime.timedelta(seconds=60)

    # time_to_price map, first timestamp that has price data
    return time_to_price, prices_rows[0][1]


def get_price(time_to_price, first_price_timestamp, timestamp):
    return time_to_price[timestamp] if timestamp >= first_price_timestamp else 0


if __name__ == "__main__":
    run_balance()
