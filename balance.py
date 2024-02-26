import time
from pprint import pprint
import datetime
from database import *
from charts.shared_charts import *

null_address = '0x0000000000000000000000000000000000000000'
dead_address = '0x000000000000000000000000000000000000dead'
smallest_balance = 1e-12


def run_balance():
    # create_balances_table()
    conn = create_connection()

    token = pepefork

    compute_balances(conn, token.address)
    #
    # # balances = get_balances_before(conn, '2024-02-14 13:00:00', altlayer_token_address)
    # balances = get_balances_before(conn, datetime.datetime.utcnow(), token.address)
    # for i in range(100):
    #     print(balances[i])


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
    wallets = {null_address: [0, 0, 0, 0]}
    if latest_block != 0:
        # load the existing wallets from the balances table
        build_wallets(wallets, get_latest_wallet_balances(conn, token_address))

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

        block = txn[TransactionsColumns.block_number]
        sender = txn[TransactionsColumns.sender]
        recipient = txn[TransactionsColumns.recipient]
        token_address = txn[TransactionsColumns.token_address]
        value = txn[TransactionsColumns.value]
        price = get_price(time_to_price, first_price_timestamp, block_times[block])

        if value < smallest_balance or (wallets[sender][0] == 0 and value < 1):
            # don't count if value is too small
            # or if wallet has 0 balance due to SQLite REAL number precision being too low
            continue

        update_sender(wallets, sender, value, price, txn)
        update_recipient(wallets, recipient, value, price)

        # row: wallet_address, token_address, timestamp, block, balance, total_cost_basis, remaining_cost_basis,
        #   realized_gains

        # sender
        balance, total_cost_basis, remaining_cost_basis, realized_gains = wallets[sender]
        row = (sender, token_address, block_times[block], block,
               balance, total_cost_basis, remaining_cost_basis, realized_gains)
        insert_balance(conn, row)
        # recipient
        balance, total_cost_basis, remaining_cost_basis, realized_gains = wallets[recipient]
        row = (recipient, token_address, block_times[block], block,
               balance, total_cost_basis, remaining_cost_basis, realized_gains)
        insert_balance(conn, row)


def build_wallets(wallets, balances_rows):
    # wallet_address: [balance, total_cost_basis, remaining_cost_basis, realized_gains]
    for row in balances_rows:
        wallet_address, token_balance, total_cost_basis, remaining_cost_basis, realized_gains, _ = row
        wallets[wallet_address] = [token_balance, total_cost_basis, remaining_cost_basis, realized_gains]

    return wallets


def update_sender(wallets, sender, value, price, txn):
    if sender == null_address:
        # only calculate balance changes
        print(f'null address generated: {value}')
        wallets[sender][0] -= value
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
        output[row[BlockTimesColumns.block_number]] = row[BlockTimesColumns.timestamp][:-2] + '00'

    return output


if __name__ == "__main__":
    run_balance()
