import sqlite3
import datetime
from sqlite3 import Error
import persistqueue
from eth_abi import decode
from pprint import pprint
import time

database_path = r"/Users/tonywang/projects/stonks/test.db"
test_queue_path = r"/Users/tonywang/projects/stonks/test_queue.db"
balance_queue_path = r"/Users/tonywang/projects/stonks/balance_queue.db"
update_queue_path = r"/Users/tonywang/projects/stonks/update_queue.db"

erc20_padding = 10 ** 18

altlayer_token_address = '0x8457CA5040ad67fdebbCC8EdCE889A335Bc0fbFB'
pepefork_token_address = '0xb9f599ce614Feb2e1BBe58F180F370D05b39344E'


def create_connection(db_file=database_path):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f'database connected at {database_path} with version {sqlite3.version}')
    except Error as e:
        print(e)

    return conn


def create_transactions_table():
    # block_number, transaction_index, log_index, sender, recipient, token_address, value
    sql_create_transactions_table = """ CREATE TABLE IF NOT EXISTS transactions (
                                        block_number INTEGER NOT NULL,
                                        transaction_index INTEGER NOT NULL,
                                        log_index INTEGER NOT NULL,
                                        timestamp TEXT NOT NULL,
                                        sender TEXT NOT NULL,
                                        recipient TEXT NOT NULL,
                                        token_address TEXT NOT NULL,
                                        value REAL NOT NULL,
                                        PRIMARY KEY (recipient, token_address, block_number, transaction_index, log_index)
                                        ); """

    create_table(sql_create_transactions_table)


def create_block_times_table():
    # PRIMARY(block_number, timestamp, epoch_seconds)
    sql_create_block_table = """ CREATE TABLE IF NOT EXISTS block_times (
                                        block_number INTEGER NOT NULL,
                                        timestamp TEXT NOT NULL,
                                        epoch_seconds INTEGER NOT NULL,
                                        PRIMARY KEY (block_number, timestamp, epoch_seconds)
                                        ); """

    create_table(sql_create_block_table)


def create_prices_table():
    # PRIMARY(token_address, timestamp), token_symbol, price, volume
    sql_create_price_table = """ CREATE TABLE IF NOT EXISTS prices (
                                        token_address TEXT NOT NULL,
                                        timestamp TEXT NOT NULL,
                                        token_symbol TEXT NOT NULL,
                                        price REAL NOT NULL,
                                        volume REAL NOT NULL,
                                        PRIMARY KEY (token_address, timestamp)
                                        ); """

    create_table(sql_create_price_table)


def create_balances_table():
    # PRIMARY(wallet_address, token_address, timestamp),
    #   balance, total_cost_basis, remaining_cost_basis, realized_gains, unrealized_gains
    sql_create_balances_table = """ CREATE TABLE IF NOT EXISTS balances (
                                        wallet_address TEXT NOT NULL,
                                        token_address TEXT NOT NULL,
                                        timestamp TEXT NOT NULL,
                                        block_number INTEGER NOT NULL,
                                        balance REAL NOT NULL,
                                        total_cost_basis REAL NOT NULL,
                                        remaining_cost_basis REAL NOT NULL,
                                        realized_gains REAL NOT NULL,
                                        PRIMARY KEY (wallet_address, token_address, timestamp, block_number)
                                        ); """

    create_table(sql_create_balances_table)


def create_table(create_sql):
    conn = create_connection()
    c = conn.cursor()
    c.execute(create_sql)

    print(f'Created table with {create_sql}')


def write_txn(log, block_to_time, conn):
    """
    address: token contract address
    topics: [function signature hash, sender address, recipient address]
    data: value of transfer
    """
    block_number = log['blockNumber']
    sender = decode(['address'], log['topics'][1])[0]
    recipient = decode(['address'], log['topics'][2])[0]
    value = decode(['uint256'], log['data'])[0] / erc20_padding
    # block_number, transaction_index, log_index, timestamp, sender, recipient, token_address, value
    txn = (block_number, log['transactionIndex'], log['logIndex'], block_to_time[block_number], sender, recipient,
           log['address'], value)

    insert_txn(conn, txn)


def insert_txn(conn, txn):
    sql = '''INSERT OR REPLACE INTO transactions(block_number, transaction_index, log_index, timestamp, sender,
     recipient, token_address, value) VALUES(?,?,?,?,?,?,?,?);'''
    cur = conn.cursor()
    cur.execute(sql, txn)
    conn.commit()


def insert_block_time(conn, row):
    sql = '''INSERT OR REPLACE INTO block_times(block_number, timestamp, epoch_seconds) VALUES(?,?,?);'''
    cur = conn.cursor()
    cur.execute(sql, row)
    conn.commit()


def insert_price(conn, row):
    sql = '''INSERT OR REPLACE INTO prices(token_address, timestamp, token_symbol, price, volume) VALUES(?,?,?,?,?);'''
    cur = conn.cursor()
    cur.execute(sql, row)
    conn.commit()


def insert_balance(conn, row):
    sql = '''INSERT OR REPLACE INTO balances(wallet_address, token_address, timestamp, block_number, balance,
     total_cost_basis, remaining_cost_basis, realized_gains) VALUES(?,?,?,?,?,?,?,?);'''
    cur = conn.cursor()
    cur.execute(sql, row)
    conn.commit()


def get_latest_block_times_block(conn):
    cursor = conn.cursor()
    query = f"""
        SELECT MAX(block_number) AS latest_block_number
        FROM block_times;
        """
    cursor.execute(query)
    result = cursor.fetchall()
    return result[0][0]


def get_latest_transactions_block(conn, token_address):
    cursor = conn.cursor()
    query = f"""
        SELECT MAX(block_number) AS latest_block_number
        FROM transactions
        WHERE token_address = '{token_address}';
        """
    cursor.execute(query)
    result = cursor.fetchall()
    return result[0][0]


def get_latest_balances_block(conn, token_address):
    cursor = conn.cursor()
    query = f"""
        SELECT MAX(block_number) AS latest_block_number
        FROM balances
        WHERE token_address = '{token_address}';
        """
    cursor.execute(query)
    result = cursor.fetchall()
    return result[0][0]


def get_latest_price_timestamp(conn, token_address):
    cursor = conn.cursor()
    query = f"""
        SELECT MAX(timestamp) AS latest_timestamp
        FROM prices
        WHERE token_address = '{token_address}';
        """
    cursor.execute(query)
    result = cursor.fetchall()
    return datetime.datetime.fromisoformat(result[0][0])


def get_latest_wallet_balances(conn, token_address):
    cursor = conn.cursor()

    start_time = time.time()
    query = f"""
        SELECT 
            *
        FROM (
            SELECT 
                wallet_address, 
                balance,
                total_cost_basis,
                remaining_cost_basis,
                realized_gains,
                ROW_NUMBER() OVER (PARTITION BY wallet_address ORDER BY block_number DESC) AS row_num
            FROM 
                balances
            WHERE 
                token_address = '{token_address}'
        ) AS ranked
        WHERE 
            row_num = 1;
        """
    cursor.execute(query)
    balances = cursor.fetchall()
    print("Total unique wallets are: ", len(balances))
    print(f'query time: {time.time() - start_time}')
    return balances


def attach_queue(queue):
    """
    class AckStatus(object):
        inited = '0'        # all new entries start with 0
        ready = '1'         # ready objects will be given to consumers
        unack = '2'         # has been given to a consumer, but not acked yet.
                                unack will revert to ready when queue is attached
        acked = '5'
        ack_failed = '9'
    """
    path = test_queue_path
    if queue == 'test':
        path = test_queue_path
    elif queue == 'balance':
        path = balance_queue_path
    elif queue == 'update':
        path = update_queue_path
    return persistqueue.UniqueAckQ(path=path, multithreading=True)


def add_items_to_queue(start, end, queue='test', increment=1):
    q = attach_queue(queue)
    q.clear_acked_data(keep_latest=0, max_delete=10000000)
    i = start

    while i <= end:
        q.put(i)
        i += increment

    print(f'added items from {start} to {end} for processing to {queue} queue')


def only_print_queue(only_size=False, queue='test'):
    q = attach_queue(queue)
    print(f'queue size: {q.qsize()}')
    if only_size:
        exit()
    full_data = q.queue()
    values = [element['data'] for element in full_data if element['status'] != 5]
    pprint(values)
    exit()
