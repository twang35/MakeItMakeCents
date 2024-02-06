import sqlite3
from sqlite3 import Error
import persistqueue
from eth_abi import decode
from pprint import pprint

database_path = r"/Users/tonywang/projects/stonks/test.db"
test_queue_path = r"/Users/tonywang/projects/stonks/test_queue.db"
balance_queue_path = r"/Users/tonywang/projects/stonks/balance_queue.db"

erc20_padding = 10 ** 18


def create_connection(db_file=database_path):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f'database connected at {database_path} with version {sqlite3.version}')
    except Error as e:
        print(e)

    return conn


def create_txn_table():
    # block_number, transaction_index, log_index, sender, recipient, token_id, value, timestamp
    sql_create_transactions_table = """ CREATE TABLE IF NOT EXISTS transactions (
                                        block_number INTEGER NOT NULL,
                                        transaction_index INTEGER NOT NULL,
                                        log_index INTEGER NOT NULL,
                                        sender TEXT NOT NULL,
                                        recipient TEXT NOT NULL,
                                        token_id TEXT NOT NULL,
                                        value REAL NOT NULL,
                                        PRIMARY KEY (recipient, token_id, block_number, transaction_index, log_index)
                                        ); """

    create_table(sql_create_transactions_table)


def create_block_time_table():
    # PRIMARY(block_number, timestamp, epoch_seconds)
    sql_create_block_table = """ CREATE TABLE IF NOT EXISTS block_time (
                                        block_number INTEGER NOT NULL,
                                        timestamp TEXT NOT NULL,
                                        epoch_seconds INTEGER NOT NULL,
                                        PRIMARY KEY (block_number, timestamp, epoch_seconds)
                                        ); """

    create_table(sql_create_block_table)


def create_price_table():
    # PRIMARY(token_address, timestamp), token_symbol, price, volume
    sql_create_price_table = """ CREATE TABLE IF NOT EXISTS price (
                                        token_address TEXT NOT NULL,
                                        timestamp TEXT NOT NULL,
                                        token_symbol TEXT NOT NULL,
                                        price REAL NOT NULL,
                                        volume REAL NOT NULL,
                                        PRIMARY KEY (token_address, timestamp)
                                        ); """

    create_table(sql_create_price_table)


def create_table(create_sql):
    conn = create_connection()
    c = conn.cursor()
    c.execute(create_sql)

    print(f'Created table with {create_sql}')


def create_txn(log):
    """
    address: token contract address
    topics: [function signature hash, sender address, recipient address]
    data: value of transfer
    """
    sender = decode(['address'], log['topics'][1])[0]
    recipient = decode(['address'], log['topics'][2])[0]
    value = decode(['uint256'], log['data'])[0] / erc20_padding
    # block_number, transaction_index, log_index, sender, recipient, token_id, value
    return log['blockNumber'], log['transactionIndex'], log['logIndex'], sender, recipient, log['address'], value


def write_txn(log, conn):
    txn = create_txn(log)

    insert_txn(conn, txn)


def insert_txn(conn, txn):
    """
    Insert or replace a new txn row into the table.
    """

    sql = '''INSERT OR REPLACE INTO transactions(block_number, transaction_index, log_index, sender, recipient,
     token_id, value) VALUES(?,?,?,?,?,?,?)'''
    cur = conn.cursor()
    cur.execute(sql, txn)
    conn.commit()


def insert_block_time(conn, row):
    """
    Insert or replace a new txn row into the table.
    """

    sql = '''INSERT OR REPLACE INTO block_time(block_number, timestamp, epoch_seconds) VALUES(?,?,?)'''
    cur = conn.cursor()
    cur.execute(sql, row)
    conn.commit()


def insert_price(conn, row):
    """
    Insert or replace a new txn row into the table.
    """

    sql = '''INSERT OR REPLACE INTO price(token_address, timestamp, token_symbol, price, volume) VALUES(?,?,?,?,?)'''
    cur = conn.cursor()
    cur.execute(sql, row)
    conn.commit()


"""
class AckStatus(object):
    inited = '0'        # all new entries start with 0
    ready = '1'         # unack will revert to ready when queue is attached and ready objects will be given to consumers
    unack = '2'         # has been given to consumer, but not acked yet
    acked = '5'
    ack_failed = '9'
"""
def attach_queue(queue='test'):
    path = test_queue_path
    if queue == 'test':
        path = test_queue_path
    elif queue == 'balance':
        path = balance_queue_path
    return persistqueue.UniqueAckQ(path=path, multithreading=True)


def add_blocks_for_processing(start, end, queue='test', increment=1):
    q = attach_queue(queue)
    q.clear_acked_data(keep_latest=0, max_delete=10000000)
    i = start

    while i <= end:
        q.put(i)
        i += increment

    print(f'added blocks from {start} to {end} for processing to {queue} queue')


def only_print_queue(only_size=False, queue='test'):
    q = attach_queue(queue)
    print(f'queue size: {q.qsize()}')
    if only_size:
        exit()
    full_data = q.queue()
    values = [element['data'] for element in full_data if element['status'] != 5]
    pprint(values)
    exit()
