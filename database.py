import sqlite3
from sqlite3 import Error
import persistqueue
from eth_abi import decode

database_path = r"/Users/tonywang/projects/stonks/test.db"
queue_path = r"/Users/tonywang/projects/stonks/test_queue.db"

erc20_padding = 10 ** 18


def create_connection(db_file=database_path):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
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


def attach_queue(path=queue_path):
    return persistqueue.UniqueAckQ(path=path)
