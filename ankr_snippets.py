from eth_typing import BlockNumber
from web3 import Web3, HTTPProvider
from pprint import pprint
import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)

    return conn


def create_table():
    database = r"/Users/tonywang/projects/stonks/test.db"

    # block_number, transaction_index, sender, recipient, token_id, value, timestamp

    sql_create_transactions_table = """ CREATE TABLE IF NOT EXISTS transactions (
                                            block_number INTEGER NOT NULL,
                                            transaction_index INTEGER NOT NULL,
                                            sender TEXT NOT NULL,
                                            recipient TEXT NOT NULL,
                                            token_id TEXT NOT NULL,
                                            value REAL NOT NULL,
                                            timestamp INTEGER NOT NULL,
                                            PRIMARY KEY (block_number, transaction_index)
                                        ); """

    # create a database connection
    conn = create_connection(database)

    # create tables
    if conn is not None:
        # create projects table
        try:
            c = conn.cursor()
            c.execute(sql_create_transactions_table)
        except Error as e:
            print(e)
    else:
        print("Error! cannot create the database connection.")


def test_block_number():
    url = 'https://rpc.ankr.com/eth'  # url string

    web3 = Web3(HTTPProvider(url))
    block = web3.eth.get_block(block_identifier=BlockNumber(19089948), full_transactions=True)
    # pprint(block['transactions'])
    pprint(block)
    print('finished test_block_number')


def write_txn(conn):
    url = 'https://rpc.ankr.com/eth'  # url string

    web3 = Web3(HTTPProvider(url))
    block = web3.eth.get_block(block_identifier=BlockNumber(19089948), full_transactions=True)
    transactions = block['transactions']
    example = transactions[151]

    # block_number, transaction_index, sender, recipient, token_id, value, timestamp
    txn = (block['number'], example['transactionIndex'], example['from'], 'recipient_test'
           , example['to']
           , 0.0123
           , block['timestamp']
           )
    print(txn)
    insert_txn(conn, txn)
    # pprint(web3.to_json(example))
    # pprint(block)
    print('finished write_txn')

def insert_txn(conn, txn):
    """
    Insert a new txn row into the table.
    """

    sql = ''' INSERT INTO transactions(block_number, transaction_index, sender, recipient, token_id, value, timestamp)
              VALUES(?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, txn)
    conn.commit()
    return cur.lastrowid


database = r"/Users/tonywang/projects/stonks/test.db"
conn = create_connection(database)
# test_block_number()
# create_table()
write_txn(conn)
