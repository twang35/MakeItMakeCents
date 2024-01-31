import sqlite3
from sqlite3 import Error
import persistqueue

database_path = r"/Users/tonywang/projects/stonks/test.db"
queue_path = r"/Users/tonywang/projects/stonks/test_queue.db"


def create_connection():
    return create_connection_from_file(database_path)


def create_connection_from_file(db_file=database_path):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)

    return conn


def create_txn_table(conn):
    # block_number, transaction_index, log_index, sender, recipient, token_id, value, timestamp
    sql_create_transactions_table = """ CREATE TABLE IF NOT EXISTS transactions (
                                            block_number INTEGER NOT NULL,
                                            transaction_index INTEGER NOT NULL,
                                            log_index INTEGER NOT NULL,
                                            sender TEXT NOT NULL,
                                            recipient TEXT NOT NULL,
                                            token_id TEXT NOT NULL,
                                            value REAL NOT NULL,
                                            timestamp INTEGER NOT NULL,
                                            PRIMARY KEY (recipient, token_id, block_number, transaction_index, log_index)
                                        ); """

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


def insert_txn(conn, txn):
    """
    Insert or replace a new txn row into the table.
    """

    sql = '''INSERT OR REPLACE INTO transactions(block_number, transaction_index, log_index, sender, recipient,
     token_id, value, timestamp) VALUES(?,?,?,?,?,?,?)'''
    cur = conn.cursor()
    cur.execute(sql, txn)
    conn.commit()


def attach_queue():
    return persistqueue.UniqueAckQ(path=queue_path)


conn = create_connection()
create_txn_table(conn)
