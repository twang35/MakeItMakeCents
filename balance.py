import time
from pprint import pprint
import datetime
from database import *


def main():
    conn = create_connection()
    cursor = conn.cursor()

    # read first txn
    # block_number, transaction_index, log_index, sender, recipient, token_id, value
    # use limit 1000 and https://stackoverflow.com/questions/14468586
    query = """
        SELECT * FROM transactions 
        WHERE (block_number, transaction_index, log_index) > (18940311, 3, 2)
        ORDER BY block_number ASC 
        LIMIT 100
        """
    cursor.execute(query)
    records = cursor.fetchall()
    print("Total rows are:  ", len(records))
    print("Printing each row")
    for row in records:
        print(row)

    # get associated timestamps
    # load all timestamps in memory?

    # get price if available, else default to 0

    # write first row to balances
    #   PRIMARY(wallet_address, token_address, epoch_seconds),
    #   timestamp, balance, average_cost_basis, realized_gains, unrealized_gains


main()
