import time
import requests
from eth_typing import BlockNumber
from web3 import Web3, HTTPProvider
from pprint import pprint
import datetime

from database import *
from blockchain import ankr_endpoint, process_blocks, batch_block_time_increment, batch_txn_logs_increment

altlayer_token_address = '0x8457CA5040ad67fdebbCC8EdCE889A335Bc0fbFB'


def update():
    # only_print_queue(queue='balance')

    conn = create_connection()

    web3 = Web3(HTTPProvider(ankr_endpoint))
    latest_finalized_block = web3.eth.get_block(block_identifier='finalized')['number']

    # update block_times
    last_block_time_block = get_latest_block_times_block(conn)
    add_blocks_for_processing(start=last_block_time_block, end=latest_finalized_block, queue='update',
                              increment=batch_block_time_increment)
    process_blocks(queue='update', table='block_times', max_block_num=latest_finalized_block)

    # update transactions, uses block_times to populate timestamp field
    last_txn_block = get_latest_transactions_block(conn, altlayer_token_address)
    # last_txn_block = 18932562  # Jan-04-2024 07:45:23 AM +UTC for recreating the transactions table
    add_blocks_for_processing(start=last_txn_block, end=latest_finalized_block, queue='update',
                              increment=batch_txn_logs_increment)
    process_blocks(queue='update', table='transactions', max_block_num=latest_finalized_block,
                   token_address=altlayer_token_address)

    # update prices

    # update balances


if __name__ == "__main__":
    update()
