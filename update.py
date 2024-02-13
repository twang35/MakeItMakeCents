import time
import requests
from eth_typing import BlockNumber
from web3 import Web3, HTTPProvider
from pprint import pprint
import datetime

from balance import compute_balances
from database import *
from blockchain import ankr_endpoint, process_blocks, batch_block_time_increment, batch_txn_logs_increment
from price import batch_price_date_increment, process_price_dates

altlayer_token_address = '0x8457CA5040ad67fdebbCC8EdCE889A335Bc0fbFB'


def update():
    # only_print_queue(queue='update')

    token_address = altlayer_token_address
    queue_name = 'update'
    conn = create_connection()
    web3 = Web3(HTTPProvider(ankr_endpoint))

    latest_finalized_block = web3.eth.get_block(block_identifier='finalized')['number']
    # start_block = 18937380
    start_txn_block = get_latest_transactions_block(conn, token_address)
    # start_time = datetime.datetime.fromisoformat('2024-02-03 20:26:00')
    start_time = get_latest_price_timestamp(conn, token_address)

    # update block_times
    add_items_to_queue(start=get_latest_block_times_block(conn),
                       end=latest_finalized_block, queue=queue_name,
                       increment=batch_block_time_increment)
    process_blocks(queue='update', table='block_times', max_block_num=latest_finalized_block)
    print('=== completed updating block_times =========')

    # update transactions, uses block_times to populate timestamp field
    add_items_to_queue(start=start_txn_block,
                       end=latest_finalized_block, queue=queue_name,
                       increment=batch_txn_logs_increment)
    process_blocks(queue=queue_name, table='transactions', max_block_num=latest_finalized_block,
                   token_address=token_address)
    print('=== completed updating transactions =========')

    # update prices
    add_items_to_queue(start=start_time,
                       end=datetime.datetime.utcnow(), queue=queue_name,
                       increment=batch_price_date_increment)
    process_price_dates(token_address=token_address, queue_name=queue_name)
    print('=== completed updating prices =========')

    # update balances
    compute_balances(conn, altlayer_token_address)
    print('=== completed updating balances =========')


if __name__ == "__main__":
    update()
