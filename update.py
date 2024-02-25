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


def update():
    # only_print_queue(queue='update')

    # Update existing token #########################
    # update_all_data(pepefork.address)

    # Download new token #########################
    token_address = xcad.address

    start_txn_block = 12419395  # xcad started around 12429395
    start_price_time = datetime.datetime.fromisoformat('2021-05-12 00:45:59')

    update_all_data(token_address, new_token=True, start_txn_block=start_txn_block, start_price_time=start_price_time)


def update_all_data(token_address, new_token=False, start_txn_block=None, start_price_time=None):
    queue_name = 'update'
    conn = create_connection()
    web3 = Web3(HTTPProvider(ankr_endpoint))

    if not new_token:
        start_txn_block = get_latest_transactions_block(conn, token_address)
        start_price_time = get_latest_price_timestamp(conn, token_address)

    # update for new tokens if the start_txn_block is before the earliest block in block_times
    if new_token:
        print('====== start updating block_times before earliest block_times block =============')
        earliest_block_times_block = get_earliest_block_times_block(conn)
        add_items_to_queue(start=start_txn_block,
                           end=earliest_block_times_block, queue=queue_name,
                           increment=batch_block_time_increment)
        process_blocks(queue='update', table='block_times', max_block_num=earliest_block_times_block)
        print('====== completed updating block_times before earliest block_times block =========')

    latest_finalized_block = web3.eth.get_block(block_identifier='finalized')['number']

    # update block_times until latest finalized block
    print('====== start updating block_times =============')
    add_items_to_queue(start=get_latest_block_times_block(conn),
                       end=latest_finalized_block, queue=queue_name,
                       increment=batch_block_time_increment)
    process_blocks(queue='update', table='block_times', max_block_num=latest_finalized_block)
    print('====== completed updating block_times =========')

    # update transactions, uses block_times to populate timestamp field
    print('====== start updating transactions ============')
    add_items_to_queue(start=start_txn_block,
                       end=latest_finalized_block, queue=queue_name,
                       increment=batch_txn_logs_increment)
    process_blocks(queue=queue_name, table='transactions', max_block_num=latest_finalized_block,
                   token_address=token_address)
    print('====== completed updating transactions ========')

    # update prices
    print('====== start updating prices ==================')
    add_items_to_queue(start=start_price_time,
                       end=datetime.datetime.utcnow(), queue=queue_name,
                       increment=batch_price_date_increment)
    process_price_dates(token_address=token_address, queue_name=queue_name)
    print('====== completed updating prices ==============')

    # update balances
    print('====== start updating balances ================')
    compute_balances(conn, token_address)
    print('====== completed updating balances ============')


if __name__ == "__main__":
    update()
