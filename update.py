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
    check_queue_size(queue='update')

    # Update existing token #########################
    update_all_data(altlayer)
    update_all_data(pepefork)
    update_all_data(xcad)
    update_all_data(mubi)

    # Download new token #########################
    # token_address = mubi.address
    #
    # start_txn_block = 18510688  # mubi started around 18520688
    # start_price_time = datetime.datetime.fromisoformat('2023-11-05 00:45:59')
    #
    # update_all_data(token_address, new_token=True, start_txn_block=start_txn_block, start_price_time=start_price_time)

    print('update completed')


def check_queue_size(queue):
    q = attach_queue(queue)
    q_size = q.qsize()
    if q_size == 0:
        print(f'queue size: {q_size}')
    else:
        if input(f'queue size: {q_size} \nCancel processing (y, n): ').lower() == 'y':
            exit()


def update_all_data(token, new_token=False, start_txn_block=None, start_price_time=None):
    queue_name = 'update'
    conn = create_connection()
    web3 = Web3(HTTPProvider(ankr_endpoint))

    if not new_token:
        start_txn_block = get_latest_transactions_block(conn, token.address)
        start_price_time = get_latest_price_timestamp(conn, token.address)

    # update for new tokens if the start_txn_block is before the earliest block in block_times
    if new_token:
        print(f'====== start updating block_times before earliest block_times block for {token.name} =============')
        earliest_block_times_block = get_earliest_block_times_block(conn)
        add_items_to_queue(start=start_txn_block,
                           end=earliest_block_times_block, queue=queue_name,
                           increment=batch_block_time_increment)
        process_blocks(queue='update', table='block_times', max_block_num=earliest_block_times_block)
        print(f'====== completed updating block_times before earliest block_times block for {token.name} =========')

    latest_finalized_block = web3.eth.get_block(block_identifier='finalized')['number']

    # update block_times until latest finalized block
    print(f'====== start updating block_times for {token.name} =============')
    add_items_to_queue(start=get_latest_block_times_block(conn),
                       end=latest_finalized_block, queue=queue_name,
                       increment=batch_block_time_increment)
    process_blocks(queue='update', table='block_times', max_block_num=latest_finalized_block)
    print(f'====== completed updating block_times for {token.name} =========')

    # update transactions, uses block_times to populate timestamp field
    print(f'====== start updating transactions for {token.name} ============')
    add_items_to_queue(start=start_txn_block,
                       end=latest_finalized_block, queue=queue_name,
                       increment=batch_txn_logs_increment)
    process_blocks(queue=queue_name, table='transactions', max_block_num=latest_finalized_block,
                   token_address=token.address)
    print(f'====== completed updating transactions for {token.name} ========')

    # update prices
    print(f'====== start updating prices for {token.name} ==================')
    add_items_to_queue(start=start_price_time,
                       end=datetime.datetime.utcnow(), queue=queue_name,
                       increment=batch_price_date_increment)
    process_price_dates(token_address=token.address, queue_name=queue_name)
    print(f'====== completed updating prices for {token.name} ==============')

    # update balances
    print(f'====== start updating balances for {token.name} ================')
    compute_balances(conn, token.address)
    print(f'====== completed updating balances for {token.name} ============')


if __name__ == "__main__":
    update()
