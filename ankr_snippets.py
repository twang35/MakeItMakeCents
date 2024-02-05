import eth_utils
from eth_typing import BlockNumber
from web3 import Web3, HTTPProvider
from pprint import pprint
from eth_abi import decode
import datetime
from database import *

ankr_endpoint = 'https://rpc.ankr.com/eth'  # url string
# Web3.utils.keccak256("Transfer(address,address,uint256)")
transfer_function_hash = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
token_address = '0x8457CA5040ad67fdebbCC8EdCE889A335Bc0fbFB'  # AltLayer Token (ALT)


def main():
    # only_print_queue()

    # started with 69461 queue size, start=19082604, end=19152064
    add_blocks_for_processing(start=19082604, end=19152064)
    # table: transactions, block_time
    process_blocks(table='block_time')


def only_print_queue():
    q = attach_queue()
    print(f'queue size: {q.qsize()}')
    full_data = q.queue()
    values = [element['data'] for element in full_data if element['status'] != 5]
    pprint(values)
    exit()


def process_blocks(table='transactions'):
    q = attach_queue()
    conn = create_connection()
    web3 = Web3(HTTPProvider(ankr_endpoint))
    i = 0

    while q.qsize() > 0:
        if i % 10 == 0:
            print(f'queue size: {q.qsize()}')
        i += 1

        item = q.get()
        if table == 'transactions':
            process_txn_block(item, conn, web3)
        elif table == 'block_time':
            process_block_time_block(item, conn, web3)

        q.ack(item)


def process_txn_block(block, conn, web3):
    matching_logs = web3.eth.get_logs({
        'fromBlock': block,
        'toBlock': block,
        'topics': [transfer_function_hash],
        'address': token_address,
    })

    # parse topics and only process target address
    for log in matching_logs:
        write_txn(log, conn)


def process_block_time_block(block_num, conn, web3):
    block_data = web3.eth.get_block(block_identifier=BlockNumber(block_num), full_transactions=False)

    epoch_time = block_data['timestamp']
    timestamp_string = datetime.datetime.utcfromtimestamp(epoch_time).strftime('%Y-%m-%d %H:%M:%S')

    row = (block_num, timestamp_string, epoch_time)

    insert_block_time(conn, row)


def add_blocks_for_processing(start, end):
    q = attach_queue()
    q.clear_acked_data(keep_latest=0, max_delete=100000)
    i = start

    while i <= end:
        q.put(i)
        i += 1

    print(f'added blocks from {start} to {end} for processing')


main()
