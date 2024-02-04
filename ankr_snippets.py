import eth_utils
from eth_typing import BlockNumber
from web3 import Web3, HTTPProvider
from pprint import pprint
from eth_abi import decode
from database import *

erc20_padding = 10 ** 18
ankr_endpoint = 'https://rpc.ankr.com/eth'  # url string
# Web3.utils.keccak256("Transfer(address,address,uint256)")
transfer_function_hash = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
token_address = '0x8457CA5040ad67fdebbCC8EdCE889A335Bc0fbFB'  # AltLayer Token (ALT)


def main():
    only_print_queue()

    # started with 69461 queue size
    # add_blocks_for_processing(start=19082604, end=19152064)
    # process_blocks()


def only_print_queue():
    q = attach_queue()
    print(f'queue size: {q.qsize()}')
    full_data = q.queue()
    values = [element['data'] for element in full_data if element['status'] != 5]
    pprint(values)
    exit()


def process_blocks():
    q = attach_queue()
    conn = create_connection()
    i = 0

    while q.qsize() > 0:
        if i % 10 == 0:
            print(f'queue size: {q.qsize()}')
        i += 1

        item = q.get()
        process_block(item, conn)
        q.ack(item)


def process_block(block, conn):
    web3 = Web3(HTTPProvider(ankr_endpoint))
    matching_logs = web3.eth.get_logs({
        'fromBlock': block,
        'toBlock': block,
        'topics': [transfer_function_hash],
        'address': token_address,
    })

    # parse topics and only process target address
    for log in matching_logs:
        write_txn(log, conn)


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


def add_blocks_for_processing(start, end):
    q = attach_queue()
    q.clear_acked_data(keep_latest=0)
    i = start

    while i <= end:
        q.put(i)
        i += 1

    print(f'added blocks from {start} to {end} for processing')


main()
