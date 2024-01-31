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
target_recipient = '0xD7acDd917Df3AB3349eb6377fDf519Fd44B42d44'


def main():
    add_blocks_for_processing(start=19089948, end=19089950)
    process_blocks()


def process_blocks():
    q = attach_queue()
    i = 0

    while q.qsize() > 0:
        if i % 10 == 0:
            print(f'queue size: {q.qsize()}')
        i += 1

        item = q.get()
        process_block(item)
        q.ack(item)


def process_block(block):
    web3 = Web3(HTTPProvider(ankr_endpoint))
    matching_logs = web3.eth.get_logs({
        'fromBlock': block,
        'toBlock': block,
        'topics': [transfer_function_hash],
        'address': '0x8457CA5040ad67fdebbCC8EdCE889A335Bc0fbFB',  # AltLayer Token (ALT)
    })

    # parse topics and only process target address
    for log in matching_logs:
        recipient = decode(['address'], log['topics'][2])[0]
        if recipient.lower() == target_recipient.lower():
            write_txn(log)



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


def write_txn(log):
    txn = create_txn(log)

    conn = create_connection()
    insert_txn(conn, txn)
    print('finished write_txn')


def add_blocks_for_processing(start, end):
    q = attach_queue()
    q.clear_acked_data(keep_latest=0)
    i = start

    while i <= end:
        q.put(i)
        i += 1

    print(f'added blocks from {start} to {end} for processing')


main()
