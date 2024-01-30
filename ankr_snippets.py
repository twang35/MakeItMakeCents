from eth_typing import BlockNumber
from web3 import Web3, HTTPProvider
from pprint import pprint
from eth_abi import decode
from database import *

erc20_padding = 10 ** 18
# Web3.utils.keccak256("Transfer(address,address,uint256)")
transfer_function_hash = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'


def write_txn(conn):
    url = 'https://rpc.ankr.com/eth'  # url string

    web3 = Web3(HTTPProvider(url))
    block = web3.eth.get_block(block_identifier=BlockNumber(19089948), full_transactions=True)
    transactions = block['transactions']
    example = transactions[151]
    input_data = example['input']
    print(f"input data: {(input_data, 0)}")

    # eth_utils.decode_hex(input)
    recipient, value = decode(['address', 'uint256'], input_data[4:])
    print(f'OG sender: {example["from"]}, recipient: {recipient}, value: {value / 10 ** 18}')

    matching_logs = web3.eth.get_logs({
        'fromBlock': 19089948,
        'toBlock': 19089948,
        'topics': [transfer_function_hash],
        'address': '0x8457CA5040ad67fdebbCC8EdCE889A335Bc0fbFB',  # AltLayer Token (ALT)
    })
    print(matching_logs)

    q = attach_queue()
    q.put(19089948)
    q.put(19089949)
    q.put(19089950)
    # item = q.get(timeout=1)
    # print(f'queue item: {item}')
    # q.ack(item)



    # block_number, transaction_index, sender, recipient, token_id, value, timestamp
    txn = (block['number'], example['transactionIndex'], example['from'], recipient, example['to'],
           value / erc20_padding, block['timestamp'])
    print(f'txn to write: {txn}')
    # insert_txn(conn, txn)
    # pprint(web3.to_json(example))
    print('finished write_txn')


conn = create_connection()
write_txn(conn)
