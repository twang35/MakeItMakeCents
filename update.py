import time
import requests
from eth_typing import BlockNumber
from web3 import Web3, HTTPProvider
from pprint import pprint
import datetime

from blockchain import *
from database import *

altlayer_token_address = '0x8457CA5040ad67fdebbCC8EdCE889A335Bc0fbFB'

def main():
    conn = create_connection()
    last_txn_block = get_latest_transactions_block(conn, altlayer_token_address)

    web3 = Web3(HTTPProvider(ankr_endpoint))
    latest_finalized_block = web3.eth.get_block(block_identifier='finalized')['number']

    # update transactions
    add_blocks_for_processing(start=last_txn_block, end=latest_finalized_block, queue='test',
                              increment=batch_txn_logs_increment)
    process_blocks(queue='test', table='transactions', token_address=altlayer_token_address)


main()
