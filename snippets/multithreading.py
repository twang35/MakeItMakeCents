import time
from time import sleep

from eth_typing import BlockNumber
from web3 import Web3, HTTPProvider
from database import *
import datetime
from eth_abi import decode
from concurrent.futures import ThreadPoolExecutor, wait

"""
This is not being used because it's too fast and runs into API rate limit errors within a few seconds.
"""
def main():
    # attach things
    web3 = Web3(HTTPProvider('https://rpc.ankr.com/eth'))

    # add to balance queue
    add_items_to_queue(start=2430001, end=2430100, queue='balance')

    queue = attach_queue('balance')
    start = time.time()
    process_blocks_threaded(web3, queue)
    print(f'time: {time.time() - start}')


def process_blocks_threaded(web3, queue):
    with ThreadPoolExecutor(max_workers=10) as executor:
        while queue.qsize() > 0:
            to_write = []
            futures = [executor.submit(fetch_block, web3, queue, to_write) for _ in range(10)]
            # wait for all download tasks to complete
            done, not_done = wait(futures)
            for item, _, _ in to_write:
                queue.ack(item)


def process_blocks(web3, queue):
    while queue.qsize() > 0:
        to_write = []
        fetch_block(web3, queue, to_write)
        for item, _, _ in to_write:
            queue.ack(item)


def fetch_block(web3, queue, to_write):
    item = queue.get(block=False)
    block_data = web3.eth.get_block(block_identifier=BlockNumber(item), full_transactions=False)
    epoch_time = block_data['timestamp']
    timestamp_string = datetime.datetime.utcfromtimestamp(epoch_time).strftime('%Y-%m-%d %H:%M:%S')

    row = (item, timestamp_string, epoch_time)

    to_write.append(row)


main()
