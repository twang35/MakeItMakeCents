import time
import requests
from eth_typing import BlockNumber
from web3 import Web3, HTTPProvider
from pprint import pprint
import datetime
from database import *

with open('transpose_key.txt', 'r') as f:
    transpose_key = f.read()
ankr_endpoint = 'https://rpc.ankr.com/eth'  # url string
# Web3.utils.keccak256("Transfer(address,address,uint256)")
transfer_function_hash = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'
altlayer_token_address = '0x8457CA5040ad67fdebbCC8EdCE889A335Bc0fbFB'  # AltLayer Token (ALT)

batch_block_time_increment = 5000
batch_txn_logs_increment = 100


def blockchain():
    print('hello blockchain')
    only_print_queue(queue='test')

    # transactions: start=19082604, end=19152064
    # transactions: start=18937380, end=19082604, 401071 txns before, 402644 after
    # add_items_to_queue(start=18937380, end=19082604, queue='test', increment=batch_txn_logs_increment)

    # block_time:   start=16300000, end=19166426 batch
    # add_items_to_queue(start=19082604, end=19082604, queue='test')

    # table: transactions, block_times
    process_blocks(queue='test', table='transactions', max_block_num=19202534)


# max_block_num ensures only finalized blocks are processed: current finalized block is 19202534 as of Feb 11, 2024
def process_blocks(queue, table, max_block_num, token_address=altlayer_token_address):
    q = attach_queue(queue=queue)
    conn = create_connection()
    web3 = Web3(HTTPProvider(ankr_endpoint))
    i = 0
    start = time.time()
    print_interval = 10
    block_to_time = get_block_times_map(conn)

    while q.qsize() > 0:
        if i % print_interval == 0:
            end = time.time()
            q_size = q.qsize()
            velocity = print_interval / (end - start)
            estimate = str(datetime.timedelta(seconds=q_size / velocity))
            print(f'queue size: {q_size}, velocity: {"%.4f" % velocity} elements/second,'
                  f' completion estimate: {estimate}')
            start = time.time()
        i += 1

        item = q.get()
        if table == 'transactions':
            process_txn_block(item, conn, web3, token_address, max_block_num, block_to_time)
        elif table == 'block_times':
            process_batch_block_times(item, max_block_num, conn)

        q.ack(item)


def process_txn_block(block, conn, web3, token_address, max_block_num, block_to_time):
    # parse topics and only process target address
    matching_logs = web3.eth.get_logs({
        'fromBlock': block,
        'toBlock': min(block + batch_txn_logs_increment, max_block_num),
        'topics': [transfer_function_hash],
        'address': token_address,
    })

    for log in matching_logs:
        write_txn(log, block_to_time, conn)


def process_batch_block_times(block_num, max_block_num, conn):
    block_times = get_transpose_block_times(block_num, block_num + batch_block_time_increment-1, max_block_num)
    time.sleep(1)
    i = 0

    while i < len(block_times):
        block_time = block_times[i]
        timestamp = datetime.datetime.fromisoformat(block_time['timestamp'])
        epoch_time = int(timestamp.timestamp())
        timestamp_string = timestamp.strftime('%Y-%m-%d %H:%M:%S')

        row = (block_time['block_number'], timestamp_string, epoch_time)

        insert_block_time(conn, row)
        i += 1


def get_transpose_block_times(start, end, max_block):
    url = "https://api.transpose.io/sql"
    sql_query = f"""
    SELECT block_number, timestamp
    FROM ethereum.blocks
    WHERE block_number BETWEEN {start} AND {min(end+1, max_block+1)};
    """

    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': transpose_key,
    }
    response = requests.post(url,
                             headers=headers,
                             json={
                                 'sql': sql_query,
                             },
                             )

    block_times = response.json()['results']
    print('Transpose credits charged:', response.headers.get('X-Credits-Charged', None))
    return block_times


def get_block_times_map(conn):
    cursor = conn.cursor()
    query = """
        SELECT * FROM block_times;
        """
    cursor.execute(query)
    block_times_rows = cursor.fetchall()
    block_to_time = {}

    # block_number, timestamp, epoch_seconds
    for row in block_times_rows:
        block_to_time[row[0]] = row[1]

    return block_to_time


if __name__ == "__main__":
    blockchain()
