from collections import defaultdict

from plotly.subplots import make_subplots
from charts.shared_charts import *
from balance import *
from volume import *


def run_transaction_rows():
    token = pepefork

    conn = create_connection()
    cursor = conn.cursor()

    # calculate balances using txns and produce chart output
    timestamps, percentiles = generate_exchange_volume(cursor, token.address,
                                                       granularity=datetime.timedelta(minutes=60))

    # generate hourly graph
    prices = load_prices(cursor, token.address)
    create_volume_graph(prices, percentiles, timestamps, token, left_offset=1,
                        alt_title=f'{token.name} exchange volume',  view_date_start='2024-03-02 00:00:00')

    print('completed run_transaction_rows')


class TxnCounts:
    def __init__(self):
        self.sent_count = 0
        self.received_count = 0
        self.sent_to = {}
        self.sent_to = defaultdict(lambda: 0, self.sent_to)
        self.received_from = {}
        self.received_from = defaultdict(lambda: 0, self.received_from)


def generate_exchange_volume(cursor, token_address, granularity):
    print(f'running generate_exchange_volume')

    start = time.time()
    wallets = {null_address: WalletInfo(null_address)}
    timestamps = []
    percentiles = {  # balances are bucketed down
        0.1: [],
        1: [],
        10: [],
        50: [],
        100: [],
    }
    wallet_percentiles = generate_wallet_percentiles(cursor, percentiles, token_address)
    defi_addresses = get_all_defi_addresses(cursor)
    # defi_addresses = get_only_uniswap_addresses(cursor)

    # add exchange percentile
    percentiles['exchange'] = []
    for defi_address in defi_addresses.keys():
        wallet_percentiles[defi_address] = 'exchange'

    txns, block_times, price_grabber = load_data(cursor=cursor, token_address=token_address, latest_block=0)

    # get first hour
    current_hour = datetime.datetime.fromisoformat(txns[0][TransactionsColumns.timestamp][:-5] + '00:00')

    i = 0
    last_i = -1  # to avoid divide by zero error on first pass
    txn_i = 0
    print_interval = datetime.timedelta(days=7)
    print_time = current_hour

    txn_counts = {}

    # while not end of balances
    while txn_i < len(txns):
        if current_hour >= print_time:
            end = time.time()
            velocity = (i - last_i) / (end - start)
            estimate = str(datetime.timedelta(seconds=(len(txns) - i) / velocity))
            print(f'remaining transactions to process: {len(txns) - i}, velocity: {"%.4f" % velocity} elements/second,'
                  f' completion estimate: {estimate}')
            print(f'current_hour: {current_hour}')
            last_i = i
            print_time += print_interval
            start = time.time()
        i += 1

        # get one hour of balances_rows
        txn_i, to_process_rows = get_next_rows(i=txn_i, table_rows=txns,
                                               timestamp_column_num=TransactionsColumns.timestamp,
                                               before_timestamp=current_hour + granularity)

        # process changes to wallets map
        volume_totals = process_txns(wallets, to_process_rows, price_grabber, defi_addresses, txn_counts)

        # update percentiles
        update_balance_percentiles(percentiles, volume_totals, wallet_percentiles)
        timestamps.append(str(current_hour))

        current_hour += granularity

    # debug info ##############
    # get largest txn addresses
    sorted_txn_counts = [(txn_count.sent_count + txn_count.received_count, address, txn_count)
                         for address, txn_count in txn_counts.items()]
    sorted_txn_counts.sort(reverse=True)

    for i in range(10):
        print(sorted_txn_counts[i])
        largest_received_from = [(count, address) for address, count in sorted_txn_counts[i][2].received_from.items()]
        largest_received_from.sort(reverse=True)
        largest_sent_to = [(count, address) for address, count in sorted_txn_counts[i][2].sent_to.items()]
        largest_sent_to.sort(reverse=True)
        print('sorted')

    print(f'completed generate_percentiles: {time.time() - start}')
    return timestamps, percentiles


def process_txns(wallets, to_process_rows, price_grabber, defi_addresses, txn_count):
    # wallet_address: volume
    volume_totals = {}

    for txn in to_process_rows:
        block = txn[TransactionsColumns.block_number]
        sender = txn[TransactionsColumns.sender]
        recipient = txn[TransactionsColumns.recipient]
        value = txn[TransactionsColumns.value]
        price = price_grabber.get_price_from_block(block)

        if value_too_small(value, wallets, sender):
            continue

        # debug data
        if sender not in txn_count:
            txn_count[sender] = TxnCounts()
        if recipient not in txn_count:
            txn_count[recipient] = TxnCounts()
        txn_count[sender].sent_count += 1
        txn_count[sender].sent_to[recipient] += 1
        txn_count[recipient].received_count += 1
        txn_count[recipient].received_from[sender] += 1

        update_sender(wallets, sender, value, price, txn)
        update_recipient(wallets, recipient, value, price)

        # only count defi traffic
        if sender in defi_addresses or recipient in defi_addresses:
            update_volume(volume_totals, sender, value * -1)  # negative value as the sender is selling
            update_volume(volume_totals, recipient, value)

    return volume_totals


def get_all_defi_addresses(cursor):
    known_addresses_rows = get_all_known_addresses(cursor)

    defi_addresses = {}
    for row in known_addresses_rows:
        defi_addresses[row[KnownAddressesColumns.wallet_address]] = row

    return defi_addresses


def get_only_uniswap_addresses(cursor):
    known_addresses_rows = get_all_known_addresses(cursor)

    defi_addresses = {}
    for row in known_addresses_rows:
        if (row[KnownAddressesColumns.wallet_name] is not None
                and 'uniswap' in row[KnownAddressesColumns.wallet_name].lower()):
            defi_addresses[row[KnownAddressesColumns.wallet_address]] = row

    return defi_addresses


if __name__ == "__main__":
    run_transaction_rows()
