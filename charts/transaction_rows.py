from plotly.subplots import make_subplots
from charts.shared_charts import *
from balance import *


def run_transaction_rows():
    token = pepefork

    conn = create_connection()
    cursor = conn.cursor()

    # calculate balances using txns and produce chart output
    timestamps, percentiles = generate_exchange_volume(cursor, token.address)

    # generate hourly graph
    create_volume_graph(prices, percentiles, timestamps, token, left_offset=1)

    print('completed run_transaction_rows')


def generate_exchange_volume(cursor, token_address):
    print(f'running generate_volume')

    start = time.time()
    # wallet_address: [balance, total_cost_basis, remaining_cost_basis, realized_gains]
    wallets = {null_address: [0, 0, 0, 0]}
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

    txns, block_times, price_grabber = load_data(cursor=cursor, token_address=token_address, latest_block=0)

    # get first hour
    current_hour = datetime.datetime.fromisoformat(txns[0][TransactionsColumns.timestamp][:-5] + '00:00')

    start = time.time()
    i = 0
    last_i = 0
    txn_i = 0
    print_interval = datetime.timedelta(days=7)
    print_time = current_hour
    granularity = datetime.timedelta(minutes=120)

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
        txn_totals = process_txns(wallets, to_process_rows, token_address, price_grabber)

        # update percentiles
        update_balance_percentiles(percentiles, txn_totals, wallet_percentiles)
        timestamps.append(str(current_hour))

        current_hour += granularity

    print(f'completed generate_percentiles: {time.time() - start}')
    return timestamps, percentiles


def process_txns(wallet_balances, to_process_rows, price_grabber):
    totals = {}

    for txn in to_process_rows:
        block = txn[TransactionsColumns.block_number]
        sender = txn[TransactionsColumns.sender]
        recipient = txn[TransactionsColumns.recipient]
        value = txn[TransactionsColumns.value]
        price = price_grabber.get_price(block)

        if value_too_small(value, wallets, sender):
            continue

        update_sender(wallets, sender, value, price, txn)
        update_recipient(wallets, recipient, value, price)



def get_all_defi_addresses(cursor):
    known_addresses_rows = get_all_known_addresses(cursor)

    defi_addresses = {}
    for row in known_addresses_rows:
        defi_addresses[row[KnownAddressesColumns.wallet_address]] = row

    return defi_addresses


if __name__ == "__main__":
    run_transaction_rows()
