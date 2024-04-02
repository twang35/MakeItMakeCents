import time
import plotly.graph_objects as go

from database import *


def add_price_trace(prices, fig, left_offset=0, row=1):
    # 0 price, 1 timestamp
    prices_y = [row[0] for row in prices][left_offset:]
    prices_timestamp = [row[1] for row in prices][left_offset:]

    fig.add_trace(
        go.Scatter(x=prices_timestamp, y=prices_y, name="price", line=dict(color='black')),
        secondary_y=True,
        row=row,
        col=1
    )


def load_prices(cursor, token_address, after_timestamp=None, before_timestamp=None):
    start_time = time.time()
    after_query = '' if after_timestamp is None else f' AND timestamp >= "{after_timestamp}"'
    before_query = '' if before_timestamp is None else f' AND timestamp < "{before_timestamp}"'
    query = f'''
        SELECT 
            price,
            timestamp
        FROM 
            prices
        WHERE
            token_address="{token_address}"
        {after_query}
        {before_query}
        ORDER BY
            timestamp;
        '''
    cursor.execute(query)
    prices = cursor.fetchall()
    print("Total prices rows are:  ", len(prices))
    print(f'load_prices time: {time.time() - start_time}')
    return prices


class TimestampData:
    def __init__(self, data, timestamps):
        self.data = data
        self.timestamps = timestamps
        self.first_hour = datetime.datetime.fromisoformat(timestamps[0][:-5] + '00:00') \
            if type(timestamps[0]) is str else timestamps[0]


def load_structured_prices(cursor, token_address, after_timestamp=None, before_timestamp=None):
    prices = load_prices(cursor, token_address, after_timestamp, before_timestamp)
    return TimestampData([row[0] for row in prices], [row[1] for row in prices])


def load_structured_test_data_prices(cursor, data_type):
    query = f"""
            SELECT 
                *
            FROM 
                test_data
            WHERE
                type='{data_type}'
            ORDER BY
                timestamp;
            """
    cursor.execute(query)
    prices = cursor.fetchall()
    return TimestampData([row[TestDataColumns.price] for row in prices],
                         [row[TestDataColumns.timestamp] for row in prices])


def get_price_map(cursor, token_address):
    query = f"""
            SELECT * FROM prices
            where token_address='{token_address}'
            ORDER by timestamp;
            """
    cursor.execute(query)
    prices_rows = cursor.fetchall()
    print("Total prices rows are: ", len(prices_rows))
    return to_prices_map(prices_rows)


def to_prices_map(prices_rows):
    time_to_price = {}
    for row in prices_rows:
        time_to_price[row[PricesColumns.timestamp]] = row[PricesColumns.price]

    cur_time = datetime.datetime.fromisoformat(prices_rows[0][PricesColumns.timestamp])
    # some charts grab the price at the end of the hour. If the end of the hour is in the future, there is no price.
    # Adding extra 1 day buffer in case of day granularity for future charts.
    last_time = datetime.datetime.utcnow() + datetime.timedelta(days=1)

    last_price = time_to_price[str(cur_time)]

    while cur_time < last_time:
        # replace missing times with last price
        if str(cur_time) not in time_to_price:
            time_to_price[str(cur_time)] = last_price
        else:
            last_price = time_to_price[str(cur_time)]
        cur_time += datetime.timedelta(seconds=60)

    # time_to_price map, first timestamp that has price data
    return time_to_price, prices_rows[0][PricesColumns.timestamp]


def get_price(time_to_price, first_price_timestamp, timestamp):
    return time_to_price[timestamp] if timestamp >= first_price_timestamp else 0


def load_transactions_table(cursor, token_address):
    query = f"""
        SELECT * FROM transactions
        WHERE token_address = '{token_address}'
        ORDER BY block_number, log_index;
        """
    cursor.execute(query)
    rows = cursor.fetchall()
    print("Total balances rows are: ", len(rows))
    return rows


def load_balances_table(cursor, token_address):
    query = f"""
        SELECT * FROM balances
        WHERE token_address = '{token_address}'
        ORDER BY block_number;
        """
    cursor.execute(query)
    rows = cursor.fetchall()
    print("Total balances rows are: ", len(rows))
    return rows


# essentially a paginated way to get the next hour of balances_rows changes
def get_next_rows(i, table_rows, timestamp_column_num, before_timestamp):
    before_str = str(before_timestamp)
    output = []

    while i < len(table_rows) and table_rows[i][timestamp_column_num] < before_str:
        output.append(table_rows[i])
        i += 1

    return i, output


def generate_wallet_percentiles(cursor, percentiles, token_address):
    wallet_percentiles = {}

    # load the existing wallets from the balances table
    balances_rows = get_largest_alltime_wallet_balances(cursor, token_address)

    # balances: [(balance, wallet_address), ()...]
    balances = [(row[BalancesColumns.balance], row[BalancesColumns.wallet_address]) for row in balances_rows]
    balances.sort(reverse=True)

    percentile_debug = {}
    i = 0
    percentile_i = -1
    for key in percentiles.keys():
        if key not in percentile_debug:
            percentile_debug[key] = []
        # update percentile_i to the location of the end of the percentile key
        if i > percentile_i:
            percentile_i = (key / 100 * len(balances)) - 1

        # add the percentile key mapping to all addresses under that percentile_i
        while i < len(balances) and i <= percentile_i:
            percentile_debug[key].append(balances[i][1])
            wallet_percentiles[balances[i][1]] = key
            i += 1

    return wallet_percentiles, percentile_debug


def get_percentile_addresses(cursor, token_address, percentile):
    percentiles = {  # balances are bucketed down
        0.1: [],
        1: [],
        10: [],
        50: [],
        100: [],
    }
    _, wallets_by_percentile = generate_wallet_percentiles(cursor, percentiles, token_address)

    return wallets_by_percentile[percentile]
