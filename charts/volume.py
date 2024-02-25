from dataclasses import dataclass

from plotly.subplots import make_subplots
from charts.shared_charts import *
from database import *


# get buy and sell activity for all users
# then split by all-time high balance percentile
# then add exchange addresses as a category for balance percentile
def volume():
    print('volume')

    conn = create_connection()
    cursor = conn.cursor()

    # go through balances and find diffs per hour
    # load balances
    token_address = token_addresses['pepefork']
    balances_rows = load_balances_table(cursor, token_address)

    # calculate sum of diffs and add to chart output


    # generate chart


if __name__ == "__main__":
    volume()
