from dataclasses import dataclass

from plotly.subplots import make_subplots
from charts.shared_charts import *
from database import *


def balance_percentiles():
    print('balance_percentiles')
    print(BalancesColumns.wallet_address)


if __name__ == "__main__":
    balance_percentiles()
