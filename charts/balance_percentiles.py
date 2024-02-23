from dataclasses import dataclass

from plotly.subplots import make_subplots
from charts.shared_charts import *
from database import *


def balance_percentiles():
    print('balance_percentiles')
    fig = make_subplots(
        rows=5, cols=2,
        specs=[[{"rowspan": 2, "colspan": 2}, None],
               [None, None],
               [{}, {"rowspan": 2}],
               [{}, None],
               [{}, {}]],
        print_grid=True)

    fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], name="(1,1)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], name="(3,1)"), row=3, col=1)
    fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], name="(3,2)"), row=3, col=2)
    fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], name="(4,1)"), row=4, col=1)
    fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], name="(5,1)"), row=5, col=1)
    fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], name="(5,2)"), row=5, col=2)

    fig.update_layout(title_text="specs examples")
    fig.update_layout(hovermode="x unified")
    fig.show()
    print(BalancesColumns.wallet_address)


if __name__ == "__main__":
    balance_percentiles()
