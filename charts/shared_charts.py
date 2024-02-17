import time
import plotly.graph_objects as go


def add_price_trace(prices, fig):
    # 0 price, 1 timestamp
    prices_y = [row[0] for row in prices]
    prices_timestamp = [row[1] for row in prices]

    fig.add_trace(
        go.Scatter(x=prices_timestamp, y=prices_y, name="price"),
        secondary_y=True,
    )


def load_prices(cursor, token_address):
    start_time = time.time()
    query = f"""
        SELECT 
            price,
            timestamp
        FROM 
            prices
        WHERE
            token_address='{token_address}'
        ORDER BY
            timestamp;
        """
    cursor.execute(query)
    prices = cursor.fetchall()
    print("Total prices rows are:  ", len(prices))
    print(f'load_prices time: {time.time() - start_time}')
    return prices