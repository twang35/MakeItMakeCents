# Using graph_objects
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from database import *
import time

def main():
    conn = create_connection()

    balances = load_data(conn, '0xbd513f67ed9bccc8364176e8b97bb93a5030e777')
    print(len(balances))

    # 0 wallet_address, 1 timestamp, 2 balance, 3 total_cost_basis, 4 remaining_cost_basis, 5 realized_gains
    df = pd.DataFrame(dict(
        x = [row[1] for row in balances],
        y = [row[5] for row in balances]
    ))

    fig = px.line(df, x='x', y='y')
    fig.show()


def load_data(conn, wallet_address):
    cursor = conn.cursor()

    start_time = time.time()
    query = f"""
        SELECT 
            wallet_address, 
            timestamp,
            balance, 
            total_cost_basis,
            remaining_cost_basis,
            realized_gains
        FROM 
            balances
        WHERE 
            wallet_address = '{wallet_address}'
        ORDER BY
            timestamp;
        """
    cursor.execute(query)
    balances = cursor.fetchall()
    print("Total unique balances rows are:  ", len(balances))
    print(f'query time: {time.time() - start_time}')
    return balances


main()
