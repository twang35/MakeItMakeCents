from database import *


def run_db_cleanup():
    token = pepefork
    conn = create_connection()
    delete_rows(token, 19272196, conn, dry_run=True)
    print('run_db_cleanup completed')


def delete_rows(token, block_number, conn, dry_run=True):
    cursor = conn.cursor()
    query = f"""
            FROM balances
            WHERE token_address = '{token.address}'
            AND block_number > {block_number};
            """
    if dry_run:
        query = "SELECT COUNT(*) " + query
    else:
        query = "DELETE " + query
    cursor.execute(query)
    result = cursor.fetchall()
    conn.commit()
    print(result)


if __name__ == "__main__":
    run_db_cleanup()