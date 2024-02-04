import requests

with open('transpose_key.txt', 'r') as f:
    transpose_key = f.read()

url = "https://api.transpose.io/sql"
# sql_query = """
# WITH before_timestamp AS (
#     SELECT *
#     FROM ethereum.token_prices
#     WHERE token_address = '0x111111111117dc0aa78b770fa6a738034120c302'
#       AND timestamp < '2023-02-02 14:15:16'
#     ORDER BY timestamp DESC
#     LIMIT 1
# ),
# after_timestamp AS (
#     SELECT *
#     FROM ethereum.token_prices
#     WHERE token_address = '0x111111111117dc0aa78b770fa6a738034120c302'
#       AND timestamp >= '2023-02-02 14:15:16'
#     ORDER BY timestamp ASC
#     LIMIT 1
# )
# SELECT *
# FROM before_timestamp
# UNION ALL
# SELECT *
# FROM after_timestamp;"""
sql_query = """
SELECT *
FROM ethereum.token_prices
WHERE token_address = '0x111111111117dc0aa78b770fa6a738034120c302'
    AND block_number = 16541503"""

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

print(response.text)
print('Credits charged:', response.headers.get('X-Credits-Charged', None))
