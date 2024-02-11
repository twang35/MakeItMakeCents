import requests

with open('syve_key.txt', 'r') as f:
    syve_key = f.read()
url = (f"https://api.syve.ai/v1/sql?key={syve_key}")
query = ("select * from eth_token_balances where token_address = '0x111111111117dc0aa78b770fa6a738034120c302' order by "
         "balance_token desc limit 50")
# query = "SELECT pool_address FROM eth_dex_trades WHERE token_address = '0x111111111117dc0aa78b770fa6a738034120c302' LIMIT 10"
data = {'query': query}

response = requests.post(url, json=data)


print(response.text)
