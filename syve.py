import requests

with open('syve_key.txt', 'r') as f:
    syve_key = f.read()
url = f"https://api.syve.ai/v1/filter-api/token-balances?key={syve_key}&gte:block_number=16000000&lte:block_number=17000000&size=10&sort=desc"

response = requests.get(url)

print(response.text)