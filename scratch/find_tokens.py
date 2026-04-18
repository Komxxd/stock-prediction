import requests

url = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
print("Downloading master list...")
response = requests.get(url)
data = response.json()

targets = ["NIFTYBEES-EQ", "BANKBEES-EQ"]
results = {}

for item in data:
    if item['symbol'] in targets:
        results[item['symbol']] = item['token']

print("\nTokens Found:")
for symbol, token in results.items():
    print(f"{symbol}: {token}")
