# Define the network client
from xrpl.clients import JsonRpcClient
from xrpl.wallet import generate_faucet_wallet
from xrpl.core import addresscodec
from xrpl.models.requests.account_info import AccountInfo
import json
from xrpl.wallet import Wallet
from xrpl.constants import CryptoAlgorithm
from xrpl.models.transactions import Payment
from xrpl.utils import xrp_to_drops
from xrpl.transaction import submit_and_wait


FULL_HISTORY_SERVER_URL = "https://s2.ripple.com:51234/"
TESTNET_JSON_RPC_URL = "https://s.altnet.rippletest.net:51234/"
client = JsonRpcClient(TESTNET_JSON_RPC_URL)

"""
# Create a wallet using the testnet faucet:
# https://xrpl.org/xrp-testnet-faucet.html
test_wallet = generate_faucet_wallet(client, debug=True)

print(test_wallet)

# Create an account str from the wallet
test_account = test_wallet.address

print(test_account)
"""

"""
public_key: EDB261634CB9FB648FD753F1B2DE681C8BBFCC86F8A2E36EE4C43CA2A0303D1D4F
private_key: -HIDDEN-
classic_address: rDm5BAjmRWe9vWnki7MbTbyyEbRWwnD5W3
"""

classic_address = "r4HhGFtP7MnTKrL6tn8MiKnjgbMcpyqDA1"

# Testnet Credentials ----------------------------------------------------------
test_wallet = Wallet.from_seed(seed="sEdVNPcisD77k8mDy4Cx4j2YT6VrGrk", algorithm=CryptoAlgorithm.ED25519)
print(f"test_wallet.address: {test_wallet.address}")  # "r4HhGFtP7MnTKrL6tn8MiKnjgbMcpyqDA1"


# Prepare payment
my_tx_payment = Payment(
    account=classic_address,
    amount=xrp_to_drops(22),
    destination="rPT1Sjq2YGrBMTttX4GZHjKu9dyfzbpAYe",
)

# Sign and submit the transaction
tx_response = submit_and_wait(my_tx_payment, client, test_wallet)

test_xaddress = addresscodec.classic_address_to_xaddress(classic_address, tag=12345, is_test_network=True)
print("Classic address: ", classic_address)
print("X-address: ", test_xaddress)

# Look up info about your account
acct_info = AccountInfo(
    account=classic_address,
    ledger_index="validated",
    strict=True,
)
response = client.request(acct_info)
result = response.result
print("response.status: ", response.status)
print(json.dumps(response.result, indent=4, sort_keys=True))
print("done")
