import time
from time import sleep

from eth_typing import BlockNumber
from web3 import Web3, HTTPProvider, AsyncWeb3
from database import *
import datetime
from eth_abi import decode
from concurrent.futures import ThreadPoolExecutor, wait


def main():
    create_known_addresses_table()
    # read from combinedAllLabels.json

    # save to grab all known of various exchanges and save them all to the database

    print('done')


main()
