from database import *
import json
from snippets.manual_addresses import address_types


def run_known_addresses():
    create_known_addresses_table()
    write_addresses()
    read_addresses('binance')


def read_addresses(search_term):
    conn = create_connection()

    known_addresses = get_all_known_addresses(conn)

    for address_data in known_addresses:
        print(address_data)


def write_addresses():
    search_term = 'binance'

    # create_known_addresses_table()
    conn = create_connection()

    # read from combinedAllLabels.json
    f = open('snippets/combinedAllLabels.json')

    # returns JSON object as a dictionary
    etherscan_data = json.load(f)

    for address, attributes in etherscan_data.items():
        name = attributes.get('name')
        name = None if (name is None or name == "") else name.lower()
        if ((name is not None and search_term in name)
                or any(search_term in item.lower() for item in attributes.get('labels'))):
            # wallet_address, name_tag, type, description, etherscan_tags
            row = (address, name, None, None, json.dumps(attributes.get('labels')))
            insert_known_addresses(conn, row)

    # save to grab all known of various exchanges and save them all to the database

    print('done')


run_known_addresses()
