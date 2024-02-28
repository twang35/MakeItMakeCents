from database import *
import json


def known_addresses():
    search_term = 'binance'

    # create_known_addresses_table()
    conn = create_connection()

    # read from combinedAllLabels.json
    f = open('snippets/combinedAllLabels.json')

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    for address, attributes in data.items():
        name = attributes.get("name")
        name = name.lower() if name is not None else name
        if search_term in name or any(search_term in item.lower() for item in attributes.get('labels')):
            # wallet_address, name_tag, type, description, etherscan_tags
            row = (address, attributes.get("name"), None, None, json.dumps(attributes.get("labels")))
            insert_known_addresses(conn, row)

    # save to grab all known of various exchanges and save them all to the database

    print('done')


known_addresses()
