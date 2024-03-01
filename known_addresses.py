from database import *
import json
from snippets.manual_addresses import *


def run_known_addresses():
    # create_known_addresses_table()
    conn = create_connection()

    search_term = 'kucoin'
    # write_addresses(search_term)
    # search_addresses(search_term)
    save_hard_coded_wallets(conn)
    save_all_manual_addresses(conn)

    print('run_known_addresses completed')


def search_addresses(search_term):
    conn = create_connection()

    known_addresses = get_all_known_addresses(conn)

    found_addresses = [item for item in known_addresses
                       if (item[KnownAddressesColumns.wallet_name] is not None
                           and search_term in item[KnownAddressesColumns.wallet_name])
                       or (item[KnownAddressesColumns.etherscan_labels] is not None
                           and any(search_term in label.lower()
                                   for label in item[KnownAddressesColumns.etherscan_labels]))]

    for address in found_addresses:
        print(address)


def write_addresses(search_term):
    conn = create_connection()

    # read from combinedAllLabels.json
    f = open('snippets/combinedAllLabels.json')
    # returns JSON object as a dictionary
    etherscan_data = json.load(f)

    for address, attributes in etherscan_data.items():
        name = attributes.get('name')
        name = None if (name is None or name == "") else name
        if ((name is not None and search_term in name.lower())
                or any(search_term in item.lower() for item in attributes.get('labels'))):
            # wallet_address, name_tag, type, description, etherscan_tags
            row = (address, name, None, None, json.dumps(attributes.get('labels')))
            insert_known_addresses(conn, row)


def save_all_manual_addresses(conn):
    save_hard_coded_wallets(conn)

    # combinedAllLabels.json is from Jun 17, 2023
    etherscan_data = json.load(open('snippets/combinedAllLabels.json'))

    search_term_address_types = {}
    search_terms = []
    for search_term, address_type in address_types.items():
        search_terms.append(search_term)
        search_term_address_types[search_term] = address_type

    # slow algo time, search for all search_terms inside all etherscan_data items
    for address, attributes in etherscan_data.items():
        # save if search_term in name
        name = attributes.get('name')
        if name is not None:
            for search_term in search_terms:
                if search_term in name.lower():
                    write_to_known_addresses(address, attributes, search_term, search_term_address_types, conn)

        labels = attributes.get('labels')
        if labels is not None:
            for search_term in search_terms:
                if any(search_term in item.lower() for item in labels):
                    write_to_known_addresses(address, attributes, search_term, search_term_address_types, conn)


def write_to_known_addresses(address, attributes, search_term, search_term_address_types, conn):
    name = attributes.get('name')
    name = None if (name is None or name == "") else name
    # wallet_address, name_tag, type, description, etherscan_tags
    row = (
        address, name, json.dumps([search_term_address_types[search_term]]), None, json.dumps(attributes.get('labels')))
    insert_known_addresses(conn, row)


def save_hard_coded_wallets(conn):
    # save hard-coded wallet_details
    for row in wallet_details:
        insert_known_addresses(conn, row)


if __name__ == "__main__":
    run_known_addresses()
