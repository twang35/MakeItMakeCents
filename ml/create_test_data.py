from database import *
import numpy as np


class TestDataTypes:
    # centered around 100 with 10 amplitude and 0 slope
    easy_horizontal = 'easy_horizontal'

π
def create_test_data():
    # create_test_data_table()

    cycles = 200                # how many sine cycles
    resolution = cycles * 10    # how many datapoints to generate

    length = np.pi * 2 * cycles
    wave = np.sin(np.arange(0, length, length / resolution))

    wave = [point * 10 + 100 + np.random.normal(0, 1) for point in wave]

    print(wave)

    save_to_test_data_table(wave, data_type=TestDataTypes.easy_horizontal)

    print('done create_test_data')


def save_to_test_data_table(wave, data_type, start_date='2024-01-02 03:00:00'):
    conn = create_connection()

    cur_date = datetime.datetime.fromisoformat(start_date)

    for point in wave:
        row = (point, data_type, str(cur_date))
        insert_test_data(conn, row)
        cur_date += datetime.timedelta(minutes=60)


if __name__ == "__main__":
    create_test_data()