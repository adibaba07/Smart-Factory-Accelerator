import sys
import json
import time
import numpy as np
import pandas as pd
import logging
import requests


def get_logger(logname):
    # Create and configure logger
    logging.basicConfig(filename='app.log',
                        filemode='a',
                        format='%(name)s: %(levelname)s - %(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')

    # Creating an object
    logger = logging.getLogger(logname)

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)
    return logger


def generate(host, port, token, data, interval_ms, verbose):

    interval_secs = interval_ms / 1000.0

    def iter_json(x):
        payload = x.to_json()
        # payload = json.dumps(x)
        headers = {'Content-Type': 'application/json', }
        if verbose:
            logger.debug(payload)
        url_post = 'http://{0}:{1}/api/v1/{2}/telemetry'.format(host, port, token)
        requests.post(url_post, headers=headers, data=payload)
        time.sleep(interval_secs)
        print(payload)

    while True:
        data.apply(iter_json, axis=1)
        # time.sleep(interval_secs)
        # grouped_data = data.groupby("Timestamp")
        # for timestamp, values in grouped_data:
        #     df = grouped_data.get_group(timestamp)
        #     df.apply(iter_json,axis=1)
        #     time.sleep(interval_secs)


def loaddata(filepath1, filepath2, keys):
    drop_cols = True
    cols_drop = ['op_setting3', 's1','s5','s6','s10','s14','s16','s18','s19']
    data = pd.read_csv(filepath1, sep=" ", header=None)
    data.drop(data.columns[[-1, -2]], axis=1, inplace=True)
    data.columns = keys

    if drop_cols:
        data = data.drop(cols_drop,axis=1)

    y_true = pd.read_csv(filepath2, sep=" ", header=None)
    y_true.drop(y_true.columns[[1]], axis=1, inplace=True)
    y_true.columns = ['life']
    y_true = y_true.set_index(y_true.index + 1)
    y_true['max'] = data.groupby('unit')['cycles'].max() + y_true['life']
    y_true_new = [y_true['max'][i] for i in data.unit]
    truth_df = pd.DataFrame(y_true_new, columns=['max_total'])
    data['RUL'] = truth_df['max_total'] - data['cycles']
    data['max'] = truth_df['max_total']

    # ADD NEW LABEL TEST
    w1 = 35
    w0 = 25
    data['label1'] = np.where(data['RUL'] <= w1, 1, 0)
    data['label2'] = data['label1']
    data.loc[data['RUL'] <= w0, 'label2'] = 2
    data = data.dropna(axis=1)

    return data


def main(config_path):
    """main entry point, load and validate config and call generate"""
    with open(config_path) as handle:
        config = json.load(handle)
        http_config = config.get("http", {})
        misc_config = config.get("misc", {})
        data_config = config.get("data", {})

        logger.debug("http config: {0}\nmisc config: {1}\ndata config: {2}".format(http_config, misc_config, data_config))

        interval_ms = misc_config.get("interval_ms", 5000)
        verbose = misc_config.get("verbose", False)

        host = http_config.get("host", "127.0.0.1")
        port = http_config.get("port", 8083)
        token = http_config.get("token", "5Ywk3eeSpXtYhsA0Dzuu")

        file_path1 = data_config.get("file_path1")
        file_path2 = data_config.get("file_path2")
        columns = data_config.get("columns").split(",")

        generate(host, port, token, loaddata(file_path1, file_path2, columns), interval_ms, verbose)


if __name__ == '__main__':
    logger = get_logger(__name__)
    if len(sys.argv) < 2:
        logger.error("Please provide configuration file path.")
    else:
        main(sys.argv[1])