import logging
import sys
import json
import time
import requests
import random
import pandas as pd


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
        print(payload)

    while True:
        grouped_data = data.groupby("Timestamp")
        for timestamp, values in grouped_data:
            df = grouped_data.get_group(timestamp)
            df.apply(iter_json,axis=1)
            time.sleep(interval_secs)


def loaddata(file_path):
    data = pd.read_csv(file_path)
    data.drop(data.columns[[]],axis=1, inplace=True)
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
        token = http_config.get("token", "KkFSogMUNDcKd5M30KzW")

        file_path = data_config.get("file_path")
        # columns = data_config.get("columns").split(",")

        # token needs to be added in generate function if ThingsBoard is to be included
        generate(host, port, token, loaddata(file_path), interval_ms, verbose)


if __name__ == '__main__':
    logger = get_logger(__name__)
    if len(sys.argv) < 2:
        logger.error("Please provide configuration file path.")
    else:
        main(sys.argv[1])
