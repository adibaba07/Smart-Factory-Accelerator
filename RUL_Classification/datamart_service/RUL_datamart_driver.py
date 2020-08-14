"""A simple sensor data generator that sends to an MQTT broker via paho"""

import sys
import json
import time
import numpy as np
import paho.mqtt.client as mqtt

import pandas as pd
import logging


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


def generate(host, port, topic, data, interval_ms, verbose):
    mqttc = mqtt.Client()

    mqttc.connect(host, port)

    interval_secs = interval_ms / 1000.0

    def iter_json(x):
        payload = x.to_json()
        if verbose:
            logger.debug(payload)
        print(payload)
        mqttc.publish(topic, payload)
        # time.sleep(interval_secs)

    # while True:
    #     data.apply(iter_json, axis=1)
    while True:
        grouped_data = data.groupby("cycles")
        for cycle, values in grouped_data:
            # print(cycle)
            df = grouped_data.get_group(cycle).sample(frac=1).reset_index(drop=True)
            df.apply(iter_json, axis=1)
            time.sleep(interval_secs)

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
    w1 = 45
    w0 = 15
    data['label1'] = np.where(data['RUL'] <= w1, 1, 0)
    data['label2'] = data['label1']
    data.loc[data['RUL'] <= w0, 'label2'] = 2
    data = data.dropna(axis=1)

    return data


def main(config_path):
    """main entry point, load and validate config and call generate"""
    with open(config_path) as handle:
        config = json.load(handle)
        mqtt_config = config.get("mqtt", {})
        misc_config = config.get("misc", {})
        data_config = config.get("data", {})

        logger.debug("mqtt config: {0}\nmisc config: {1}\ndata config: {2}".format(mqtt_config, misc_config, data_config))

        interval_ms = misc_config.get("interval_ms", 5000)
        verbose = misc_config.get("verbose", False)

        host = mqtt_config.get("host", "localhost")
        port = mqtt_config.get("port", 1883)
        topic = mqtt_config.get("topic", "rawmessage1")

        file_path1 = data_config.get("file_path1")
        file_path2 = data_config.get("file_path2")
        columns = data_config.get("columns").split(",")

        generate(host, port, topic, loaddata(file_path1, file_path2, columns), interval_ms, verbose)


if __name__ == '__main__':
    logger = get_logger(__name__)
    if len(sys.argv) < 2:
        logger.error("Please provide configuration file path.")
    else:
        main(sys.argv[1])
