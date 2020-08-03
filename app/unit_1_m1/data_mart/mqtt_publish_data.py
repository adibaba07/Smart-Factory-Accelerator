"""A simple vibrational sensor data publisher that sends to an MQTT broker (not ThingsBoard) via paho"""

import sys
import json
import time

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
    # mqttc.username_pw_set(token)
    mqttc.connect(host, port, keepalive=60)

    interval_secs = interval_ms / 1000.0

    def iter_json(x):
        payload = x.to_json()
        # payload = json.dumps(x)
        if verbose:
            logger.debug(payload)
        print(payload)
        mqttc.publish(topic, payload)

    while True:
        grouped_data = data.groupby("Timestamp")
        for timestamp, values in grouped_data:
            df = grouped_data.get_group(timestamp)
            df.apply(iter_json,axis=1)
            time.sleep(interval_secs)


def loaddata(file_path):
    data = pd.read_csv(file_path)
    # test = data['2004-02-15 22:52:39':]  # sending test data to broker as training data is already being used by model
    data.drop(data.columns[[]],axis=1, inplace=True)
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
        # token = mqtt_config.get("token", "LveCEy1eQ5sOfrV0YGLw")
        # topic = mqtt_config.get("topic", "actual_data")
        topic = mqtt_config.get("topic", "sensors-u1-m1")

        file_path = data_config.get("file_path")
        # columns = data_config.get("columns").split(",")

        # token needs to be added in generate function if ThingsBoard is to be included
        generate(host, port, topic, loaddata(file_path), interval_ms, verbose)


if __name__ == '__main__':
    logger = get_logger(__name__)
    if len(sys.argv) < 2:
        logger.error("Please provide configuration file path.")
    else:
        main(sys.argv[1])
