import ast
import requests
import joblib
import time
# from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score

from kafka import KafkaConsumer
import pandas as pd
import numpy as np
# import paho.mqtt.client as mqtt
import sys
import json
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


def read_config(config_path):
    """main entry point, load and validate config and call generate"""
    try:
        with open(config_path) as handle:
            config = json.load(handle)
            http_config = config.get("http", {})
            kafka_config = config.get("kafka", {})

            logger.debug("http config: {0}\nkafka config: {1}".format(http_config, kafka_config))
            return [http_config, kafka_config]
    except IOError as error:
        logger.error("Error opening config file '%s'" % config_path, error)
    except Exception as e:
        logger.error("Error {0}".format(e))


# def scale(df):
    # return (df - df.min())/(df.max()-df.min())


def generate(host, port, token, data, interval_ms, verbose):
    # mqttc = mqtt.Client()

    # mqttc.connect(host, port)

    interval_secs = interval_ms / 1000.0
    headers = {'Content-Type': 'application/json', }
    payload = data.to_json(orient='records')
    if verbose:
        logger.debug(payload)
    print(payload)
    url_post = 'http://{0}:{1}/api/v1/{2}/telemetry'.format(host, port, token)
    requests.post(url_post, headers=headers, data=payload)

    # mqttc.publish(topic, payload)
    time.sleep(interval_secs)


if __name__=="__main__":
    logger = get_logger(__name__)
    http_config, kafka_config = {}, {}
    if len(sys.argv) < 2:
        logger.debug("Please provide configuration file path.")
    else:
        http_config, kafka_config = read_config(sys.argv[1])

    # scaler = joblib.load("scaler_svc.joblib")
    model = joblib.load("svc_rul.joblib")
    # filtercolumns = ['s1', 's5', 's6','s10', 's14', 's16', 's18', 's19', 'op_setting3']

    consumer = KafkaConsumer(kafka_config.get("topic", "RULsensors"),
                             bootstrap_servers="{0}:{1}".format(kafka_config.get("host", "localhost"),
                                                                kafka_config.get("port", "9092")))

    for raw_message in consumer:
        message = ast.literal_eval(raw_message.value.decode('utf-8'))
        # print(message)

        # for col in filtercolumns:
        #     del message[col]

        for k, v in message.items():
            message[k] = float(v)

        test = pd.DataFrame(message,index=[0])
        # print(test.shape)

        ntest = test.copy()
        # print(ntest.shape)

        features = ntest.columns.drop(['unit', 'cycles', 'RUL', 'max', 'label1', 'label2'])
        # ntest = ntest.dropna(axis=1, inplace=True)
        # print(ntest[features].shape)

        pred_test = model.predict(ntest[features])
        # print(pred_test)
        pred_test = np.where(pred_test > 0.5, 1, 0)
        # print('Accuracy: {}'.format(accuracy_score(ntest['label1'], pred_test)))

        # add results to original data
        test['label1_pred'] = pred_test

        test.drop(['label1_pred'], axis=1,inplace=True)

        result = test

        generate(http_config.get("host", "127.0.0.1"), http_config.get("port", 8083),
                 http_config.get("token", "zgbapyVBBeiqtIoLlZ1Z"), result, int(http_config.get("sleeptime", 5000)),
                 True)