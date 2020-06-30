import ast
import pandas as pd
import numpy as np
import requests
# from pandas import DataFrame
from kafka import KafkaConsumer
# from joblib import load
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
import paho.mqtt.client as mqtt
import sys
import json
import time
import logging
from math import sqrt

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
            misc_config = config.get("misc", {})
            kafka_config = config.get("kafka", {})

            logger.debug("http config: {0}\nkafka config: {1}\nmisc config: {2}".format(http_config, kafka_config, misc_config))
            return [http_config, kafka_config, misc_config]
    except IOError as error:
        logger.error("Error opening config file '%s'" % config_path, error)
    except Exception as e:
        logger.error("Error {0}".format(e))


def generate(host, port, token, data, interval_ms, verbose):
    interval_secs = interval_ms / 1000.0
    headers = {'Content-Type': 'application/json', }
    # payload = data.to_json(orient='table')
    # print(data)
    payload = data.to_json(orient="records")
    if verbose:
        logger.debug(payload)
    # print(type(payload))
    print(json.loads(payload)[0])
    payload1 = json.loads(payload)[0]
    url_post = 'http://{0}:{1}/api/v1/{2}/telemetry'.format(host, port, token)
    requests.post(url_post, headers=headers, data=payload1)
    # print(payload)
    # mqttc.publish(topic, payload)
    time.sleep(interval_secs)


if __name__ == "__main__":
    logger = get_logger(__name__)
    http_config, kafka_config, misc_config = {}, {}, {}
    if len(sys.argv) < 2:
        logger.debug("Please provide configuration file path.")
    else:
        http_config, kafka_config, misc_config = read_config(sys.argv[1])

    # Load saved 'scaler' and 'Model' files
    scaler = joblib.load("../ml_service/scaler.joblib")
    model = joblib.load("../ml_service/model.joblib")
    # filtercolumns = ["Timestamp"]

    consumer = KafkaConsumer(kafka_config.get("topic","sensors-u1m1"),
                             bootstrap_servers="{0}:{1}".format(kafka_config.get("host", "localhost"),
                                                                kafka_config.get("port", "9092")))

    for raw_message in consumer:
        message = ast.literal_eval(raw_message.value.decode('utf-8'))
        # print(message)

        # for col in filtercolumns:
            # del message[col]

        # for k,v in message.items():
            # print(v)
            # message[k] = float(v)

        test = pd.DataFrame(message, index=[0])
        # print(test.head())
        test.index = pd.to_datetime(test['Timestamp'], format="%d-%m-%Y %H:%M")
        # test = test.sort_index()
        test1 = test.drop(columns = ['Timestamp'])
        # print(test1.head())
        X_test = test1.copy()
        # X_test.drop(X_test.columns[[0]],axis=1,inplace=True)

        # normalize 'test' data between 0 and 1
        X_test = scaler.transform(X_test)

        # reshape inputs for LSTM [samples, timesteps, features]
        X_test1 = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        # print("Test data shape:", X_test.shape)

        # Calculate reconstruction loss on test set
        X_pred = model.predict(X_test1)

        X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
        X_pred = pd.DataFrame(X_pred, columns=test1.columns)
        # X_pred.index = test1.index

        scored = pd.DataFrame()
        Xtest = X_test1.reshape(X_test1.shape[0], X_test1.shape[2])
        scored['Loss_MAE'] = np.mean(np.abs(X_pred - Xtest), axis=1)
        scored['RMSE'] = np.mean(sqrt(mean_squared_error(Xtest,X_pred)))
        scored['threshold'] = 0.25  # Will be determined from model
        scored['Anomaly'] = scored['Loss_MAE'] > scored['threshold']
        # print(scored.head())
        # print(X_pred.head())
        # print(scored['Loss_MAE'][0])

        anomalies = scored[scored.Anomaly == True]
        # print(anomalies.head())

        # result = pd.merge(test1, X_pred, on='Timestamp')  # actual test + predicted
        # result_1 = pd.merge(scored, on='Timestamp')  # Loss_MAE + Threshold + Anomaly class(T/F)
        # scored.reset_index(drop=True,inplace=True)

        # print(scored.head())
        result_new = scored.astype({"Loss_MAE":float,"RMSE":float,"threshold":float,"Anomaly":bool})

        generate(http_config.get("host", "localhost"), http_config.get("port", 8083),
                 http_config.get("token", "KkFSogMUNDcKd5M30KzW"),
                 result_new, int(http_config.get("sleeptime", 5000)), misc_config.get("verbose", False))




