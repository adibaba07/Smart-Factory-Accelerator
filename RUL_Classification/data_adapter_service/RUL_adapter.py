import paho.mqtt.client as mqtt
from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable
import time
import json
import sys
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


## KAFKA
def send_message_to_kafka(message):
    """
    Sends message to kafka. Async by default.
    :param message:
    :return:
    """
    print(message)
    producer.send(kafka_config.get("topic", "RULsensors"), message)


## MQTT
# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    """
    Callback for connect to MQTT event
    :param client:
    :param userdata:
    :param flags:
    :param rc:
    :return: None
    """
    client.subscribe(mqtt_config.get("topic", "rawmessage1"))

def on_disconnect(client, user_data, rc):
    """
    Callback for disconnect event
    :param client:
    :param user_data:
    :param rc:
    :return: None
    """

    print("""Disconnected
    client: %s
    user_data: %s
    rc: %s
    """ % (client, user_data, rc))


def on_message(client, userdata, msg):
    """
    The callback for when a PUBLISH message is received from the server.
    :param client:
    :param userdata:
    :param msg:
    :return: None
    """
    send_message_to_kafka(msg.payload)


def mqtt_to_kafka_run():
    """Pick messages off MQTT queue and put them on Kafka"""

    client_name = "smart_factory_connector_RUL"
    client = mqtt.Client(client_id=client_name)

    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    # TODO: the queue address should be an env var
    client.connect(mqtt_config.get("host", "localhost"), mqtt_config.get("port", 1883), 60)

    client.loop_forever()


def read_config(config_path):
    """main entry point, load and validate config and call generate"""
    try:
        with open(config_path) as handle:
            config = json.load(handle)
            mqtt_config = config.get("mqtt", {})
            kafka_config = config.get("kafka", {})

            logger.debug("mqtt config: {0}\nkafka config: {1}".format(mqtt_config, kafka_config))
            return [mqtt_config, kafka_config]
    except IOError as error:
        logger.error("Error opening config file '%s'" % config_path, error)
    except Exception as e:
        logger.error("Error {0}".format(e))


if __name__ == '__main__':
    logger = get_logger(__name__)
    mqtt_config, kafka_config = {}, {}
    if len(sys.argv) < 2:
        logger.error("Please provide configuration file path.")
    else:
        mqtt_config, kafka_config = read_config(sys.argv[1])
    attempts = 0
    while attempts < 10:
        try:
            producer = KafkaProducer(bootstrap_servers="{0}:{1}".format(kafka_config.get("host", "localhost"),
                                                                        kafka_config.get("port", "9092")))
            mqtt_to_kafka_run()
        except NoBrokersAvailable:
            logger.error("No Brokers. Attempt %s" % attempts)
            attempts = attempts + 1
            time.sleep(2)