import logging
import traceback

import configparser


def read_config():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config


logging.basicConfig(
    filename="error.log", filemode="w", format="%(name)s - %(levelname)s - %(message)s", level=logging.ERROR
)


def log_exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = str(e) + "\n" + traceback.format_exc()
            logging.error(f"An exception occurred in {func.__name__}: {error_message}")
            raise e  # re-throw the exception after logging

    return wrapper
