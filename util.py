import logging
import os
import sys
import traceback

import configparser
from contextlib import contextmanager


def read_config():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config


class PrintLogger:
    def __init__(self, logfile):
        self.logfile = logfile
        self._original_stdout = sys.stdout

    def __enter__(self):
        logging.basicConfig(filename=self.logfile, level=logging.INFO)
        self._logger = logging.getLogger("")
        self._logger.setLevel(logging.INFO)
        sys.stdout = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

    def write(self, message):
        if message.rstrip() != "":
            self._logger.info(message.rstrip())

    def flush(self):
        for handler in self._logger.handlers:
            handler.flush()


@contextmanager
def redirect_print_to_log(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = PrintLogger(log_file)
    try:
        yield logger.__enter__()
    finally:
        logger.__exit__(None, None, None)


logging.basicConfig(
    filename="log/error.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.ERROR,
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
