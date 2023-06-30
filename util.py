import logging
import os
import sys
import traceback

import configparser
from contextlib import contextmanager


from colorama import Fore, Style


def pretty_print_dict(d, float_precision=3, sort_keys=True, indent=0):
    print()
    color_map = {float: Fore.BLUE, int: Fore.GREEN, str: Fore.CYAN, list: Fore.MAGENTA, bool: Fore.YELLOW}
    # Calculate longest key length for proper indentation
    key_len_max = max(len(key) for key in d.keys()) + 1

    if sort_keys:
        d = dict(sorted(d.items()))

    for key, value in d.items():
        key = key.replace("_", " ").title()

        # Apply color based on the value's type
        color = color_map.get(type(value), Fore.RESET)

        if isinstance(value, float):
            value = f"{value:.{float_precision}f}"
        if isinstance(value, list):
            print(
                "\t" * indent + f"{str(key).ljust(key_len_max)}: ",
                end="",
            )
            for i, item in enumerate(value):
                if isinstance(item, float):
                    item = f"{item:.{float_precision}f}"
                if i == 0:
                    print(f"{color}{Style.BRIGHT}{value[0]}{Style.RESET_ALL}")
                else:
                    print(
                        f"{' ' * (indent * 8 + key_len_max + 2)}{color}{Style.BRIGHT}{str(item)}{Style.RESET_ALL}",
                        end="\n",
                    )
        elif isinstance(value, dict):
            print("\t" * indent + f"{str(key).ljust(key_len_max)}:")
            pretty_print_dict(value, float_precision, sort_keys, indent + 1)
        else:
            print(
                "\t" * indent
                + f"{str(key).ljust(key_len_max)}: {color}{Style.BRIGHT}{str(value)}{Style.RESET_ALL}"
            )


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
