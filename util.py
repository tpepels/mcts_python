import logging
import os
import sys
import traceback

import configparser
from contextlib import contextmanager


from colorama import Fore, Style


GLOBAL_HIGHLIGHTING = True


def pretty_print_dict(d, float_precision=3, sort_keys=True, indent=0):
    color_map = {float: Fore.BLUE, int: Fore.GREEN, str: Fore.CYAN, list: Fore.MAGENTA, bool: Fore.YELLOW}

    # Function to format different types of values
    def format_value(v, key):
        if "time" in key.lower() and isinstance(v, (int, float)):
            return format_time(v), type(v)
        elif isinstance(v, bool):  # handle bool before int
            return str(v), type(v)
        elif isinstance(v, float):
            return f"{v:.{float_precision}f}", type(v)
        elif isinstance(v, int):
            return f"{v:,}", type(v)
        else:
            return str(v), type(v)

    # Function to colorize different types of values
    def colorize_value(v, type_v, bright=True):
        if GLOBAL_HIGHLIGHTING:
            color = color_map.get(type_v, Fore.RESET)
            if bright:
                return f"{color}{Style.BRIGHT}{v}{Style.RESET_ALL}"
            else:
                return f"{color}{v}{Style.RESET_ALL}"
        else:
            return v

    # Calculate longest key length for proper indentation
    key_len_max = max(len(key) for key in d.keys()) + 1

    if sort_keys:
        d = dict(sorted(d.items()))

    for key, value in d.items():
        key = key.replace("_", " ").title()

        if isinstance(value, list):
            value = [colorize_value(*format_value(v, key)) for v in value]
            print("\t" * indent + f"{str(key).ljust(key_len_max)}: ", end="")
            for i, v in enumerate(value):
                if i == 0:
                    print(v)
                else:
                    print(f"{' ' * (indent * 8 + key_len_max + 2)}{v}", end="\n")
        elif isinstance(value, dict):
            print("\t" * indent + f"{str(key).ljust(key_len_max)}:")
            pretty_print_dict(value, float_precision, sort_keys, indent + 1)
        else:
            value = colorize_value(*format_value(value, key))
            print("\t" * indent + f"{str(key).ljust(key_len_max)}: {value}")


def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = ""

    if hours:
        time_str += f"{int(hours)} hours, "
    if minutes:
        time_str += f"{int(minutes)} minutes, "
    time_str += f"{seconds:.2f} seconds"

    return time_str


def read_config():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config


class PrintLogger:
    def __init__(self, logfile):
        self.logfile = logfile
        self._original_stdout = sys.stdout
        self._log_file = None

    def __enter__(self):
        self._log_file = open(self.logfile, "a")
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        self._log_file.close()

    def write(self, message):
        if message.rstrip() != "":
            self._log_file.write(message.rstrip() + "\n")

    def flush(self):
        self._log_file.flush()


@contextmanager
def redirect_print_to_log(log_file):
    global GLOBAL_HIGHLIGHTING
    # Disable highlighting so we don't write color codes to the log.
    GLOBAL_HIGHLIGHTING = False

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = PrintLogger(log_file)
    try:
        yield logger.__enter__()
    finally:
        logger.__exit__(None, None, None)


def log_exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = str(e) + "\n" + traceback.format_exc()
            log_file = f"log/{func.__name__}_error.log"
            with open(log_file, "a") as f:
                f.write(f"An exception occurred in {func.__name__}: {error_message}\n")
            raise e  # re-throw the exception after logging

    return wrapper
