import configparser
import io
import os
import sys
import time
import traceback
from contextlib import contextmanager

from colorama import Fore, Style

GLOBAL_HIGHLIGHTING = True


def pretty_print_dict(d, float_precision=3, sort_keys=True, indent=0):
    color_map = {float: Fore.BLUE, int: Fore.GREEN, str: Fore.CYAN, list: Fore.MAGENTA, bool: Fore.YELLOW}

    def format_value(v, key):
        if isinstance(v, tuple) and all(isinstance(sub_v, (int, float)) for sub_v in v):
            return str(
                tuple(str(sub_v) if isinstance(sub_v, int) else f"{sub_v:.{float_precision}f}" for sub_v in v)
            ).replace("'", ""), type(v)
        elif isinstance(v, (list, tuple)):  # handle list/tuple before other types
            return str(type(v)(format_value(sub_v, key)[0] for sub_v in v)).replace("'", ""), type(v)
        elif "time" in key.lower() and isinstance(v, (int, float)):
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
    key_len_max = max((len(key) for key in d.keys()), default=0) + 1

    if sort_keys:
        d = dict(sorted(d.items()))

    for key, value in d.items():
        key = key.replace("_", " ").title()

        if isinstance(value, list):
            value = [colorize_value(*format_value(v, key)) for v in value]
            print("\t" * indent + f"{str(key).ljust(key_len_max)}: ", end="")
            for i, v in enumerate(value):
                # if i == 0:
                print(v, end=" ")
                # else:
                #     print(f"{' ' * (indent * 8 + key_len_max + 2)}{v}", end=" ")
            print()
        elif isinstance(value, dict):
            print("\t" * indent + f"{str(key).ljust(key_len_max)}:")
            pretty_print_dict(value, float_precision, sort_keys, indent + 1)
        else:
            value = colorize_value(*format_value(value, key))
            print("\t" * indent + f"{str(key).ljust(key_len_max)}: {value}")


def format_time(seconds):
    negative = False
    if seconds < 0:
        negative = True
        seconds = -seconds  # make it positive for calculation

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = ""

    if hours:
        time_str += f"{int(hours)} hrs., "
    if minutes:
        time_str += f"{int(minutes)} min., "
    time_str += f"{seconds:.2f} sec."

    if negative:
        time_str = "-" + time_str

    return time_str


def read_config():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config


# TODO Deze gebruiken om wat meer info te krijgen over de errors
class ErrorLogger:
    def __init__(self, func, log_dir="."):
        self.func = func
        self.log_dir = log_dir
        self.old_stdout = sys.stdout

    def __enter__(self):
        self.start_time = time.time()
        sys.stdout = self.captured_output = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_value, traceback_):
        sys.stdout = self.old_stdout
        if exc_type is not None:
            os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)
            log_file = os.path.join(
                self.log_dir,
                f"errors_{self.func.__name__}_{int(self.start_time)}.log",
            )
            with open(log_file, "w") as file:
                file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(self.start_time))}\n")
                file.write(f"Function: {self.func.__name__}\n")
                file.write("Print Output:\n")
                file.write(self.captured_output.getvalue())
                file.write("\n")
                file.write(f"Exception occurred: {exc_type.__name__}: {str(exc_value)}\n")
                file.write("Traceback (most recent call last):\n")
                traceback.print_tb(traceback_, file=file)


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
            # add a timestamp to the error message
            timestamp = f"Error timestamp: {time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())}\n"
            error_message = "*-*-*-" * 80 + "\n" + timestamp + "\n" + str(e) + "\n" + traceback.format_exc()
            log_file = f"log/{func.__name__}_error.log"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "a") as f:
                f.write(f"An exception occurred in {func.__name__}: {error_message}\n")
            raise e  # re-throw the exception after logging

    return wrapper


def abbreviate(word):
    vowels = "aeiouAEIOU"
    if word[0] in vowels:
        return word[0] + "".join([letter for letter in word[1:] if letter not in vowels])
    else:
        return "".join([letter for letter in word if letter not in vowels])
