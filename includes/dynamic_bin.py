# cython: language_level=3
import gc
from statistics import mean, median, mode, variance
import cython
from colorama import Fore, init
import numpy as np

init(autoreset=True)  # Automatically resets the color of each print statement


class DynamicBin:
    def __init__(self, num_bins):
        self.num_bins = num_bins
        self.clear()

    def clear(self):
        self.data = []
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self.bin_counts = [0] * self.num_bins  # Initialize with zeros
        self.bin_edges = []
        self.zero_count = 0
        gc.collect()

    def add_data(self, new_data):
        if new_data == 0:
            self.zero_count += 1

        self.data.append(new_data)
        self.min_val = min(self.min_val, new_data)
        self.max_val = max(self.max_val, new_data)

    @cython.locals(bin_width=cython.double, value=cython.double, i=cython.int, condition=cython.bint)
    def calculate_bins(self):
        if self.min_val == self.max_val:
            print("Warning: min_val and max_val are equal. Unable to calculate bins.")
            return

        bin_width = (self.max_val - self.min_val) / (self.num_bins)

        self.bin_edges = [self.min_val + i * bin_width for i in range(self.num_bins + 1)]
        # Reset counts
        self.bin_counts = [0] * self.num_bins

        for value in self.data:
            if value == 0:  # 0's are handled separately
                continue

            for i in range(len(self.bin_edges) - 1):
                if i == 0:  # First bin is inclusive on low side
                    condition = self.bin_edges[i] <= value < self.bin_edges[i + 1]
                elif i == len(self.bin_edges) - 2:  # Last bin is inclusive on both sides
                    condition = self.bin_edges[i] < value <= self.bin_edges[i + 1]
                else:
                    condition = self.bin_edges[i] <= value < self.bin_edges[i + 1]

                if condition:
                    self.bin_counts[i] += 1
                    break

    def get_bins(self) -> cython.list:
        self.calculate_bins()  # Calculate bins only when needed
        return self.bin_counts

    def print_bins(self) -> cython.void:
        self.calculate_bins()  # Calculate bins only when needed
        for i in range(self.bin_edges.size() - 1):
            print(f"{self.bin_edges[i]} - {self.bin_edges[i + 1]}: {self.bin_counts[i]}")

    @cython.locals(
        total_count=cython.double,
        d=cython.double,
        num_chars=cython.int,
        percentage=cython.str,
        lower_bound=cython.double,
        upper_bound=cython.double,
        i=cython.int,
        count=cython.int,
        map_key=cython.int,
        scale_factor=cython.int,
        bin_size=cython.double,
        density=list,
        max_density=cython.double,
        zero_density=cython.double,
        zero_percentage=cython.str,
        num_zero_chars=cython.int,
        printed_zero_row=cython.bint,  # bint for boolean
    )
    def plot_bin_counts(self, name: str):
        print(f"{Fore.YELLOW}Calculating bins for {name}, {len(self.data):,} data points")
        # Show the statistics
        print(f"{Fore.YELLOW}Min: {self.min_val:.2f} Max: {self.max_val:.2f}, {self.zero_count:,} zeros")
        self.calculate_bins()
        total_count = sum(self.bin_counts) + self.zero_count  # Adding zero_count to the total

        if total_count == 0:
            print(f"No data for {name}")
            # Print all fields to see why there is no data
            print(f"{len(self.data)=} {self.min_val=} {self.max_val=} {self.num_bins=} {self.bin_counts=}")
            return

        density = [count / total_count for count in self.bin_counts]
        max_density = max(density)
        max_density = max(max_density, (self.zero_count / total_count))
        scale_factor = 50
        bin_size = (self.max_val - self.min_val) / cython.cast(cython.double, self.num_bins)

        print(f"{Fore.GREEN}Density function for {name} (N: {total_count:,})")
        printed_zero_row = self.zero_count == 0  # Initialize flag

        # Determine when to print zero row
        print_zero_last = self.max_val <= 0
        print_zero_first = self.min_val >= 0

        zero_density = self.zero_count / total_count
        zero_percentage = "{:.2f}".format(zero_density * 100)
        num_zero_chars = int(zero_density * scale_factor / max_density)

        if print_zero_first and not printed_zero_row:
            print(
                f"{'[0]':>14}: {Fore.YELLOW}{'#' * num_zero_chars} {Fore.RESET}\t{Fore.CYAN}{self.zero_count:,} ({zero_percentage}%)"
            )
            printed_zero_row = True

        for i, d in enumerate(density):
            percentage = f"{d*100.:.2f}"
            num_chars = int(d * scale_factor / max_density)
            lower_bound = self.min_val + i * bin_size
            upper_bound = self.min_val + (i + 1) * bin_size
            if i == len(density) - 1:
                bin_range = f"[{lower_bound:.2f}, {upper_bound:.2f}]"
            else:
                bin_range = f"[{lower_bound:.2f}, {upper_bound:.2f})"
            last_bin = lower_bound
            color = Fore.RED if lower_bound < 0 else Fore.BLUE

            if (
                last_bin < 0
                and lower_bound > 0
                and not printed_zero_row
                and self.min_val <= 0 <= self.max_val
            ):
                print(
                    f"{'[0]':>14}: {Fore.YELLOW}{'#' * num_zero_chars} {Fore.RESET}\t{Fore.CYAN}{self.zero_count:,} ({zero_percentage}%)"
                )
                printed_zero_row = True

            print(
                f"{bin_range:>14}: {color}{'#' * num_chars} {Fore.RESET}\t{Fore.CYAN}{self.bin_counts[i]:,} ({percentage}%)"
            )

            if print_zero_last and not printed_zero_row:
                print(
                    f"{'[0]':>14}: {Fore.YELLOW}{'#' * num_zero_chars} {Fore.RESET}\t{Fore.CYAN}{self.zero_count:,} ({zero_percentage}%)"
                )

        print(f"{Fore.MAGENTA}--" * 30)

    @cython.locals(
        min_value=cython.double,
        max_value=cython.double,
        scale_factor_y=cython.double,
        data_length=cython.int,
        window_size=cython.int,
        i=cython.int,
        y=cython.int,
        avg=cython.double,
        plot_height=cython.int,
        plot_width=cython.int,
        color=cython.str,
        y_labels=list,
        max_label_length=cython.int,
        label_str=cython.str,
        stats_dict=cython.dict,
        row=cython.list,
        key=cython.str,
        value=cython.double,
        averaged_data_length=cython.int,
        name=cython.str,
        median=cython.bint,
    )
    def plot_time_series(self, name, plot_width=50, plot_height=20, median=0):
        data_length = len(self.data)
        if data_length <= 1:
            print("No data to plot, data length <= 1")
            return
        # Calculate the moving average
        window_size = int(data_length / plot_width)
        # Pre-calculate the length of the averaged_data list
        averaged_data_length = (data_length - window_size + 1) // window_size
        # Pre-allocate memory for averaged_data
        averaged_data = [0] * (averaged_data_length + 1)

        if not median:
            for i in range(0, (data_length - window_size + 1), window_size):
                window = self.data[i : i + window_size]
                averaged_data[i // window_size] = (
                    sum(window) / window_size
                )  # Since window_size = len(window), we can just use window_size directly
        else:
            for i in range(0, (data_length - window_size + 1), window_size):
                window = self.data[i : i + window_size]
                median_value = np.median(window)  # Using numpy's median function for efficiency
                averaged_data[i // window_size] = median_value  # Storing the median value instead of the mean

        # min_value = min(averaged_data)
        # max_value = max(averaged_data)

        min_value = np.percentile(averaged_data, 10)
        max_value = np.percentile(averaged_data, 90)

        if min_value == max_value or data_length == 0:
            print(f"{Fore.RED}Insufficient or uniform data")
            return

        scale_factor_y = plot_height / (max_value - min_value)

        # Initialize empty plot
        plot = [[" " for _ in range(plot_width)] for _ in range(plot_height)]

        for k, avg in enumerate(averaged_data):
            y = int(round((max_value - avg) * scale_factor_y))
            if y >= plot_height:  # Ensure y is within the bounds of the plot
                y = plot_height - 1
            elif y < 0:  # Ensure very small values are plotted at the bottom
                y = 0

            color = Fore.RED if avg < 0 else Fore.BLUE
            plot[y][k] = f"{color}*{Fore.RESET}"

        # Add Y-axis markers
        y_labels = [(max_value - y / scale_factor_y) for y in range(0, plot_height, plot_height // 5)]
        max_label_length = max([len(f"{label:.2f}") for label in y_labels])

        stats_dict = self.calculate_statistics()
        print(f"{Fore.GREEN}Time-series for {name} (N: {data_length:,}, window size: {window_size:,})")
        # Print the statistics
        for key, value in stats_dict.items():
            print(f"{Fore.GREEN}{key}: {value:,.2f}")
        # Print the plot
        for y, row in enumerate(plot):
            if y // 5 < len(y_labels):
                label = y_labels[y // 5] if y % (plot_height // 5) == 0 else ""
            else:
                label = ""

            if isinstance(label, (int, float)):
                label_str = f"{label:,.2f}".rjust(max_label_length)
            else:
                label_str = label.rjust(max_label_length)
            print(f"{Fore.CYAN}{label_str}{Fore.RESET} | " + "".join(row))
        # Print the x-axis resolution below the x-axis
        print(
            f"{Fore.GREEN}{' ' * (max_label_length + 2)}Each x-axis step represents an average over {window_size:,} data points.{Fore.RESET}\n"
        )

    @cython.locals(
        avg=cython.double,
        mean_val=cython.double,
        median_val=cython.double,
    )
    def calculate_statistics(self) -> dict:
        if len(self.data) == 0:
            print(f"{Fore.RED}No data to compute statistics for.{Fore.RESET}")
            return {}
        mean_val = mean(self.data)  # Calculate mean using statistics.mean
        median_val = median(self.data)  # Calculate median using statistics.median
        variance_val = variance(self.data)  # Calculate variance using statistics.variance
        mode_val = mode(self.data)  # Calculate mode using statistics.mode
        return {"mode": mode_val, "variance": variance_val, "mean": mean_val, "median": median_val}
