import json
import copy
import os
import sys


def split_list_parameter(experiment):
    """
    Splits the list parameters of an experiment into two halves,
    generating two sets of experiments if applicable.
    """
    for key, value in experiment.items():
        if isinstance(value, dict):  # Dive deeper if the value is a dictionary
            for subkey, subvalue in value.items():
                if isinstance(subvalue, list) and len(subvalue) > 1:  # Check for list parameters
                    midpoint = len(subvalue) // 2
                    # Split the list parameter into two halves
                    first_half = subvalue[:midpoint]
                    second_half = subvalue[midpoint:]

                    # Create two copies of the experiment with each half of the list parameter
                    experiment_first_half = copy.deepcopy(experiment)
                    experiment_second_half = copy.deepcopy(experiment)
                    experiment_first_half[key][subkey] = first_half
                    experiment_second_half[key][subkey] = second_half
                    return [experiment_first_half, experiment_second_half]
    # Return the original experiment in a list if no split is applicable
    return [experiment]


def do_split(input_file):
    # Generate output filenames based on input filename
    base_name = os.path.splitext(input_file)[0]
    output_file_1 = f"{base_name}_1.json"
    output_file_2 = f"{base_name}_2.json"

    # Load the original experiments from file
    with open(input_file, "r") as file:
        experiments = json.load(file)

    # Prepare the lists to store the split experiments
    experiments_first_half = []
    experiments_second_half = []

    # Process each experiment, splitting when list parameters are found
    for experiment in experiments:
        # Handle special cases like {"top_n": ...} by including them in both halves
        if "top_n" in experiment:
            experiments_first_half.append(experiment)
            experiments_second_half.append(experiment)
            continue

        # Split the experiment based on list parameters
        split_experiments = split_list_parameter(experiment)
        experiments_first_half.append(split_experiments[0])
        if len(split_experiments) > 1:  # If the experiment was split, add the second half to the second list
            experiments_second_half.append(split_experiments[1])

    # Write the split experiments to new JSON files
    with open(output_file_1, "w") as file:
        json.dump(experiments_first_half, file, indent=2)

    with open(output_file_2, "w") as file:
        json.dump(experiments_second_half, file, indent=2)

    # Rename the input file to indicate it has been processed
    processed_file_name = base_name + "_split"
    os.rename(input_file, processed_file_name)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    do_split(input_file)
