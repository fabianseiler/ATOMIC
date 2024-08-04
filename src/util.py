"""
Created by Fabian Seiler at 23.07.24

Consists of useful utility functions
"""

import csv
import re
import os
import shutil


def open_csv(string: str, printS: bool = False) -> str:
    with open(string, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)

        # Initialize an empty string to store the contents
        content_string = ""

        # Read each row in the CSV file
        for row in csv_reader:
            # Convert the row to a string and concatenate it to the content_string
            content_string += ' '.join(row) + ' '

        content_string = "PWL(" + content_string + ")"
    if printS:
        print(content_string)
    return content_string


def extract_energy_from_log(file_path: str) -> float | None:
    # Match any line starting with "sum:" and extract the number after "=" to extract the energy consumption
    pattern = re.compile(r'^sum:.*?=(-?\d+\.?\d*[eE]?[-+]?\d*)')

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                # Extract the number after "="
                number = match.group(1)
                return number

    return None


def simplify_number_string(s: str) -> str:
    """
    Simplify a string with a number to match SPICE
    :param s: string
    :return: Cleaned string
    """
    if '.' in s and s[-2] == '0' and s[-3] == '.':
        return s[:-3] + s[-1]
    return s


def r_on(d, pos=1, R_H=10) -> str:
    """
    Automatic Calculation of Ron with deviation
    :param d: given deviation
    :param pos: if dev is added or subtracted
    :param R_H: Reference resistance
    :return: Cleaned up String with new resistance value
    """
    r = R_H*(1+d/100) if pos == 1 else R_H*(1-d/100)
    return simplify_number_string(f"{r}k")


def r_off(d, pos=1, R_L=1000) -> str:
    """
    Automatic Calculation of Roff with deviation
    :param d: given deviation
    :param pos: if dev is added or subtracted
    :param R_L: Reference resistance
    :return: Cleaned up String with new resistance value
    """
    r = R_L*(1+d/100) if pos == 1 else R_L*(1-d/100)
    return simplify_number_string(f"{r}k")


def resistance_comb9(dev: int, R_H=10, R_L=1000) -> [[str], [str]]:
    """
    Returns two lists with 9 resistance combinations
    :param dev: deviation of the resistance
    :param R_H: High State Resistance
    :param R_L: Low State Resistance
    :return: Ron and Roff lists with varying resistance combinations
    """
    R_on_c = [r_on(dev, 0), r_on(dev, 0), r_on(dev, 0),
              f"{R_H}k", f"{R_H}k", f"{R_H}k",
              r_on(dev, 1), r_on(dev, 1), r_on(dev, 1)]
    R_off_c = [r_off(dev, 0), f"{R_L}k", r_off(dev, 1),
               r_off(dev, 0), f"{R_L}k", r_off(dev, 1),
               r_off(dev, 0), f"{R_L}k", r_off(dev, 1)]
    return R_on_c, R_off_c


def copy_pwm_files(config: dict) -> None:
    """
    Overwrite the PWM files for the given topology
    :param config: config file
    """
    # Iterate over files
    for file in os.listdir(f"./Structures/{config["topology"]}"):
        # If file is a CSV and was used in the current algorithm => overwrite current files in folder
        if file.endswith(".csv"):
            if os.path.exists(f"outputs/PWM_output/{file}"):
                os.remove(f"./Structures/{config["topology"]}/{file}")
                shutil.copy(f"outputs/PWM_output/{file}", f"./Structures/{config["topology"]}/{file}")
            else:
                # If files are unused rewrite the content to not use them
                with open(f"./Structures/{config["topology"]}/{file}", "w") as f:
                    if 'sw' in f.name.split('/')[-1]:
                        f.write(f"0,-100\n {config["cycle_time"] * (config["steps"] + 1)},-100")
                    else:
                        f.write(f"0,0\n {config["cycle_time"] * (config["steps"] + 1)},0")

