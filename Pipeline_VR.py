"""
Created by Moritz Hinkel @ 09.02.2025
"""
import json
import argparse

from src.ControlLogicGenerator import ControlLogicGenerator
from src.Plotter import Plotter
from src.util import copy_pwm_files


if __name__ == '__main__':

    config_files = ['./configs/Serial_exact_rohani.json', './configs/Serial_exact_karimi.json', './configs/Serial_exact_seiler.json', './configs/Serial_exact_teimoory.json']
    dev_r = [0, 10, 20, 30, 40, 50]
    dev_v = [0, 1, 2, 3, 4, 5, 6]


    for file in config_files:
        
        print(f"\n--------------- Evaluating algorithm {file} ---------------\n")

        with open(file, 'r') as f:
            config = json.load(f)

        # Automatically create PWM signals and store them in "PWM_output"
        CLG = ControlLogicGenerator(config)
        CLG.eval_algo()
        print("\n--------------- PWM Signals created! ---------------\n")

        # Copy the files to the folder of the corresponding topology (This removes the old files !)
        copy_pwm_files(config, CLG.step_size)
        print(f"""\n--------- Files of {config["topology"]} topology overwritten! --------\n""")

        # Plotter
        PLT = Plotter(config, calculate_energy=False)

        PLT.run_deviation_experiments(dev_r, dev_v)

