"""
Created by Moritz Hinkel @ 09.02.2025

Run the deviation experiments mentioned in the SMACD paper for different serial adder algorithms
"""
import json

from src.ControlLogicGenerator import ControlLogicGenerator
from src.Plotter import Plotter
from src.util import copy_pwm_files


if __name__ == '__main__':

    algorithms = ['exact_rohani', 'exact_karimi', 'exact_seiler', 'exact_teimoory']
    config_files = [f'./configs/Serial_{algo}.json' for algo in algorithms]
    dev_r = [0, 10, 20, 30, 40, 50]
    dev_v = [0, 1, 2, 3, 4, 5, 6]



    for algorithm in algorithms:
        
        print(f"\n--------------- Evaluating algorithm {algorithm} ---------------\n")

        with open(config_files[algorithms.index(algorithm)], 'r') as f:
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
        print(f"""\n--------- Deviation Experiments completed --------\n""")

        passmatrix = PLT.get_passmatrices_for_deviation(algorithm)
        
        print(f"""\n--------- Passmatrix for algorithm {algorithm} --------\n""")
        print(passmatrix)

