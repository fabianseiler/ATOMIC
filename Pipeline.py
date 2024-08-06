"""
Created by Fabian Seiler @ 26.07.24
"""
import json
import argparse

from src.LogicState import LogicState
from src.PWMWriter import PWMWriter
from src.Simulator import Simulator
from src.Plotter import Plotter
from src.util import copy_pwm_files


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the pipeline and evaluate the IMPLY algorithm')
    parser.add_argument('--config_file', type=str, default='./configs/SIAFA1.json', help='Configuration file')

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    # Check if the algorithm is valid and the resulting states are correct
    LS = LogicState(config)
    LS.calc_algorithm(plot_tt=True)
    print("\n\n--------------- Logic States verified! ---------------\n")

    # Automatically create PWM signals and store them in "PWM_output"
    PWM = PWMWriter(config)
    PWM.eval_algo()
    print("\n\n--------------- PWM Signals created! ---------------\n")

    # Copy the files to the folder of the corresponding topology (This removes the old files !)
    copy_pwm_files(config)
    print(f"\n\n--------- Files of {config["topology"]} overwritten! --------\n")

    # Plotter
    PLT = Plotter(config)
    PLT.plot_deviation_scatter(max_dev=50, recompute=True)
    PLT.plot_deviation_range(max_dev=50, recompute=False)
    print(f"\n\n--------- Deviation Experiments completed --------\n")

    dev = 20
    for comb in range(8):
        comb_str = bin(comb)[2:].zfill(3)
        PLT.plot_waveforms_with_deviation(comb_str, dev=dev, recompute=False)
    print(f"\n\n--------- Waveforms with deviation {dev} saved --------\n")

    PLT.save_algorithm_files(f"{config["algorithm"].split(".")[0]}")
