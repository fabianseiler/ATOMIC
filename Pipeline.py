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
    parser.add_argument('--config_file', type=str, default='./configs/SSAx2.json', help='Configuration file')
    parser.add_argument('--max_dev', type=int, default=50, help='Maximum deviation for experiments, recommended: 50')
    parser.add_argument('--dev_wf', type=int, default=20, help='Deviations at which waveforms are plotted')
    parser.add_argument('--fig_type', type=str, default='png',
                        choices=['pdf', 'png', 'svg'], help='Type of the plot that will be stored')

    args = parser.parse_args()
    max_dev = args.max_dev
    dev_wf = int(args.dev_wf)
    fig_type = args.fig_type

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    # Check if the algorithm is valid and the resulting states are correct
    LS = LogicState(config)
    LS.calc_algorithm(plot_tt=True)
    print("\n--------------- Logic States verified! ---------------\n")

    # Automatically create PWM signals and store them in "PWM_output"
    PWM = PWMWriter(config)
    PWM.eval_algo()
    print("\n--------------- PWM Signals created! ---------------\n")

    # Copy the files to the folder of the corresponding topology (This removes the old files !)
    copy_pwm_files(config)
    print(f"\n--------- Files of {config["topology"]} overwritten! --------\n")

    # Plotter
    PLT = Plotter(config)
    PLT.plot_deviation_scatter(max_dev=max_dev, recompute=True, fig_type=fig_type)
    PLT.plot_deviation_range(max_dev=max_dev, recompute=False, fig_type=fig_type)
    print(f"\n--------- Deviation Experiments completed --------\n")

    for comb in range(8):
        comb_str = bin(comb)[2:].zfill(3)
        PLT.plot_waveforms_with_deviation(comb_str, dev=dev_wf, recompute=False, fig_type=fig_type)
    print(f"\n--------- Waveforms with deviation {dev_wf} saved --------\n")

    PLT.save_algorithm_files(f"{config["algorithm"].split(".")[0]}")
