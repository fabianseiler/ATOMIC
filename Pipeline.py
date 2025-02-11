"""
Created by Fabian Seiler @ 26.07.24
Modified by Moritz Hinkel @ 11.02.2025
"""
import json
import argparse
import pathlib

from src.FunctionalValidation import FunctionalValidation
from src.ControlLogicGenerator import ControlLogicGenerator
from src.Plotter import Plotter
from src.util import copy_pwm_files

def run_pipeline(config_file: str, max_dev: int = 20, dev_wf: int = 50, fig_type: str = 'pdf') -> None:
    """
    Run the pipeline to evaluate the IMPLY algorithm
    :param config: Path to the configuration file
    :param max_dev: Maximum deviation for range experiments
    :param dev_wf: Deviations at which waveforms are plotted
    :param fig_type: Type of the plot that will be stored
    :return: None
    """

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Check if the algorithm is valid and the resulting states are correct
    Verifier = FunctionalValidation(config)
    Verifier.calc_algorithm(plot_tt=True)
    print("\n--------------- Logic States verified! ---------------\n")

    # Automatically create PWM signals and store them in "PWM_output"
    CLG = ControlLogicGenerator(config)
    CLG.eval_algo()
    print("\n--------------- PWM Signals created! ---------------\n")

    # Copy the files to the folder of the corresponding topology (This removes the old files !)
    copy_pwm_files(config, CLG.step_size)
    print(f"""\n--------- Files of {config["topology"]} topology overwritten! --------\n""")

    # Plotter
    PLT = Plotter(config)

    # Create range for resistance deviation experiments
    dev_range = PLT.get_dev_list(max_dev=max_dev)

    PLT.plot_deviation_scatter(dev_range=dev_range, recompute=True, fig_type=fig_type)     # Set Recompute
    PLT.plot_deviation_range(dev_range=dev_range, recompute=False, fig_type=fig_type, save_dev_range=True)
    print(f"""\n--------- Deviation Experiments completed --------\n""")

    for comb in range(2**len(config["inputs"])):
        comb_str = bin(comb)[2:].zfill(len(config["inputs"]))
        PLT.plot_waveforms_with_deviation(comb_str, dev=dev_wf, recompute=False, fig_type=fig_type)
    print(f"""\n--------- Waveforms with deviation {dev_wf} saved --------\n""")

    PLT.save_algorithm_files(f"""{config["algorithm"].split(".")[0]}""")

def run_all(max_dev: int = 20, dev_wf: int = 50, fig_type: str = 'pdf') -> None:
    """
    Run the pipeline for all algorithms
    :param max_dev: Maximum deviation for range experiments
    :param dev_wf: Deviations at which waveforms are plotted
    :param fig_type: Type of the plot that will be stored
    :return: None
    """

    configs = [config_file for config_file in list(pathlib.Path('./configs').glob('*.json'))]

    for config_file in configs:
        run_pipeline(config_file, max_dev, dev_wf, fig_type)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the pipeline and evaluate the IMPLY algorithm')
    parser.add_argument('--config_file', type=str, default='./configs/Serial_exact_rohani.json', help='Configuration file')
    parser.add_argument('--max_dev', type=int, default=40, help='Maximum deviation for experiments, recommended: 40')
    parser.add_argument('--dev_wf', type=int, default=20, help='Deviations at which waveforms are plotted')
    parser.add_argument('--fig_type', type=str, default='pdf',
                        choices=['pdf', 'png', 'svg'], help='Type of the plot that will be stored')

    args = parser.parse_args()
    max_dev = args.max_dev
    dev_wf = int(args.dev_wf)
    fig_type = args.fig_type
    config_file = args.config_file

    run_pipeline(config_file, max_dev, dev_wf, fig_type)
