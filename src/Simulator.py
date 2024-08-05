"""
created by Fabian Seiler at 12.06.2024
"""
import json
import pickle
import os
import shutil
import numpy as np
from PyLTSpice import SpiceEditor, RawRead, LTspice, SimCommander
from src.util import open_csv, extract_energy_from_log, resistance_comb9
from tqdm import tqdm


class Simulator:

    def __init__(self, config):

        self.topology = config["topology"]
        self.memristors = config["memristors"]
        self.switches = config["switches"]
        self.voltages_mem = config["voltages_mem"]
        self.voltages_sw = config["voltages_sw"]

        # Open SPICE editor
        self.netlist = SpiceEditor(f"./Structures/{self.topology}/1bit_adder_cin.net")
        self.netlist_path = None

        # Set number of cycles for algorithm and the cycle time
        self.steps = config["steps"]
        self.cycle_time = config["cycle_time"]
        self.sim_time = self.steps * self.cycle_time

        if os.path.exists("./temp/"):
            shutil.rmtree("./temp/")
        os.mkdir("./temp")

    def set_parameters(self, param_values: list) -> None:
        """
        This functions rewrites the netlist of the initialized SpiceEditor, given the parameter values.
        :param param_values: list of parameter values [memristor values, R_on, R_off]
        """

        # Set stop time for transient analysis
        self.netlist.set_parameter("tstop", f"{self.sim_time}u")

        with open(f"./Structures/{self.topology}.json") as f:
            topology_data = json.load(f)

        for i, mem in enumerate(topology_data["memristors"]):
            self.netlist.set_component_value(device=f'{topology_data["voltages_mem"][i]}',
                                             value=open_csv(f"./Structures/{self.topology}/{mem}.csv"))

        for j, switch in enumerate(topology_data["switches"]):
            self.netlist.set_component_value(device=f'{topology_data["voltages_sw"][j]}',
                                             value=open_csv(f"./Structures/{self.topology}/{switch}.csv"))

        # Set the parameters
        for k, mem in enumerate(self.memristors):
            self.netlist.set_parameter(f"w_{mem}", param_values[k])
        self.netlist.set_parameters(R_on=param_values[-2], R_off=param_values[-1])

        # Save the updated netlist
        netlist_path = f"./temp/netlist.net"
        self.netlist.write_netlist(netlist_path)
        self.netlist_path = netlist_path

    def run_simulation(self, param_values: list) -> None:
        """
        This function runs the simulations of the SpiceEditor with given parameter values.
        :param param_values:
        """

        # Set the parameter values and check if netlist have been created
        self.set_parameters(param_values)
        if self.netlist_path is None:
            raise Exception("Error: No netlist has been created!")

        # Run the Simulation
        runner = SimCommander(netlist_file=self.netlist_path, simulator=LTspice)
        runner.run()
        runner.wait_completion()

    def read_raw(self) -> ([str], [np.ndarray]):
        """
        Reads the data from the raw file resulting from the simulation and stores it.
        :return: Header and output array with simulated data.
        """

        LTR = RawRead(f"./temp/netlist_1.raw")
        steps = LTR.get_steps()

        outputs = [(LTR.get_trace('time')).get_wave(step)*1e6 for step in steps]
        for mem in self.memristors:
            outputs.append([LTR.get_trace(f'V({mem})').get_wave(step) for step in steps][0])

        outputs_array = np.array(outputs)
        outputs_array[:, 0] = np.zeros(shape=(len(outputs),))     # To fix bug of LTSPICE
        header = ['time'] + [f'{mem}' for mem in self.memristors]

        return header, outputs_array

    def save_raw(self, save_path: str) -> np.ndarray:
        """
        Saves the processed data into a file with corresponding header.
        :param save_path: Path to the save file
        :return: Last row of the array
        """
        header, outputs = self.read_raw()
        with open(save_path, 'w') as f:
            f.write(' '.join(header) + '\n')
            for row in outputs.T:
                f.write(' '.join(map(str, row)) + '\n')

        # Return the last row for further evaluation
        return outputs[1:, -1]

    @staticmethod
    def read_energy():
        """
        Extracts the energy consumption from the .log file.
        :return: Energy consumption of the current simulation.
        """
        energy = extract_energy_from_log("./temp/netlist_1.log")
        if energy is None:
            raise Exception("Error: Energy could not be calculated!")
        return float(energy)

    def calculate_energy(self):
        # Calculate the energy consumption
        print("Calculating energy consumption:")
        energy = []
        R_on, R_off = "10k", "1000k"
        for inputs in tqdm(range(8)):
            name = bin(inputs)[2:].zfill(3)

            param_values = ([f"{int(name[0]) * 3}n", f"{int(name[1]) * 3}n", f"{int(name[2]) * 3}n"]
                            + [f'0n' for _ in self.memristors[2:]] + [R_on, R_off])
            self.run_simulation(param_values)
            energy.append(self.read_energy() * self.sim_time * 1e-6)
        return energy, sum(energy)/len(energy)

    def evaluate_deviation(self, dev: int = 20, save: bool = True) -> None:
        """
        Evaluates the deviation of the current simulation.
        :param dev: deviation to be evaluated
        :param save: If the results should be saved
        :return: Average energy consumption of the current algorithm
        """
        valid_res = []
        print(f"Calculating deviation {dev}:")

        # Iterate over the input combinations
        for inputs in tqdm(range(8)):

            name = bin(inputs)[2:].zfill(3)
            r_on_c, r_off_c = resistance_comb9(dev)

            comb = []

            # Iterate over the different deviation combinations
            if dev > 0:
                for R_on, R_off in zip(r_on_c, r_off_c):

                    param_values = ([f"{name[0]*3}n", f"{name[1]*3}n", f"{name[2]*3}n"]
                                    + [f'0n' for _ in self.memristors[2:]] + [R_on, R_off])
                    self.run_simulation(param_values)

                    # Save waveforms
                    if save:
                        os.makedirs(f"./outputs/Waveforms/{name}/{dev}", exist_ok=True)
                        comb.append(self.save_raw(f"./outputs/Waveforms/{name}/{dev}/{R_on}_{R_off}.txt"))

            # If the simulation is done without any deviation
            elif dev == 0:
                R_on, R_off = "10k", "1000k"
                param_values = ([f"{name[0] * 3}n", f"{name[1] * 3}n", f"{name[2] * 3}n"]
                                + [f'0n' for _ in self.memristors[2:]] + [R_on, R_off])
                self.run_simulation(param_values)

                # Save waveforms
                if save:
                    os.makedirs(f"./outputs/Waveforms/{name}/{dev}", exist_ok=True)
                    comb.append(self.save_raw(f"./outputs/Waveforms/{name}/{dev}/{R_on}_{R_off}.txt"))

            valid_res.append(comb)

        if save:
            os.makedirs(f"./outputs/deviation_results/", exist_ok=True)
            with open(f"./outputs/deviation_results/dev_{dev}", 'wb') as fp:
                pickle.dump(valid_res, fp)


