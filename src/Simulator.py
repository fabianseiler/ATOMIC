"""
created by Fabian Seiler at 12.06.2024
"""
import json
import pickle
import os
import shutil
import numpy as np
from PyLTSpice import SpiceEditor, RawRead, LTspice, SimCommander
from src.util import open_csv, extract_energy_from_log, resistance_comb9, Logger
from tqdm import tqdm


class Simulator:
    """
    This class is responsible for Netlist writing, SPICE Simulations, and energy calculations
    given PWM files, configurable parameters, and resistive deviations. The results of these experiments are stored.
    """

    def __init__(self, config):

        self.logger = Logger()
        self.logger.L.info(f'Initializing {self.__class__.__name__}')

        self.topology = config["topology"]
        self.memristors = config["memristors"]
        self.switches = config["switches"]

        with open(f"./Structures/{self.topology}.json", "r") as f:
            self.topology_data = json.load(f)
        indices = [self.topology_data["memristors"].index(item) for item in self.memristors]
        self.voltages_mem = [self.topology_data["voltages_mem"][index] for index in indices]

        indices = [self.topology_data["switches"].index(item) for item in self.switches]
        self.voltages_sw = [self.topology_data["voltages_sw"][index] for index in indices]

        # Open SPICE editor
        try:
            self.netlist = SpiceEditor(f"./Structures/{self.topology}/1bit_adder_cin.net")
        except Exception as e:
            self.logger.L.error(f'Could not create SpiceEditor due to: {e}')

        self.netlist_path = None

        # Set number of cycles for algorithm and the cycle time
        with open("./Structures/IMPLY_parameters.json", "r") as f:
            self.parameters = json.load(f)
        self.steps = config["steps"]
        self.cycle_time = self.parameters["t_pulse"]
        self.sim_time = self.steps * self.cycle_time

        with open("./Structures/VTEAM_parameters.json", "r") as g:
            self.vteam_parameters = json.load(g)

        if os.path.exists("./temp/"):
            shutil.rmtree("./temp/")
        os.mkdir("./temp")

    def set_parameters(self, param_values: list) -> None:
        """
        This functions rewrites the netlist of the initialized SpiceEditor, given the parameter values.
        :param param_values: list of parameter values [memristor values, R_on, R_off]
        """

        # Set stop time for transient analysis and R_G for IMPLY logic
        self.netlist.set_parameter("tstop", f"{self.sim_time}u")
        self.netlist.set_parameter("R_g", self.parameters["R_G"])

        for i, mem in enumerate(self.topology_data["memristors"]):
            self.netlist.set_component_value(device=f'{self.topology_data["voltages_mem"][i]}',
                                             value=open_csv(f"./Structures/{self.topology}/{mem}.csv"))

        for j, switch in enumerate(self.topology_data["switches"]):
            self.netlist.set_component_value(device=f'{self.topology_data["voltages_sw"][j]}',
                                             value=open_csv(f"./Structures/{self.topology}/{switch}.csv"))

        # Set the parameters
        for k, mem in enumerate(self.memristors):
            self.netlist.set_parameter(f"w_{mem}", param_values[k])
        self.netlist.set_parameters(R_on=param_values[-2], R_off=param_values[-1])

        # Save the updated netlist
        netlist_path = f"./temp/netlist.net"
        self.netlist.write_netlist(netlist_path)
        self.netlist_path = netlist_path

    def run_simulation(self, param_values: list, energy_sim: bool = False) -> None:
        """
        This function runs the simulations of the SpiceEditor with given parameter values.
        :param param_values: List of parameter values [memristor values, R_on, R_off]
        :param energy_sim: Should be set True to remove components that might interfere with the energy calculation
        """

        # Set the parameter values and check if netlist have been created
        self.set_parameters(param_values)
        if self.netlist_path is None:
            self.logger.L.error('No netlist file has been created')
            raise Exception("Error: No netlist has been created!")

        if energy_sim:
            self.prepare_energy_calculation()

        # Run the Simulation
        try:
            runner = SimCommander(netlist_file=self.netlist_path, simulator=LTspice)
            runner.run()
            runner.wait_completion()
        except Exception as e:
            self.logger.L.error(f'Simulation failed with parameters={param_values} due to: {e}')

    def read_raw(self) -> ([str], [np.ndarray]):
        """
        Reads the data from the raw file resulting from the simulation and stores it.
        :return: Header and output array with simulated data.
        """

        try:
            LTR = RawRead(f"./temp/netlist_1.raw")
            steps = LTR.get_steps()

            outputs = [(LTR.get_trace('time')).get_wave(step)*1e6 for step in steps]
            for mem in self.memristors:
                outputs.append([LTR.get_trace(f'V({mem})').get_wave(step) for step in steps][0])
            outputs_array = np.array(outputs)
            outputs_array[:, 0] = np.zeros(shape=(len(outputs),))     # To fix bug of LTSPICE
            header = ['time'] + [f'{mem}' for mem in self.memristors]

        except Exception as e:
            self.logger.L.error(f'Raw files could not be read due to: {e}')
            raise Exception(f'Raw files could not be read due to: {e}')

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

    def read_energy(self):
        """
        Extracts the energy consumption from the .log file.
        :return: Energy consumption of the current simulation.
        """
        energy = extract_energy_from_log("./temp/netlist_1.log")
        if energy is None:
            self.logger.L.warning("No energy consumption could be extracted from the .log file.")
            raise Exception("Error: Energy could not be calculated!")
        return float(energy)

    def prepare_energy_calculation(self) -> None:
        """
        Prepares the energy calculation of the current simulation by removing unused components that can lead to
        interference problems.
        """
        for component in self.netlist.get_components():
            if component[:2] == 'XX' and (component[2:] not in self.memristors):
                self.netlist.remove_component(component)

    def calculate_energy(self) -> ([float], float):
        """
        Calculates the average energy consumption of the current simulation.
        :return: List of energy consumption per combination, and average energy consumption.
        """
        # Calculate the energy consumption
        print("Calculating energy consumption:")
        self.logger.L.info('Started calculating energy consumption')
        energy = []
        R_on, R_off = self.vteam_parameters["R_on"], self.vteam_parameters["R_off"]
        for inputs in tqdm(range(8)):
            name = bin(inputs)[2:].zfill(3)

            param_values = ([f"{int(name[0]) * 3}n", f"{int(name[1]) * 3}n", f"{int(name[2]) * 3}n"]
                            + [f'0n' for _ in self.memristors[2:]] + [R_on, R_off])
            self.run_simulation(param_values, energy_sim=True)
            energy.append(self.read_energy() * self.sim_time * 1e-6)

        print(f"Average Energy consumption: {sum(energy)/len(energy)}")
        print(f"Energy over Combination: {energy}")
        self.logger.L.info(f'Average Energy consumption: {sum(energy)/len(energy)}')
        self.logger.L.info(f'Energy over Combinations: {energy}')
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
        self.logger.L.info(f'Started calculating deviation: {dev}')

        # Iterate over the input combinations
        for inputs in tqdm(range(8)):

            name = bin(inputs)[2:].zfill(3)
            r_on_c, r_off_c = resistance_comb9(dev, self.vteam_parameters["R_off"], self.vteam_parameters["R_on"])

            comb = []

            # Iterate over the different deviation combinations
            if dev > 0:
                for R_on, R_off in zip(r_on_c, r_off_c):

                    param_values = ([f"{int(name[0])*3}n", f"{int(name[1])*3}n", f"{int(name[2])*3}n"]
                                    + [f'0n' for _ in self.memristors[2:]] + [R_on, R_off])
                    self.run_simulation(param_values)

                    # Save waveforms
                    if save:
                        os.makedirs(f"./outputs/Waveforms/{name}/{dev}", exist_ok=True)
                        comb.append(self.save_raw(f"./outputs/Waveforms/{name}/{dev}/{R_on}_{R_off}.txt"))

            # If the simulation is done without any deviation
            elif dev == 0:
                R_on, R_off = self.vteam_parameters["R_on"], self.vteam_parameters["R_off"]
                param_values = ([f"{int(name[0]) * 3}n", f"{int(name[1]) * 3}n", f"{int(name[2]) * 3}n"]
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
            self.logger.L.info(f'Results of experiments with deviation: {dev} saved in '
                               f'\"outputs/deviation_results/dev_{dev}\"')


