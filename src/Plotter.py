"""
created by Fabian Seiler at 23.07.24
"""
import math
import pickle
import numpy as np
from matplotlib import gridspec
from src.Simulator import Simulator
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import shutil
from src.util import Logger


class Plotter:
    """
    This class builds upon the Simulator class and is responsible for the extraction and visualization of the data.
    """

    def __init__(self, config):

        self.logger = Logger()
        self.logger.L.info(f"Initializing {self.__class__.__name__}")
        self.Simulator = Simulator(config)

        # Plotting Settings
        self.plots_per_subfigure = 3
        self.colors = ['red', 'blue', 'lime', 'black', 'magenta', 'orange']
        self.linestyles = ['-', '-.', '--', ':']
        self.xtick_style = ["all", "minmax", "3rd"][0]

        # Dict with memristor name and expected logical outcome
        try:
            self.expected_logic = {}
            states = [value for value in config["output_states"].values()]

            for i, out in enumerate(config["outputs"]):
                self.expected_logic[out] = states[i]
        except Exception as e:
            self.logger.L.error(f"Loading of expected logic values failed at {self.__class__.__name__}: {e}")

        # Calculate the energy
        self.energy = self.Simulator.calculate_energy()

    def plot_waveforms_with_deviation(self, comb: str, dev: int, recompute: bool = False,
                                      show: bool = False, fig_type: str = 'pdf', plots_per_subfigure: int = 3) -> None:
        """
        Plots specific waveforms with chosen deviation
        :param comb: input combination
        :param dev: chosen deviation
        :param recompute: If the deviation experiments need to be recomputed
        :param show: If the deviation experiments should be shown
        :param fig_type: Type of plot
        :param plots_per_subfigure: Number of plots per subfigure
        """

        self.plots_per_subfigure = plots_per_subfigure

        if recompute:
            self.logger.L.info(f"Started recomputing deviation {dev}")

            if os.path.exists("./outputs/Waveforms/"):
                shutil.rmtree("./outputs/Waveforms/")
            if os.path.exists("./outputs/deviation_results/"):
                shutil.rmtree("./outputs/deviation_results/")
            self.Simulator.evaluate_deviation(dev, save=True)

        waveforms = [[] for _ in range(len(self.Simulator.memristors)+1)]

        try:
            # Load all the waveforms of the combination
            for k, file in enumerate(os.listdir(f"./outputs/Waveforms/{comb}/{dev}")):
                data = np.genfromtxt(f"./outputs/Waveforms/{comb}/{dev}/" + file, delimiter=' ', names=True)

                # Save the exact waveform for later
                if file == f"""{self.Simulator.vteam_parameters["R_on"]}_{self.Simulator.vteam_parameters["R_off"]}.txt""":
                    exact_idx = k

                waveforms[0].append(data["time"])
                for i, mem in enumerate(self.Simulator.memristors):
                    waveforms[i+1].append(data[f"{mem}"])
        except Exception as e:
            self.logger.L.error(f"Loading of waveforms failed at {self.__class__.__name__}: {e}")
            raise Exception(f"Loading of waveforms failed at {self.__class__.__name__}: {e}")

        try:
            # Interpolate the waveforms to have a common timescale
            time_common = np.linspace(0, self.Simulator.sim_time, 1000)
            wv_interpolated = [[] for _ in self.Simulator.memristors]

            for c in range(0, 9):
                for j in range(1, len(waveforms)):
                    temp = interp1d(waveforms[0][c], waveforms[j][c], bounds_error=False, fill_value="extrapolate")
                    wv_interpolated[j-1].append(temp(time_common))

            # Create a numpy array and find the min and max values for each memristor waveform
            wv_interpolated_array = [np.array(entry) for entry in wv_interpolated]

            max_points = [np.max(item, axis=0)[1:] for item in wv_interpolated_array]
            min_points = [np.min(item, axis=0)[1:] for item in wv_interpolated_array]
            time_common = time_common[1:]
        except Exception as e:
            self.logger.L.error(f"Interpolation of waveforms to common timescale failed: {e}")
            raise Exception(f"Interpolation of waveforms to common timescale failed at dev={dev}, comb={comb} due to: {e}")

        try:
            # Plot the figure according to the settings in the init
            total_wv = len(wv_interpolated_array)
            num_subplots = math.ceil(total_wv/self.plots_per_subfigure)

            fig = plt.figure(figsize=(16, max(2*num_subplots, 3)))
            if num_subplots == 1:
                gs = gridspec.GridSpec(1, 1)
            else:
                gs = gridspec.GridSpec(num_subplots, 1, height_ratios=[1, 1])
            axs = []

            for idx in range(num_subplots):
                axs.append(plt.subplot(gs[idx]))

            for idx in range(num_subplots):
                # Plot outlines of the deviation
                for i in range(self.plots_per_subfigure*idx, self.plots_per_subfigure*(idx+1)):
                    if total_wv <= i:
                        continue
                    axs[idx].plot(time_common, max_points[i],
                                  color=self.colors[i % len(self.colors)],
                                  linewidth=0.2, alpha=0.5)
                    axs[idx].plot(time_common, min_points[i],
                                  color=self.colors[i % len(self.colors)],
                                  linewidth=0.2, alpha=0.5)
                # Plot the area in between
                for i in range(self.plots_per_subfigure*idx, self.plots_per_subfigure*(idx+1)):
                    if total_wv <= i:
                        continue
                    axs[idx].fill_between(time_common, max_points[i], min_points[i],
                                          color=self.colors[i % len(self.colors)], alpha=0.3)

                # Plot the waveforms with exact R_on and R_off
                exact_wv = [wv_interpolated_array[i][exact_idx] for i in range(len(wv_interpolated_array))]

                for i in range(self.plots_per_subfigure*idx, self.plots_per_subfigure*(idx+1)):
                    if total_wv <= i:
                        continue
                    axs[idx].plot(time_common, exact_wv[i][1:], color=self.colors[i % len(self.colors)],
                                  linewidth=6, linestyle=self.linestyles[i % self.plots_per_subfigure],
                                  label=f"{self.Simulator.memristors[i]}")

                # General plot settings
                axs[idx].set_ylim([-0.02, 1.01])
                axs[idx].set_xlim([0, self.Simulator.sim_time])

                if self.xtick_style == "all":
                    ticks = [self.Simulator.cycle_time * x for x in range(self.Simulator.steps+1)]
                elif self.xtick_style == "minmax":
                    ticks = [0, self.Simulator.sim_time]
                elif self.xtick_style == "3rd":
                    ticks = [self.Simulator.cycle_time * x for x in range(self.Simulator.steps+1)][0::3]
                else:
                    ticks = []

                axs[idx].set(xticks=ticks, yticks=[0, 0.5, 1])
                axs[idx].tick_params(axis='both', which='major', labelsize=20)
                axs[idx].legend(loc='center left', fontsize=20)
                axs[idx].grid(True)

                if idx == num_subplots - 1:
                    axs[idx].set_xlabel("Time in $\mu$s", fontsize=24, labelpad=0)

            plt.tight_layout()
            fig_type = str(fig_type)
            plt.savefig(f"./outputs/Images/Comb_{comb}_{dev}.{fig_type}", bbox_inches='tight',
                        pad_inches=0.01, format=fig_type)
            if show:
                plt.show()

        except Exception as e:
            self.logger.L.error(f"Plotting waveforms at deviation={dev}, combination={comb} failed due to: {e}")

    def plot_deviation_scatter(self, max_dev: int = 50, recompute: bool = False,
                               show: bool = False, fig_type: str = 'pdf') -> None:
        """
        For every input combination plot the output states over increasing deviations
        :param max_dev: maximum deviation
        :param recompute: If the deviation experiments need to be recomputed
        :param show: If the deviation experiments should be shown
        :param fig_type: Type of plot
        """
        fig, ax = plt.subplots(1, len(self.expected_logic), figsize=(max(3*max_dev/10, 12), 4))

        # If the results are recomputed
        if recompute:
            self.logger.L.info(f"Started recomputing deviation experiments up to a deviation of {max_dev}")
            for d, dev in enumerate(self.get_dev_list(max_dev)):
                self.Simulator.evaluate_deviation(dev, True)

        try:
            # Iterate over the deviations configure above
            for d, dev in enumerate(self.get_dev_list(max_dev)):
                with open(f"./outputs/deviation_results/dev_{dev}", 'rb') as fp:
                    values = pickle.load(fp)

                for i, comb in enumerate(values):
                    expected_res = self.get_expected_logic(i)
                    output_mem = [res[0] for res in expected_res]
                    output_values = [res[1] for res in expected_res]
                    dev_values = [[comb[c][i] for c in range(len(comb))] for i in self.find_indices(output_mem)]

                    for outs in range(len(dev_values)):
                        for dev_combs in range(len(dev_values[0])):
                            color = "b" if abs(dev_values[outs][dev_combs] - output_values[outs]) <= 0.33 else "r"
                            marker = "o" if output_values[outs] == 0 else "^"
                            ax[outs].scatter(d, dev_values[outs][dev_combs], color=color, marker=marker,
                                             linewidths=1, alpha=0.5)
        except Exception as e:
            self.logger.L.error(f"Deviation scatter plot failed due to: {e}")
            raise Exception(f"Deviation scatter plot failed due to: {e}")

        # General Plot settings for the subplot
        for idx in range(len(self.expected_logic)):
            ax[idx].plot([-1, 20], [0.33, 0.33], linestyle='--', color="orange", label="Thresh LOW", alpha=0.5)
            ax[idx].plot([-1, 20], [0.66, 0.66], linestyle='--', color="orange", label="Thresh HIGH", alpha=0.5)
            ax[idx].set_ylim([0, 1])
            ax[idx].set_xlim([-0.25, len(self.get_dev_list(max_dev))-0.75])
            ax[idx].set_xticks(range(len(self.get_dev_list(max_dev))), labels=self.get_dev_list(max_dev))
            ax[idx].set_yticks([0, 1], labels=["HRS", "LRS"], rotation=45)
            ax[idx].set_xlabel("$R_{on}$ & $R_{off}$ Deviation in %", fontsize=20)
            ax[idx].grid(axis='y')
            ax[idx].set_ylabel(f"State Deviation of {output_mem[idx]}", fontsize=20)

        plt.tight_layout()
        fig_type = str(fig_type)
        plt.savefig(f"./outputs/Images/OutputDeviation_Scatter.{fig_type}", bbox_inches='tight',
                    pad_inches=0.01, format=fig_type)
        if show:
            plt.show()

    def plot_deviation_range(self, max_dev: int = 50, recompute: bool = False,
                             show: bool = False, fig_type: str = 'pdf', save_dev_range: bool = False) -> None:
        """
        Plot the range of output states for every output state
        :param max_dev: maximum deviation
        :param recompute: If the deviation experiments need to be recomputed
        :param show: If the deviation experiments should be shown
        :param fig_type: Type of plot
        :param save_dev_range: If the range of the deviation experiments should be stored
        """
        fig = plt.figure(figsize=(max(2*max_dev/10, 5), 5))

        # If the results are recomputed
        if recompute:
            self.logger.L.info(f"Started recomputing deviation experiments up to a deviation range of {max_dev}")
            for d, dev in enumerate(self.get_dev_list(max_dev)):
                self.Simulator.evaluate_deviation(dev, True)

        range_logic1 = [[], []]
        range_logic0 = [[], []]

        try:
            # Iterate over the deviations configure above
            for d, dev in enumerate(self.get_dev_list(max_dev)):

                logic1_states = [[] for _ in range(len(self.expected_logic))]
                logic0_states = [[] for _ in range(len(self.expected_logic))]

                try:
                    with open(f"./outputs/deviation_results/dev_{dev}", 'rb') as fp:
                        values = pickle.load(fp)
                except FileNotFoundError:
                    raise FileNotFoundError(f"File \"dev_{dev}\" does not exist")

                for i, comb in enumerate(values):
                    expected_res = self.get_expected_logic(i)
                    output_mem = [res[0] for res in expected_res]
                    output_values = [res[1] for res in expected_res]
                    dev_values = [[comb[c][i] for c in range(len(comb))] for i in self.find_indices(output_mem)]

                    # Get the minimum and maximum values for each output memristor
                    for j, outs in enumerate(output_values):
                        if outs == 1:
                            logic1_states[j].append([min(dev_values[j]), max(dev_values[j])])
                        else:
                            logic0_states[j].append([min(dev_values[j]), max(dev_values[j])])

                l1 = [np.array(output) for output in logic1_states]
                l0 = [np.array(output) for output in logic0_states]
                l1_max = [np.max(e, axis=0)[1] for e in l1]
                l1_min = [np.min(e, axis=0)[0] for e in l1]
                l0_max = [np.max(e, axis=0)[1] for e in l0]
                l0_min = [np.min(e, axis=0)[0] for e in l0]
                range_logic1[0].append(l1_min), range_logic1[1].append(l1_max)
                range_logic0[0].append(l0_min), range_logic0[1].append(l0_max)

            r0, r1 = np.array(range_logic0), np.array(range_logic1)
            if save_dev_range:
                text = "Deviation," + "".join([f"{str(d)}," for d in self.get_dev_list(max_dev)]) + "\n"
                for idx, out in enumerate(self.expected_logic):
                    data_str = (f"Output: {out}\n"
                                f"Logic 0 (min), " + ", ".join(map(str, r0[idx, :, 0])) + "\n"
                                f"Logic 0 (max), " + ", ".join(map(str, r0[idx, :, 1])) + "\n"
                                f"Logic 1 (min), " + ", ".join(map(str, r1[idx, :, 0])) + "\n"
                                f"Logic 1 (max), " + ", ".join(map(str, r1[idx, :, 1])) + "\n")
                    text += data_str

                with open(f"./outputs/deviation_results/deviation_range.csv", "w") as fp:
                    fp.write(text[:-1])

        except Exception as e:
            self.logger.L.info(f"Extraction of min and max values for each deviation failed due to: {e}")
            raise Exception(f"Extraction of min and max values for each deviation failed due to: {e}")

        try:
            for outs in range(len(self.expected_logic)):
                label = [key for key in self.expected_logic][outs]

                plt.fill_between(np.linspace(0, len(self.get_dev_list(max_dev))+1, len(self.get_dev_list(max_dev))),
                                 r0[0, :, outs], r0[1, :, outs], interpolate=True, color=self.colors[outs], alpha=0.3, label=label)
                plt.fill_between(np.linspace(0, len(self.get_dev_list(max_dev))+1, len(self.get_dev_list(max_dev))),
                                 r1[0, :, outs], r1[1, :, outs], interpolate=True, color=self.colors[outs], alpha=0.3)
        except Exception as e:
            self.logger.L.error(f"Plotting of deviation range failed due to: {e}")
            raise Exception(f"Plotting of deviation range failed due to: {e}")

        plt.plot([-1, 20], [0.33, 0.33], linestyle='--', color="orange")
        plt.plot([-1, 20], [0.66, 0.66], linestyle='--', color="orange")
        plt.ylim([-0.01, 1.01])
        plt.xlim([-0.25, len(self.get_dev_list(max_dev)) + 1.25])
        plt.xticks(np.linspace(0, len(self.get_dev_list(max_dev))+1, len(self.get_dev_list(max_dev))),
                   labels=self.get_dev_list(max_dev), fontsize=20)
        plt.yticks([0, 1], labels=["HRS", "LRS"], rotation=45, fontsize=20)
        plt.xlabel("$R_{on}$ & $R_{off}$ Deviation in %", fontsize=24)
        plt.grid(False)
        plt.ylabel(f"Resulting State Deviation", fontsize=24)
        plt.legend(loc='center left', fontsize=24)

        plt.tight_layout()
        fig_type = str(fig_type)
        plt.savefig(f'./outputs/Images/StateDeviations.{fig_type}', bbox_inches='tight',
                    pad_inches=0.1, format=fig_type)
        if show:
            plt.show()

    @staticmethod
    def get_dev_list(max_dev: int = 50) -> [int]:
        """
        Customized function that returns a list of specific deviation values
        :param max_dev: maximum deviation
        :return: list of deviation values
        """
        if max_dev < 5:
            return [0, max_dev]
        elif max_dev <= 10:
            return [0, 5, max_dev]
        else:
            dev_list = [0, 5, 10]
            current_value = 20
            while current_value <= max_dev:
                dev_list.append(current_value)
                current_value += 10
            if max_dev not in dev_list:
                dev_list.append(max_dev)
            return dev_list

    def get_expected_logic(self, comb: int) -> [[int, int]]:
        """
        Fetches the expected logic states and returns the expected and simulated logic states
        :param comb: input combination
        :return: list of lists containing the expected values and simulated values
        """
        return [[elem, self.expected_logic[elem][comb]] for elem in self.expected_logic]

    def find_indices(self, mem) -> [int]:
        """
        Finds indices of elements in mem
        :param mem:
        :return: list of indices
        """
        indices = []
        for i in range(len(self.Simulator.memristors)):
            if self.Simulator.memristors[i] in mem:
                indices.append(i)
        return indices

    def save_algorithm_files(self, name: str) -> None:
        """
        Stores the current files and outputs in a new folder
        :param name: name of the folder
        """
        try:
            if os.path.exists(f"./OUTPUT_FILES/{name}"):
                shutil.rmtree(f"./OUTPUT_FILES/{name}")
                os.makedirs(f"./OUTPUT_FILES/{name}", exist_ok=True)
            shutil.copytree("./outputs/deviation_results", f"./OUTPUT_FILES/{name}/deviation_results")
            shutil.copytree("./outputs/Images", f"./OUTPUT_FILES/{name}/Images")

            with open(f"./OUTPUT_FILES/{name}/energy_consumption.txt", "w") as f:
                if not self.energy:
                    f.write(f"Energy consumption calculation failed for {name}")
                else:
                    f.write(f"Average Energy consumption: {self.energy[1]}\n "
                            f"Energy over Combination: {self.energy[0]}")
            if os.path.exists(f"./OUTPUT_FILES/State_History.txt"):
                os.remove(f"./OUTPUT_FILES/State_History.txt")
            shutil.copy("./outputs/State_History.txt", "./OUTPUT_FILES/State_History.txt")
        except Exception as e:
            self.logger.L.error(f"Saving algorithm files failed due to: {e}")

        print(f"Files for {name} were saved successfully!")
        self.logger.L.info(f"Files for {name} were saved successfully!")
