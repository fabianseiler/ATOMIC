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


class Plotter:

    def __init__(self, config):
        self.Simulator = Simulator(config)
        self.plots_per_subfigure = 3
        self.colors = ['red', 'blue', 'lime', 'black', 'magenta', 'orange']
        self.linestyles = ['-', '-.', '--', ':']
        self.xtick_style = ["all", "minmax", "3rd"][0]

        # Dict with memristor name and expected logical outcome
        self.expected_logic = {}
        states = [value for value in config["output_states"].values()]

        for i, out in enumerate(config["outputs"]):
            self.expected_logic[out] = states[i]

        self.energy = self.Simulator.calculate_energy()

    def plot_waveforms_with_deviation(self, comb: str, dev: int, recompute: bool = False) -> None:
        """
        Plots specific waveforms with chosen deviation
        :param comb: input combination
        :param dev: chosen deviation
        :param recompute: If the deviation experiments need to be recomputed
        """
        if recompute:
            if os.path.exists("./outputs/Waveforms/"):
                shutil.rmtree("./outputs/Waveforms/")
            if os.path.exists("./outputs/deviation_results/"):
                shutil.rmtree("./outputs/deviation_results/")
            self.Simulator.evaluate_deviation(dev, save=True)

        waveforms = [[] for _ in range(len(self.Simulator.memristors)+1)]

        # Load all the waveforms of the combination
        for k, file in enumerate(os.listdir(f"./outputs/Waveforms/{comb}/{dev}")):
            data = np.genfromtxt(f"./outputs/Waveforms/{comb}/{dev}/" + file, delimiter=' ', names=True)

            # Save the exact waveform for later
            if file == "10k_1000k.txt":
                exact_idx = k

            waveforms[0].append(data["time"])
            for i, mem in enumerate(self.Simulator.memristors):
                waveforms[i+1].append(data[f"{mem}"])

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

        # Plot the figure according to the settings in the init
        total_wv = len(wv_interpolated_array)
        num_subplots = math.ceil(total_wv/self.plots_per_subfigure)

        fig = plt.figure(figsize=(16, 3*num_subplots))
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
        plt.savefig(f"./outputs/Images/Comb_{comb}_{dev}.pdf", bbox_inches='tight', pad_inches=0.01, format='pdf')
        plt.show()

    def plot_deviation_scatter(self, max_dev: int = 50, recompute: bool = False) -> None:
        """
        For every input combination plot the output states over increasing deviations
        :param max_dev: maximum deviation
        :param recompute: If the deviation experiments need to be recomputed
        """
        fig, ax = plt.subplots(1, len(self.expected_logic), figsize=(max(3*max_dev/10, 12), 4))

        # If the results are recomputed
        if recompute:
            for d, dev in enumerate(self.get_dev_list(max_dev)):
                self.Simulator.evaluate_deviation(dev, True)

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
        plt.savefig(f"./outputs/Images/OutputDeviation_Scatter.pdf", bbox_inches='tight', pad_inches=0.01, format='pdf')
        plt.show()

    def plot_deviation_range(self, max_dev: int = 50, recompute: bool = False) -> None:
        """
        Plot the range of output states for every output state
        :param max_dev: maximum deviation
        :param recompute: If the deviation experiments need to be recomputed
        """
        fig = plt.figure(figsize=(max(max_dev/10, 5), 5))

        # If the results are recomputed
        if recompute:
            for d, dev in enumerate(self.get_dev_list(max_dev)):
                self.Simulator.evaluate_deviation(dev, True)

        range_logic1 = [[], []]
        range_logic0 = [[], []]

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

            #l1_array, l0_array = np.array(logic1_states), np.array(logic0_states)
            #l1_max, l1_min = np.max(l1_array, axis=1)[:, 1], np.min(l1_array, axis=1)[:, 0]
            #l0_max, l0_min = np.max(l0_array, axis=1)[:, 1], np.min(l0_array, axis=1)[:, 0]
            #range_logic1[0].append(l1_min), range_logic1[1].append(l1_max)
            #range_logic0[0].append(l0_min), range_logic0[1].append(l0_max)

            l1 = [np.array(output) for output in logic1_states]
            l0 = [np.array(output) for output in logic0_states]
            l1_max = [np.max(e, axis=0)[1] for e in l1]
            l1_min = [np.min(e, axis=0)[0] for e in l1]
            l0_max = [np.max(e, axis=0)[1] for e in l0]
            l0_min = [np.min(e, axis=0)[0] for e in l0]
            range_logic1[0].append(l1_min), range_logic1[1].append(l1_max)
            range_logic0[0].append(l0_min), range_logic0[1].append(l0_max)

        r0, r1 = np.array(range_logic0), np.array(range_logic1)

        for outs in range(len(self.expected_logic)):
            label = [key for key in self.expected_logic][outs]

            plt.fill_between(np.linspace(0, len(self.get_dev_list(max_dev))+1, len(self.get_dev_list(max_dev))),
                             r0[0, :, outs], r0[1, :, outs], interpolate=True, color=self.colors[outs], alpha=0.3, label=label)
            plt.fill_between(np.linspace(0, len(self.get_dev_list(max_dev))+1, len(self.get_dev_list(max_dev))),
                             r1[0, :, outs], r1[1, :, outs], interpolate=True, color=self.colors[outs], alpha=0.3)
        plt.plot([-1, 20], [0.33, 0.33], linestyle='--', color="orange")
        plt.plot([-1, 20], [0.66, 0.66], linestyle='--', color="orange")
        plt.ylim([-0.01, 1.01])
        plt.xlim([-0.25, len(self.get_dev_list(max_dev)) + 1.25])
        plt.xticks(np.linspace(0, len(self.get_dev_list(max_dev))+1, len(self.get_dev_list(max_dev))),
                   labels=self.get_dev_list(max_dev), fontsize=14)
        plt.yticks([0, 1], labels=["HRS", "LRS"], rotation=45, fontsize=14)
        plt.xlabel("$R_{on}$ & $R_{off}$ Deviation in %", fontsize=20)
        plt.grid(False)
        plt.ylabel(f"Resulting State Deviation", fontsize=20)
        plt.legend(loc='center left', fontsize=20)

        plt.tight_layout()
        plt.savefig('./outputs/Images/StateDeviations.pdf', bbox_inches='tight', pad_inches=0.1, format='pdf')
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
        if os.path.exists(f"./OUTPUT_FILES/{name}"):
            shutil.rmtree(f"./OUTPUT_FILES/{name}")
            os.mkdir(f"./OUTPUT_FILES/{name}")
        shutil.copytree("./outputs/deviation_results", f"./OUTPUT_FILES/{name}/deviation_results")
        shutil.copytree("./outputs/Images", f"./OUTPUT_FILES/{name}/Images")

        with open(f"./OUTPUT_FILES/{name}/energy_consumption.txt", "w") as f:
            if not self.energy:
                f.write(f"Energy consumption calculation failed for {name}")
            else:
                f.write(f"Average Energy consumption: {self.energy[1]}\n "
                        f"Energy over Combination: {self.energy[0]}")

        print(f"Files for {name} were saved successfully!")

