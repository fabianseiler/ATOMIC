"""
Created by Fabian Seiler @ 22.07.24
"""
import json
import shutil
import os
from src.util import Logger


class ControlLogicGenerator:
    """
    This class is responsible for automatically creating the control logic for IMPLY and FALSE operations
    (via PWM signals in .csv files) depending on algorithm and topology given.
    """

    def __init__(self, config: dict):

        self.logger = Logger()
        self.logger.L.info(f"Initializing {self.__class__.__name__}")

        self.algo = config["algorithm"]
        # self.step_size = config["cycle_time"]
        self.memristors = config["memristors"]
        self.count = 0
        self.topology = config["topology"]
        self.output_path = f"./outputs/PWM_output/"
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        with open("./Structures/IMPLY_parameters.json", "r") as f:
            self.params = json.load(f)
        self.step_size = self.params["t_pulse"]

        # Create save files
        try:
            self.output_mem = [self.create_empty_csv(f"{self.output_path}{memristor}.csv")
                               for memristor in self.memristors]

            if self.topology == "Serial" or self.topology == "Serial-Mult":
                self.output_sw = [self.create_empty_csv(f"{self.output_path}{memristor}_sw.csv")
                                  for memristor in self.memristors]

            elif self.topology == "Semi-Serial":
                self.output_sw = []
                for j, memristor in enumerate(self.memristors):
                    if j < 2:
                        self.output_sw.append(self.create_empty_csv(f"{self.output_path}{memristor}_sw.csv"))
                    else:
                        self.output_sw.append(self.create_empty_csv(f"{self.output_path}{memristor}_sw1.csv"))
                        self.output_sw.append(self.create_empty_csv(f"{self.output_path}{memristor}_sw2.csv"))

            elif self.topology == "Semi-Parallel":
                self.output_sw = ([self.create_empty_csv(f"{self.output_path}{memristor}_sw.csv")
                                  for memristor in self.memristors]
                                  + [self.create_empty_csv(f"{self.output_path}S{j+1}.csv") for j in range(3)])
            # ADD NEW TOPOLOGIES HERE:
            # elif self.topology == "New topology":

            # Create lists with command content
            self.pwm_mem = [["0u,0"] for _ in self.output_mem]
            self.pwm_sw = [["0u,-100"] for _ in self.output_sw]

        except Exception as e:
            self.logger.L.error(f"Creation of PWM files failed at {self.__class__.__name__} due to: {e}")

    def read_algo(self) -> [[str]]:
        """
        Reads the algorithm and returns a cleaned up list with the commands for each section
        :return: [[str]]
        """
        with open(f"algorithms/{self.algo}", "r") as f:
            lines = f.readlines()

            if self.topology == "Serial" or self.topology == "Serial-Mult":
                lines_clean = [line.split(" ")[0] for line in lines]
            elif self.topology == "Semi-Serial" or self.topology == "Semi-Parallel":
                lines_sectioned = [line.split("|") for line in lines]
                lines_clean = [[cmd.strip() for cmd in line] for line in lines_sectioned]
            # ADD NEW TOPOLOGIES HERE:
            # elif self.topology == "New topology":
            else:
                self.logger.L.error(f"Invalid topology found: {self.topology}")
                raise NotImplementedError(f"Invalid topology {self.topology}")

            return lines_clean

    def eval_algo(self) -> None:
        """
        Evaluates an algorithm and writes PWM outputs to files, depending on the topology
        """

        # Clear the output folder
        shutil.rmtree(self.output_path)
        os.mkdir(self.output_path)
        self.count = 0

        # Load in the algorithm and iterate
        try:
            cmds = self.read_algo()
            for cmd in cmds:
                self.write_timestep(cmd)
        except Exception as e:
            self.logger.L.error(f"Failed to evaluate PWM outputs at {self.__class__.__name__} due to: {e}")

        try:
            if self.topology == "Serial" or self.topology == "Serial-Mult":
                # Add final lines to zero out the simulation after the steps
                for i, _ in enumerate(self.pwm_mem):
                    self.pwm_mem[i].append(f"{self.step_size * self.count + 0.001}u,0")
                    self.pwm_sw[i].append(f"{self.step_size * self.count + 0.001}u,-100")
                    self.pwm_mem[i].append(f"{self.step_size * (self.count + 1)}u,0")
                    self.pwm_sw[i].append(f"{self.step_size * (self.count + 1)}u,-100")

                # Write the content in each memristor + switches files
                for i, mem in enumerate(self.memristors):
                    with open(f"{self.output_path}{mem}.csv", "a") as f:
                        for line in self.pwm_mem[i]:
                            f.write(line + '\n')
                    with open(f"{self.output_path}{mem}_sw.csv", "a") as f:
                        for line in self.pwm_sw[i]:
                            f.write(line + '\n')

            elif self.topology == "Semi-Serial":
                # Add final lines to zero out the simulation after the steps
                for i, _ in enumerate(self.pwm_mem):
                    self.pwm_mem[i].append(f"{self.step_size * self.count + 0.001}u,0")
                    self.pwm_mem[i].append(f"{self.step_size * (self.count + 1)}u,0")
                for j, _ in enumerate(self.pwm_sw):
                    self.pwm_sw[j].append(f"{self.step_size * self.count + 0.001}u,-100")
                    self.pwm_sw[j].append(f"{self.step_size * (self.count + 1)}u,-100")

                # Write the content in each memristor + switches files
                for i, mem in enumerate(self.memristors):
                    with open(f"{self.output_path}{mem}.csv", "a") as f:
                        for line in self.pwm_mem[i]:
                            f.write(line + '\n')
                for j, sw in enumerate(self.output_sw):
                    sw = sw.split('/')[-1].split('.')[0]
                    with open(f"{self.output_path}{sw}.csv", "a") as f:
                        for line in self.pwm_sw[j]:
                            f.write(line + '\n')

            elif self.topology == "Semi-Parallel":
                # Add final lines to zero out the simulation after the steps
                for i, _ in enumerate(self.pwm_mem):
                    self.pwm_mem[i].append(f"{self.step_size * self.count + 0.001}u,0")
                    self.pwm_sw[i].append(f"{self.step_size * self.count + 0.001}u,-100")
                    self.pwm_mem[i].append(f"{self.step_size * (self.count + 1)}u,0")
                    self.pwm_sw[i].append(f"{self.step_size * (self.count + 1)}u,-100")

                # Write the content in each memristor + switches files
                for i, mem in enumerate(self.memristors):
                    with open(f"{self.output_path}{mem}.csv", "a") as f:
                        for line in self.pwm_mem[i]:
                            f.write(line + '\n')
                    with open(f"{self.output_path}{mem}_sw.csv", "a") as f:
                        for line in self.pwm_sw[i]:
                            f.write(line + '\n')

                # Write the content for the section switches
                switches_sec = ["S1", "S2", "S3"]
                for j, sw in enumerate(self.pwm_sw[-3:]):
                    with open(f"{self.output_path}{switches_sec[j]}.csv", "a") as f:
                        for line in sw:
                            f.write(line + '\n')

            # ADD NEW TOPOLOGIES HERE:
            # elif self.topology == "New topology":

        except Exception as e:
            self.logger.L.error(f"Writing PWM outputs to files failed at {self.__class__.__name__} due to: {e}")

        print("PWM Files written successfully")
        self.logger.L.info(f"PWM outputs written to files successfully")

    def write_timestep_serial(self, cmd: str) -> None:
        """
        Writes timestep in the Semi topology
        :param cmd: command line from the algorithm
        """
        # Filter out all active memristors
        digits = [int(cmd[i]) for i in range(len(cmd)) if cmd[i].isdigit()]

        for i, mem in enumerate(self.memristors):
            if i in digits:  # Check if memristor is used
                if cmd[0] == 'F':  # if false operation
                    self.pwm_mem[i].append(f"""{self.step_size * self.count + 0.001}u,{self.params["V_Reset"]}""")
                    self.pwm_sw[i].append(f"{self.step_size * self.count + 0.001}u,100")
                    self.pwm_mem[i].append(f"""{self.step_size * (self.count + 1)}u,{self.params["V_Reset"]}""")
                    self.pwm_sw[i].append(f"{self.step_size * (self.count + 1)}u,100")
                elif cmd[0] == 'I':
                    if i == digits[0]:
                        self.pwm_mem[i].append(f"""{self.step_size * self.count + 0.001}u,{self.params["V_Cond"]}""")
                        self.pwm_sw[i].append(f"{self.step_size * self.count + 0.001}u,100")
                        self.pwm_mem[i].append(f"""{self.step_size * (self.count + 1)}u,{self.params["V_Cond"]}""")
                        self.pwm_sw[i].append(f"{self.step_size * (self.count + 1)}u,100")
                    else:
                        self.pwm_mem[i].append(f"""{self.step_size * self.count + 0.001}u,{self.params["V_Set"]}""")
                        self.pwm_sw[i].append(f"{self.step_size * self.count + 0.001}u,100")
                        self.pwm_mem[i].append(f"""{self.step_size * (self.count + 1)}u,{self.params["V_Set"]}""")
                        self.pwm_sw[i].append(f"{self.step_size * (self.count + 1)}u,100")
            # If memristor is unused
            else:
                self.pwm_mem[i].append(f"{self.step_size * self.count + 0.001}u,0")
                self.pwm_sw[i].append(f"{self.step_size * self.count + 0.001}u,-100")
                self.pwm_mem[i].append(f"{self.step_size * (self.count + 1)}u,0")
                self.pwm_sw[i].append(f"{self.step_size * (self.count + 1)}u,-100")

    def write_timestep_semi_serial(self, cmd: str) -> None:
        """
        Writes timestep in the Semi-Serial topology
        :param cmd: command line from the algorithm
        """
        digits = [[int(cmd[i][j]) for j in range(len(cmd[i])) if cmd[i][j].isdigit()] for i in range(len(cmd))]
        for i, mem in enumerate(self.memristors):

            # Check if the memristor is used in this step
            if self.memristor_used(i, digits):
                # Iterate over Sections
                for section in range(len(digits)):
                    idx = i if i < 2 else (2 * i - 2 if section == 0 else 2 * i - 1)

                    if i in digits[section]:    # Check if memristor is in this section
                        if cmd[section][0] == 'F':  # if false operation
                            self.pwm_mem[i].append(f"""{self.step_size * self.count + 0.001}u,{self.params["V_Reset"]}""")
                            self.pwm_mem[i].append(f"""{self.step_size * (self.count + 1)}u,{self.params["V_Reset"]}""")
                            self.pwm_sw[idx].append(f"{self.step_size * self.count + 0.001}u,100")
                            self.pwm_sw[idx].append(f"{self.step_size * (self.count + 1)}u,100")

                        elif cmd[section][0] == 'I':  # If IMPLY operation
                            if i == digits[section][0]:
                                self.pwm_mem[i].append(f"""{self.step_size * self.count + 0.001}u,{self.params["V_Cond"]}""")
                                self.pwm_mem[i].append(f"""{self.step_size * (self.count + 1)}u,{self.params["V_Cond"]}""")
                                self.pwm_sw[idx].append(f"{self.step_size * self.count + 0.001}u,100")
                                self.pwm_sw[idx].append(f"{self.step_size * (self.count + 1)}u,100")
                            else:
                                self.pwm_mem[i].append(f"""{self.step_size * self.count + 0.001}u,{self.params["V_Set"]}""")
                                self.pwm_mem[i].append(f"""{self.step_size * (self.count + 1)}u,{self.params["V_Set"]}""")
                                self.pwm_sw[idx].append(f"{self.step_size * self.count + 0.001}u,100")
                                self.pwm_sw[idx].append(f"{self.step_size * (self.count + 1)}u,100")

                    # If memristor is not a or b => has two switches
                    elif mem not in ["a", "b"]:
                        self.pwm_sw[idx].append(f"{self.step_size * self.count + 0.001}u,-100")
                        self.pwm_sw[idx].append(f"{self.step_size * (self.count + 1)}u,-100")

                continue

            # Set voltage of memristor to 0 if unused
            self.pwm_mem[i].append(f"{self.step_size * self.count + 0.001}u,0")
            self.pwm_mem[i].append(f"{self.step_size * (self.count + 1)}u,0")

            if mem in ["a", "b"]:
                self.pwm_sw[i].append(f"{self.step_size * self.count + 0.001}u,-100")
                self.pwm_sw[i].append(f"{self.step_size * (self.count + 1)}u,-100")
            else:
                # Set voltage of corresponding switches to -100 when unused
                idx1 = 2 * i - 2
                idx2 = 2 * i - 1
                self.pwm_sw[idx1].append(f"{self.step_size * self.count + 0.001}u,-100")
                self.pwm_sw[idx1].append(f"{self.step_size * (self.count + 1)}u,-100")
                self.pwm_sw[idx2].append(f"{self.step_size * self.count + 0.001}u,-100")
                self.pwm_sw[idx2].append(f"{self.step_size * (self.count + 1)}u,-100")

    @staticmethod
    def memristor_used(mem_idx: int, digits: []) -> bool:
        """
        Checks if the memristor is used
        :param mem_idx: index of the memristor
        :param digits: list of digits used in the timestep
        :return: Boolean Value
        """
        for section in range(len(digits)):
            if mem_idx in digits[section]:
                return True
        return False

    def write_timestep_semi_parallel(self, cmd: str) -> None:
        """
        Writes timestep in the Semi-Parallel topology
        :param cmd: command line from the algorithm
        """
        digits = [[int(cmd[i][j]) for j in range(len(cmd[i])) if cmd[i][j].isdigit()] for i in range(len(cmd))]

        for i, mem in enumerate(self.memristors):
            mem_used = self.memristor_used(i, digits)

            if mem_used:
                for section in range(len(digits)):
                    if i in digits[section]:  # Check if memristor is used in this section
                        if cmd[section][0] == 'F':  # if false operation
                            self.pwm_mem[i].append(f"""{self.step_size * self.count + 0.001}u,{self.params["V_Reset"]}""")
                            self.pwm_mem[i].append(f"""{self.step_size * (self.count + 1)}u,{self.params["V_Reset"]}""")
                            self.pwm_sw[i].append(f"{self.step_size * self.count + 0.001}u,100")
                            self.pwm_sw[i].append(f"{self.step_size * (self.count + 1)}u,100")
                        elif cmd[section][0] == 'I':
                            if i == digits[section][0]:
                                self.pwm_mem[i].append(f"""{self.step_size * self.count + 0.001}u,{self.params["V_Cond"]}""")
                                self.pwm_mem[i].append(f"""{self.step_size * (self.count + 1)}u,{self.params["V_Cond"]}""")
                                self.pwm_sw[i].append(f"{self.step_size * self.count + 0.001}u,100")
                                self.pwm_sw[i].append(f"{self.step_size * (self.count + 1)}u,100")
                            else:
                                self.pwm_mem[i].append(f"""{self.step_size * self.count + 0.001}u,{self.params["V_Set"]}""")
                                self.pwm_mem[i].append(f"""{self.step_size * (self.count + 1)}u,{self.params["V_Set"]}""")
                                self.pwm_sw[i].append(f"{self.step_size * self.count + 0.001}u,100")
                                self.pwm_sw[i].append(f"{self.step_size * (self.count + 1)}u,100")
            else:
                # If memristor is unused
                self.pwm_mem[i].append(f"{self.step_size * self.count + 0.001}u,0")
                self.pwm_mem[i].append(f"{self.step_size * (self.count + 1)}u,0")
                self.pwm_sw[i].append(f"{self.step_size * self.count + 0.001}u,-100")
                self.pwm_sw[i].append(f"{self.step_size * (self.count + 1)}u,-100")

        # Check the switch position of S1,S2,S3
        if cmd[2] == 'NOP':
            # Operations are calculated in the sections
            self.pwm_sw[-2].append(f"{self.step_size * self.count + 0.001}u,-100")
            self.pwm_sw[-2].append(f"{self.step_size * (self.count + 1)}u,-100")

            for sects in [0, 1]:
                idx = -3 if sects == 0 else -1
                if cmd[sects] == 'NOP':
                    self.pwm_sw[idx].append(f"{self.step_size * self.count + 0.001}u,-100")
                    self.pwm_sw[idx].append(f"{self.step_size * (self.count + 1)}u,-100")
                else:
                    self.pwm_sw[idx].append(f"{self.step_size * self.count + 0.001}u,100")
                    self.pwm_sw[idx].append(f"{self.step_size * (self.count + 1)}u,100")
        else:
            # Operations are calculated between the sections
            self.pwm_sw[-2].append(f"{self.step_size * self.count + 0.001}u,100")
            self.pwm_sw[-2].append(f"{self.step_size * (self.count + 1)}u,100")

            res_switch = -3 if 0 in digits[-1] else -1
            self.pwm_sw[res_switch].append(f"{self.step_size * self.count + 0.001}u,100")
            self.pwm_sw[res_switch].append(f"{self.step_size * (self.count + 1)}u,100")
            other_switch = -3 if res_switch == -1 else -1
            self.pwm_sw[other_switch].append(f"{self.step_size * self.count + 0.001}u,-100")
            self.pwm_sw[other_switch].append(f"{self.step_size * (self.count + 1)}u,-100")

    def write_timestep(self, cmd: str) -> None:
        """
        Writes a timestep in the chosen topology
        :param cmd: command line from the algorithm
        """

        if self.topology == "Serial" or self.topology == "Serial-Mult":
            self.write_timestep_serial(cmd)

        elif self.topology == "Semi-Serial":
            self.write_timestep_semi_serial(cmd)

        elif self.topology == "Semi-Parallel":
            self.write_timestep_semi_parallel(cmd)
        # ADD NEW TOPOLOGIES HERE:
        # elif self.topology == "New topology":

        self.count = self.count + 1

    def create_empty_csv(self, name: str) -> str:
        """
        Creates a new empty csv file
        :param name: name of the file
        :return: name of the new csv file
        """
        with open(name, "w"):
            print(f"File {name} created!")
            self.logger.L.info(f"File {name} created!")
            return name

