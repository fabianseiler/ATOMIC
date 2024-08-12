"""
Created by Fabian Seiler @ 21.07.2024
"""
import os.path

import numpy as np
from src.util import Logger


class FunctionalValidation:
    """
    This class is responsible for creating a State History for each memristor
    and to validate the functionality of a given algorithm in IMPLY logic
    """

    def __init__(self, config: dict):

        self.logger = Logger()
        self.logger.L.info(f'Initializing class {self.__class__.__name__}')

        try:
            self.topology = config["topology"]    # ["Serial", "Semi-Serial", "Semi-Parallel"]

            self.algorithm = config["algorithm"]
            self.inputs = config["inputs"]
            self.work_memristors = config["work"]
            self.memristors = self.inputs + self.work_memristors
            self.memristor_count = len(self.memristors)

            no_inputs = len(config["inputs"])
            self.vec_len = 2 ** no_inputs

            inputs = [self.generate_input(i) for i in range(self.vec_len)]
            arrays = [np.array(input_array, dtype=np.uint8)[np.newaxis] for input_array in inputs]
            input_stack = np.vstack(arrays)
            input_stack = np.transpose(input_stack)[::-1]

            # Initialize the work memristors with 2 to see if the algorithm correctly works with arbitrary inputs
            work_array = np.array([2 for _ in range(self.vec_len)], dtype=np.uint8)[np.newaxis]
            work_stack = np.vstack([work_array for _ in self.work_memristors])

            # State and State Number
            self.state = np.vstack([input_stack, work_stack]).T
            self.state_num = []

            self.output_states = config["output_states"]
            # Correct Output States
            self.output_states_names = [array for array in self.output_states]
            self.outputs = [np.array(self.output_states[state], dtype=np.uint8) for state in self.output_states_names]

            # Create file to store state history
            if not os.path.exists("./outputs/State_History.txt"):
                open("./outputs/State_History.txt", "w")
            else:
                os.makedirs("./outputs/", exist_ok=True)
                with open("./outputs/State_History.txt", "w") as f:
                    f.write("")

        except Exception as e:
            self.logger.L.error(f"Initialization of {self.__class__.__name__} failed: {e}")

    def generate_input(self, index: int) -> [int]:
        """
        Function that generates input combinations depending on the input index
        """
        return [(index >> i) & 1 for i in range(len(self.inputs))]

    def print_states(self) -> None:
        """
        Function that prints out the logical states of each memristor
        """
        h_str = "#####################################"
        print(h_str)
        print(f"Applied operations: {self.get_state_number()}")
        header_str = " ".join([f"{mem} |" if len(mem) == 1 else f"{mem}|" for mem in self.memristors])
        print(header_str)
        print("--------------------")
        header_str += "\n--------------------"
        for i in range(pow(2, len(self.inputs))):
            header_str += "\n" + np.array2string(self.state[i, :], precision=0, separator=' | ')[1:-1]
            print(np.array2string(self.state[i, :], precision=0, separator=' | ')[1:-1])
        print(h_str)

        with (open("./outputs/State_History.txt", "a") as file):
            string = (h_str + "\n" + f"Applied operations: {self.get_state_number()}" + "\n" +
                      header_str + "\n" + h_str + "\n")
            file.write(string)
        self.logger.L.info(f"Applied Operations: {self.get_state_number()}")

    def get_state_number(self) -> [str]:
        """
        Return the current state number
        """
        return self.state_num

    def false_op(self, mem_nums: [int]) -> None:
        """
        Simulation of FALSE operation for all memristors in the mem_nums list
        :param mem_nums: list of target memristors
        """
        for mem_num in mem_nums:
            if mem_num not in range(self.memristor_count):
                self.logger.L.warning(f"Memristor {mem_num} is not in the list")
                raise ValueError(f"Incorrect memristor number: {mem_num}")

            self.state[:, mem_num] = np.zeros((1, 8))
        msg = "".join([f"{self.get_memristor(mem_num)}," for mem_num in mem_nums])[:-1]
        self.state_num.append(f"F[{msg}]")

    def imply_op(self, p: int, q: int) -> None:
        """
        Simulation of IMPLY operation (q' = p -> q) which updates states accordingly
        :param p: first input memristor
        :param q: second input / output memristor
        """
        if p not in range(self.memristor_count):
            self.logger.L.warning(f"Memristor {p} is not in the list")
            raise ValueError(f"Incorrect memristor number for P: {p}")
        if q not in range(self.memristor_count):
            self.logger.L.warning(f"Memristor {q} is not in the list")
            raise ValueError(f"Incorrect memristor number for Q: {q}")
        if p == q:
            self.logger.L.warning(f"Memristor {p} and Q {q} are same, which is not possible with IMPLY")
            raise ValueError(f"The memristors P and Q cannot be equal")

        self.state[:, q] = np.logical_or(1 - self.state[:, p], self.state[:, q])
        self.state_num.append(f"I[{self.get_memristor(p)},{self.get_memristor(q)}]")

    def calc_algorithm(self, plot_tt: bool = True) -> None:
        """
        This function evaluates an algorithm step by step, depending on the topology given
        :param plot_tt: Print the individual steps or not
        """

        # open defined algorithm pseudocode
        algo = self.algorithm
        with open(f"./algorithms/{algo}", "r") as f:
            lines = f.readlines()

            # If topology is Serial choose one operation per cycle and evaluate
            if self.topology == "Serial":
                for lnr, line in enumerate(lines):
                    if line[0] == 'F':
                        if line[2] == ',':
                            if line[4] == ',':
                                self.false_op([int(line[1]), int(line[3]), int(line[5])])
                            else:
                                self.false_op([int(line[1]), int(line[3])])
                        else:
                            self.false_op([int(line[1])])
                    elif line[0] == 'I':
                        self.imply_op(int(line[1]), int(line[3]))
                    else:
                        self.logger.L.error(f"Incorrect input for algorithm at {lnr}: {line}")
                        raise ValueError(f"Incorrect Function in algorithm line: {line}")
                    if plot_tt:
                        self.print_states()

            # If the topology has multiple parallel sections, evaluate them together
            elif self.topology == "Semi-Serial" or self.topology == "Semi-Parallel":
                for lnr, line in enumerate(lines):
                    line = line.split("|")
                    line = [line[i].strip() for i in range(len(line))]

                    # Iterate over sections to extract and evaluate commands
                    for section in range(len(line)):
                        if line[section][0] == 'F':
                            if len(line[section]) >= 3:
                                if line[section][2] == ',':
                                    self.false_op([int(line[section][1]), int(line[section][3])])
                            else:
                                self.false_op([int(line[section][1])])
                        elif line[section][0] == 'I':
                            self.imply_op(int(line[section][1]), int(line[section][3]))
                        elif line[section] == 'NOP':
                            pass
                        else:
                            self.logger.L.error(f"Incorrect input for algorithm at line {lnr}, "
                                                  f"section {section}: {line[section]}")
                            raise ValueError(f"Incorrect Function in algorithm line: {line[section]}")
                    if plot_tt:
                        self.print_states()

            # ADD NEW TOPOLOGIES HERE
            # elif.topology == "New topology":

        # Check if the operation leads to valid results
        self.check_if_valid()

    def check_if_valid(self) -> None:
        """
        Checks if every defined output state is reached or not
        :return:
        """
        for j, outs in enumerate(self.outputs):
            is_valid = False
            for i in range(len(self.memristors)):
                res = np.array_equal(self.state[:, i], outs)
                if res is True:
                    print(f"Output {self.output_states_names[j]} is correctly stored in {self.get_memristor(i)}")
                    is_valid = True
            if not is_valid:
                self.logger.L.warning(f"Output {outs} was not found in any memristive state")

    def get_memristor(self, num: int) -> str:
        """
        Returns the memristor name given num
        """
        return self.memristors[num]





