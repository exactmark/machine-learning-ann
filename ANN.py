# // Course: CS7267
# // Student name: Mark Fowler
# // Student ID: mfowle19
# // Assignment #: #5
# // Due Date: November 18, 2019
# // Signature:
# // Score:

from random import random
import math


def read_csv(path):
    this_out = []
    with open(path, "r") as in_file:
        for single_line in in_file.readlines():
            single_line = single_line.rstrip().split(',')
            this_out.append(single_line)
    return this_out


class node(object):
    def __init__(self, position=None, learning_rate=0.1):
        self.down_list = []
        self.up_list = []
        self.down_weight_list = []
        self.current_value = None
        self.list_pos = position
        self.output_val = None
        self.expected_val = None
        self.this_class = None
        self.delta = None
        self.learning_rate = learning_rate

    def __find_this_weight(self, target_down_node):
        for x, single_down_node in enumerate(self.down_list):
            if single_down_node == target_down_node:
                return self.down_weight_list[x]

    def calculate_new_value(self):
        self.current_value = 0
        for single_up_node in self.up_list:
            self.current_value += single_up_node.output_val * single_up_node.__find_this_weight(self)
        self.output_val = 1 / (1 + (math.e ** (-1 * self.current_value)))

    def calc_output_delta(self):
        working = self.output_val * (1 - self.output_val) * (self.expected_val - self.output_val)
        self.delta = working

    def calc_hidden_delta(self):
        delta_sum = 0
        for x, down_node in enumerate(self.down_list):
            delta_sum += down_node.delta * self.down_weight_list[x]
        working = delta_sum * (self.output_val * (1 - self.output_val))
        self.delta = working

    def apply_deltas(self):
        for x, single_node in enumerate(self.down_list):
            self.down_weight_list[x] += (self.output_val * self.learning_rate * single_node.delta)


class ANNModel(object):
    def __init__(self, input: int, hidden: int, output: int, output_classes, learning_rate=0.1, verbose=False,
                 add_bias=False,weight_bail = 0.003):
        self.verbose = verbose
        self.input_nodes = []
        self.hidden_nodes = []
        self.output_nodes = []
        self.add_bias = add_bias
        self.weight_bail = weight_bail
        if add_bias:
            input += 1
        self.learning_rate = learning_rate
        for x in range(input):
            self.input_nodes.append(node(x, self.learning_rate))
        for x in range(hidden):
            self.hidden_nodes.append(node(x, self.learning_rate))
        for x in range(output):
            self.output_nodes.append(node(x, self.learning_rate))
            self.output_nodes[x].this_class = output_classes[x]
        self.__create_network()

    def __create_network(self):
        directions = [[self.input_nodes, self.hidden_nodes], [self.hidden_nodes, self.output_nodes]]
        for from_list, to_list in directions:
            for x in from_list:
                for y in to_list:
                    x.down_list.append(y)
                    x.down_weight_list.append(random())
                    y.up_list.append(x)
        if self.verbose:
            print("create network done")

    def train(self, input_stream):
        if self.verbose:
            print("starting training")
            print(input_stream)
        num_epochs = 1000000
        this_iter = 0
        last_weights = self.__get_weight_list()
        while this_iter < num_epochs:
            this_iter += 1
            for single_line in input_stream:
                self.__clear_values()
                for i, this_val in enumerate(single_line[:-1]):
                    self.input_nodes[i].output_val = float(this_val)
                if self.add_bias:
                    self.input_nodes[-1].output_val = 1
                for single_node in self.output_nodes:
                    if single_line[-1] == single_node.this_class:
                        single_node.expected_val = 1
                    else:
                        single_node.expected_val = 0
                self.__calc_network()
                self.__calc_output_delta()
                self.__calc_hidden_delta()
                self.__apply_weights()
            new_weight_list = self.__get_weight_list()
            weight_sum = 0
            for x in range(len(new_weight_list)):
                weight_sum+= abs(last_weights[x]-new_weight_list[x])
            if weight_sum<self.weight_bail:
                print("weight delta is less than %f after %i epochs"%(self.weight_bail,this_iter))
                this_iter=num_epochs
            last_weights = new_weight_list

    def __get_weight_list(self):
        weight_list =[]
        for single_node in (self.input_nodes+self.hidden_nodes):
            weight_list+=single_node.down_weight_list
        return weight_list

    def __clear_values(self):
        list_list = [self.input_nodes, self.hidden_nodes, self.output_nodes]
        for single_list in list_list:
            for single_node in single_list:
                single_node.current_value = None

    def __calc_network(self):
        for single_node in self.hidden_nodes:
            single_node.calculate_new_value()
        for single_node in self.output_nodes:
            single_node.calculate_new_value()

    def calc_output_delta(self):
        self.__calc_output_delta()

    def calc_hidden_delta(self):
        self.__calc_hidden_delta()

    def apply_weights(self):
        self.__apply_weights()

    def __calc_output_delta(self):
        for single_node in self.output_nodes:
            single_node.calc_output_delta()
            # if self.verbose:
            #     print(single_node.delta)

    def __calc_hidden_delta(self):
        for single_node in self.hidden_nodes:
            single_node.calc_hidden_delta()
            # if self.verbose:
            #     print(single_node.delta)

    def __apply_weights(self):
        for single_node in self.input_nodes:
            single_node.apply_deltas()
        for single_node in self.hidden_nodes:
            single_node.apply_deltas()

    def classify(self, input):
        self.__clear_values()
        for i, this_val in enumerate(input):
            self.input_nodes[i].output_val = float(this_val)
        if self.add_bias:
            self.input_nodes[-1].output_val = 1
        self.__calc_network()
        if self.verbose:
            for single_node in self.output_nodes:
                print(single_node.output_val)
        output_list = []
        for single_node in self.output_nodes:
            output_list.append([single_node.output_val, single_node.this_class])
        output_list.sort(reverse=True)
        return output_list
