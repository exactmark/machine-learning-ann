# // Course: CS7267
# // Student name: Mark Fowler
# // Student ID: mfowle19
# // Assignment #: #5
# // Due Date: November 18, 2019
# // Signature:
# // Score:


from unittest import TestCase
from ANN import ANNModel, read_csv
import random


class TestANNModel(TestCase):
    def test_xor(self):
        random.seed(1)
        my_model = ANNModel(2, 2, 2, ["0", "1"], verbose=False, add_bias=True, weight_bail=0.0001)
        input_list = read_csv("XOR.csv")
        my_model.train(input_list)
        for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            print("for input [%i,%i]" % (x[0], x[1]))
            print(my_model.classify(x))

    def test_and(self):
        # random.seed(1)
        my_model = ANNModel(2, 3, 2, ["0", "1"], verbose=False, add_bias=True, weight_bail=0.00001)
        input_list = read_csv("AND.csv")
        my_model.train(input_list)
        for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            print("for input [%i,%i]" % (x[0], x[1]))
            print(my_model.classify(x))

    def test_or(self):
        random.seed(1)
        my_model = ANNModel(2, 3, 2, ["0", "1"], verbose=False, add_bias=True, weight_bail=0.00001)
        input_list = read_csv("OR.csv")
        my_model.train(input_list)
        for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            print("for input [%i,%i]" % (x[0], x[1]))
            print(my_model.classify(x))

    def test_nand(self):
        random.seed(1)
        my_model = ANNModel(2, 2, 2, ["0", "1"], verbose=False, add_bias=True, weight_bail=0.00001)
        input_list = read_csv("NAND.csv")
        my_model.train(input_list)
        for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            print("for input [%i,%i]" % (x[0], x[1]))
            print(my_model.classify(x))

    def test_iris(self):
        random.seed(1)
        my_model = ANNModel(4, 4, 3, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"], verbose=False,
                            add_bias=False)
        input_list = read_csv("iris.data")
        # input_list=input_list[1:2]
        my_model.train(input_list)
        print("expecting Iris-setosa")
        print(my_model.classify([5.1, 3.5, 1.4, 0.2]))
        print("expecting Iris-versicolor")
        print(my_model.classify([7.0, 3.2, 4.7, 1.4]))
        print("expecting Iris-virginica")
        print(my_model.classify([6.3, 3.3, 6.0, 2.5]))

    def test_slides_example_classify(self):
        my_model = self.create_slide_example_model()
        my_model.classify([10, 30, 20])
        self.assertAlmostEqual(my_model.output_nodes[0].output_val, 0.750, 3)
        self.assertAlmostEqual(my_model.output_nodes[1].output_val, 0.957, 3)
        # print("done")

    def test_slides_example_training(self):
        my_model = self.create_slide_example_model()
        my_model.classify([10, 30, 20])
        my_model.calc_output_delta()
        my_model.calc_hidden_delta()
        my_model.apply_weights()
        # self.assertAlmostEqual(my_model.input_nodes[0].down_weight_list[0],0.1999295,7)
        # self.assertAlmostEqual(my_model.input_nodes[0].down_weight_list[1],0.69741,5)
        for single_node in my_model.input_nodes:
            for single_weight in single_node.down_weight_list:
                print(single_weight)

    def create_slide_example_model(self):
        my_model = ANNModel(3, 2, 2, ["1", "0"], verbose=True)
        my_model.input_nodes[0].down_weight_list = [0.2, 0.7]
        my_model.input_nodes[1].down_weight_list = [-0.1, -1.2]
        my_model.input_nodes[2].down_weight_list = [0.4, 1.2]
        my_model.hidden_nodes[0].down_weight_list = [1.1, 3.1]
        my_model.hidden_nodes[1].down_weight_list = [0.1, 1.17]
        my_model.output_nodes[0].expected_val = 1
        my_model.output_nodes[1].expected_val = 0
        return my_model

    def test_classify(self):
        pass
        # self.fail()
