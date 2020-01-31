import random
from ANN import ANNModel,read_csv

random.seed(0)
my_model = ANNModel(2, 2, 2, ["0", "1"], verbose=True)
input_list = read_csv("XOR.csv")
# input_list=input_list[1:2]
my_model.train(input_list)