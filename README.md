# machine-learning-ann


## Design
A simple implementation of an ANN model for CS7267. 

Usage is shown by the included test file. 

## Output
Sample output for testing with the iris dataset is below:

```
Ran 1 test in 94.017s

OK
weight delta is less than 0.003000 after 14168 epochs
expecting Iris-setosa
[[0.8798784115260301, 'Iris-setosa'], [0.6348787767576033, 'Iris-versicolor'], [5.256623647215288e-06, 'Iris-virginica']]
expecting Iris-versicolor
[[0.3034907914341639, 'Iris-versicolor'], [0.26045611185322276, 'Iris-virginica'], [3.0074967852025754e-16, 'Iris-setosa']]
expecting Iris-virginica
[[0.9998885857096543, 'Iris-virginica'], [0.10965641139553296, 'Iris-versicolor'], [3.2926692496047785e-31, 'Iris-setosa']]
```

Sample output for XOR is below with a 3,2,2 layout:
```
Ran 1 test in 8.135s

OK

Process finished with exit code 0
for input [0,0]
[[0.9670665008118271, '0'], [0.03294916835736065, '1']]
for input [0,1]
[[0.9759833717212454, '1'], [0.024004496388652662, '0']]
for input [1,0]
[[0.9759047544801756, '1'], [0.024083086350132822, '0']]
for input [1,1]
[[0.9698180576201698, '0'], [0.030197353050351386, '1']]
```

Sample output for AND is below with a 3,2,2 layout. Note that it cannot find the solution for both true:
```
Ran 1 test in 52.537s

OK

Process finished with exit code 0
weight delta is less than 0.000010 after 630380 epochs
for input [0,0]
[[0.999971080619064, '0'], [3.3664454937567234e-05, '1']]
for input [0,1]
[[0.9966728207586415, '0'], [0.0032673454668099007, '1']]
for input [1,0]
[[0.9967470766444639, '0'], [0.0032789528467706635, '1']]
for input [1,1]
[[0.5000683587757113, '0'], [0.49993230679516315, '1']]
```

However--- adding an extra hidden layer (2,3,2) gives the following results for AND:
```
Ran 1 test in 46.186s

OK

Process finished with exit code 0
weight delta is less than 0.000010 after 433117 epochs
for input [0,0]
[[0.9998493533757875, '0'], [0.00015052535291171427, '1']]
for input [0,1]
[[0.997328650056038, '0'], [0.0026667166171506123, '1']]
for input [1,0]
[[0.9978205259599962, '0'], [0.0021889382942061027, '1']]
for input [1,1]
[[0.995098408614141, '1'], [0.004899339692300282, '0']]
```

Similar behavior was seen in the OR testing.

### ANN.py
ANN.py includes the ANNModel, a simple function for reading comma separated values from a text file, and a Node model. The ANNModel will take an argument to cause it to make and include a bias node.

### test_ANNModel.py
test_ANNModel.py includes sample usage for training and classifying data points.  


