# BasicNeuron
# Richard Nguyen
# 9 Dec 2018
#
# A neuron which learns, given the age and wage of a person
# whether or not they'll buy a unnamed product (example data)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Each tuple represents a person; their age, salary, 
# and whether or not they would buy a eg. a house
DataF = pd.read_excel("dataset.xlsx",sheet_name='sheet1')
peopleList = DataF.values

# A person for which only age and salary is known
person = [40.0, 80000.0]

# Graph boundaries
xAXIS = 105000
yAXIS = 105

fLEARNINGSPEED = 0.02
iITERATIONS = 30000


# Plots the cost of 
def makeCostGraph( fAllCosts ):
    plt.plot(fAllCosts)


# Plots the decision boundary and each person onto the scatterplot
def make_graph():
    plt.axis([0, yAXIS, 0, xAXIS])
    plt.grid()
    
    # Displays the neural network's decision boundary
    for i in range(0,51):
        for e in range(0, 51):
            col = "yellow"
            num = (i/w1Coeff * w1*2) + (e*1000/w2Coeff * w2*2) + b
            ped = neuron(num)
            
            if ped >= 0.5:
                col = "green"
        
            plt.scatter(i*2,2*e*1000,c=col)
    
    # Plots each person in the dataset
    for i in range(len(peopleList)):
        point = peopleList[i]
        colour = "red"

        if point[2] == 1:
            colour = "blue"

        plt.scatter(point[0], point[1], c = colour) 
    
    plt.scatter(person[0], person[1], c = "purple")
    
    plt.show()


# Prevents hard to work with numbers
def getridofe(x):
    if "e" in str(x):
        return 0
    else:
        return x

# The sigmoid function
def sigmoid(x):
    return getridofe( 1.0 / (1.0 + np.exp(-x)) )

# The derivative of the sigmoid function
def sigmoid_deriv(y):
    return sigmoid(y) * (1.0 - sigmoid(y))


# Takes in the sum of inputs to the neuron and applies a function (sigmoid)
def neuron(x):
    return sigmoid(x)


# Calculates coefficients to normalize data (to < 10)
w1Coeff = 1.0
w2Coeff = 1.0
for i in peopleList:
    while i[0]/w1Coeff > 10:
        w1Coeff *= 10
for e in peopleList:
    while e[1]/w2Coeff > 10:
        w2Coeff *= 10

# Neuron related variables
costList = []
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()


# Begin training the neuron
for i in range(iITERATIONS):
    # Chooses a random person to train with
    rand = np.random.randint(len(peopleList))
    point = peopleList[rand]
    
    inputSum = (point[0]/w1Coeff * w1) + (point[1]/w2Coeff * w2) + b

    # What the NN thinks the answer is
    prediction = neuron(inputSum)
    prediction = getridofe(prediction)
    #What the actual answer is
    target = point[2]
            
    # Cost Function
    cost = np.square( prediction - target )
    cost = getridofe(cost)

    # Weights are completely off, re-scramble weights
    if cost == 1.0:
        w1 = np.random.randn()
        w2 = np.random.randn()
        b = np.random.randn()
    
    costList.append(cost)

    cost_prediction_deriv = 2 * (prediction - target)
    prediction_deriv = sigmoid_deriv(inputSum)
    
    # Derivatives of inputSum with respect to ...
    inputSum_w1_deriv = point[0]/w1Coeff
    inputSum_w2_deriv = point[1]/w2Coeff
    inputSum_b_deriv = 1
    
    # Cost derivatives with respect to ...
    # Move every variable closer to f'(0) = 0 w/ respect to itself
    cost_deriv_w1 = cost_prediction_deriv * prediction_deriv * inputSum_w1_deriv
    cost_deriv_w2 = cost_prediction_deriv * prediction_deriv * inputSum_w2_deriv
    cost_deriv_b = cost_prediction_deriv * prediction_deriv * inputSum_b_deriv
    
    # With each iteration each variable gets closer to f'(0)
    w1 = w1 - (fLEARNINGSPEED * cost_deriv_w1)
    w2 = w2 - (fLEARNINGSPEED * cost_deriv_w2)
    b = b - (fLEARNINGSPEED * cost_deriv_b)


# Now that the NN is trained, predict the missing data
inputSum = (person[0]/w1Coeff * w1) + (person[1]/w2Coeff * w2) + b
prediction = neuron(inputSum)


# Display a graph regarding the cost (inaccuracy*) of the neuron   
plt.figure(1)
make_graph()
plt.figure(2)
makeCostGraph(costList)

print("Probability of purchase")
print( round(prediction, 6) )