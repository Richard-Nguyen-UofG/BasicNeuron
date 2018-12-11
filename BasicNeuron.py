# BasicNeuron
# Richard Nguyen
# 9 Dec 2018
#
# A neuron which learns given the age and wage of a person
# whether or not they'll buy this product (example data)

# TO DO:
# Automatically remove (or option) to remove outliers
# Input from a spreadsheet instead
# Add labels for graphs

import numpy as np
import matplotlib.pyplot as plt

# Each point represents a person; their age and salary, 
# and whether or not they would buy a product
fPeopleList = [[64.0, 73000.0, 1],
               [27.0, 37000.0, 0],
               [30.0, 45000.0, 0],
               [38.0, 50000.0, 0],
               [35.0, 49000.0, 0],
               [68.0, 78000.0, 1],
               [70.0, 84000.0, 1],
               [75.0, 96000.0, 1],
               [73.0, 82000.0, 1],
               [40.0, 35000.0, 0],
               [32.0, 38000.0, 0],
               [70.0, 70000.0, 1],
               [22.0,  9000.0, 0],
               [70.0, 65000.0, 1],
               [74.0, 80000.0, 1]]

# A person which only age, and salary is known
fPerson = [50.0, 50000.0]

iDATAAXISX = 100000
iDATAAXISY = 80

fLEARNINGSPEED = 0.02

iITERATIONS = 50000

def makeCostGraph( fAllCosts ):
    plt.plot(fAllCosts)

# Create a scatterplot of the fPeopleList
def make_graph():
    plt.axis([0, iDATAAXISY, 0, iDATAAXISX])
    plt.grid()
    
    # Plot each point
    for i in range(len(fPeopleList)):
        point = fPeopleList[i]
        colour = "red"

        if point[2] == 1:
            colour = "blue"

        plt.scatter(point[0], point[1], c = colour) #c= colour
    
    plt.scatter(fPerson[0], fPerson[1], c = "purple")
    plt.show()

# Prevents super very numbers
def getridofe(x):
    if "e" in str(x):
        return 0
    else:
        return x

def sigmoid(x):
    return getridofe( 1.0 / (1.0 + np.exp(-x)) )

def sigmoid_deriv(y):
    return sigmoid(y) * (1.0 - sigmoid(y))

#Takes in the sum of inputs to the neuron and applies a function (sigmoid)
def neuron(x):
    return sigmoid(x)


# Calculates coefficients to normalize data (to < 10)
fW1Coeff = 1.0
fW2Coeff = 1.0
for i in fPeopleList:
    while i[0]/fW1Coeff > 10:
        fW1Coeff *= 10
for e in fPeopleList:
    while e[1]/fW2Coeff > 10:
        fW2Coeff *= 10

fCostList = []
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

for i in range(iITERATIONS):
    # Chooses a random person to train with
    rand = np.random.randint(len(fPeopleList))
    point = fPeopleList[rand]
    
    fInputSum = (point[0]/fW1Coeff * w1) + (point[1]/fW2Coeff * w2) + b

    #What the NN thinks the answer is
    prediction = neuron(fInputSum)
    prediction = getridofe(prediction)
    #What the actual answer is
    target = point[2]
            
    #Cost Function
    cost = np.square( prediction - target )
    cost = getridofe(cost)

    # Weights are completely off, re-scramble weights
    if cost == 1.0:
        w1 = np.random.randn()
        w2 = np.random.randn()
        b = np.random.randn()
    
    fCostList.append(cost)

    cost_prediction_deriv = 2 * (prediction - target)
    prediction_deriv = sigmoid_deriv(fInputSum)
    
    #Derivatives of fInputSum with respect to ...
    fInputSum_w1_deriv = point[0]/fW1Coeff
    fInputSum_w2_deriv = point[1]/fW2Coeff
    fInputSum_b_deriv = 1
    
    # Cost derivatives with respect to ...
    # Move every variable closer to f'(0) = 0 w/ respect to itself
    cost_deriv_w1 = cost_prediction_deriv * prediction_deriv * fInputSum_w1_deriv
    cost_deriv_w2 = cost_prediction_deriv * prediction_deriv * fInputSum_w2_deriv
    cost_deriv_b = cost_prediction_deriv * prediction_deriv * fInputSum_b_deriv
    
    #With each iteration each variable gets closer to f'(0)
    w1 = w1 - (fLEARNINGSPEED * cost_deriv_w1)
    w2 = w2 - (fLEARNINGSPEED * cost_deriv_w2)
    b = b - (fLEARNINGSPEED * cost_deriv_b)

#Now that the NN is trained, predict the missing data
fInputSum = (fPerson[0]/fW1Coeff * w1) + (fPerson[1]/fW2Coeff * w2) + b
prediction = neuron(fInputSum)

#Display whether or not cost is actually decreasing or not    
plt.figure(1)
make_graph()
plt.figure(2)
makeCostGraph(fCostList)

print("Probability of purchase")
print( round(prediction, 6) )