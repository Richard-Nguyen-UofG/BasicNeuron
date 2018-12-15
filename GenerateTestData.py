# GenerateTestData.py
# Richard Nguyen
# 15 Dec 2018
# Generates data for the neuron to use

#Import Libraries
import random
from pandas import DataFrame

# Work in Progress
def takeParameters():
    sCategory1 = "Age"
    sCategory2 = "Wage"
    iPurchase = 0 #0or1
    
    iPoints = 100
    print(sCategory1 + sCategory2 + iPurchase + iPoints)

# Generates whether or not this person is a buy, based off a probability
def generateRandom(x):
    point = random.randint(0,11)/10.0
    if point < x:
        point = 1.0
    else:
        point = 0.0
    return point

# Generate dataset
def generateData():
    iPoints = 1000
   
    personList = []
    
    for i in range(0, iPoints):
        p1 = random.randint(1,101)
        p2 = random.randint(1000,100001)
        
        if p2 >= p1*1000:
            p3 = generateRandom(0.7)
        elif p2 < p1*1000:
            p3 = generateRandom(0.3)
        else:
            p3 = generateRandom(0.5)
        #if p2 > 100*(p1-50)*(p1-50) + 5000:
        #    p3 = 1.0
        #else:
        #    p3 = 0.0
        
        #if p1 > 45.0:
        #    p3 = 1.0
        #else:
        #    p3 = 0.0
        person = (p1, p2, p3)
        personList.append(person)

    return personList

people = generateData()
df = DataFrame(people, columns = ['Age', 'Wage', 'Purchase'])
df.to_excel('dataset.xlsx', sheet_name='sheet1', index=False)
print(df)
        