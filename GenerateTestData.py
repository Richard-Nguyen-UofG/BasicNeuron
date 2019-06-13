# GenerateTestData.py
# Richard Nguyen
# 15 Dec 2018
# Generates example people data (age, salary, purchase unnamed item)

# Import Libraries
import random
from pandas import DataFrame

# Generates dataset
def generateData():
    iPoints = 1000
   
    personList = []
    
    for i in range(0, iPoints):
        age = random.randint(1,100)
        salary = random.randint(1000,100000)
        
        # Determines function of data (decision boundary)
        if salary > (age*400) + 35000:
            purchase = 1.0
        else:
            purchase = 0.0

        person = (age, salary, purchase)
        personList.append(person)

    return personList

people = generateData()
df = DataFrame(people, columns = ['Age', 'Salary', 'Purchase'])
df.to_excel('dataset.xlsx', sheet_name='sheet1', index=False)
print(df)
        