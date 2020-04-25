import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' # url of data
    data = pd.read_csv(url,header=None) # load data
    return data.values # return data

def option_1(t):
    map_dict = {
        'Iris-setosa':1,
        'Iris-versicolor':0,
        'Iris-virginica':0
    }
    

def option_2():
    pass

def option_3():
    pass

data = load_data() # load data
number_of_patterns, number_of_attributes = data.shape # get the rows and columns
print(f'Number of patters: {number_of_patterns}\nNumber of attributes: {number_of_attributes}') # print row and columns

table_x = data[:, [0,1,2,3]] # keep only columns 1 2 and 3 from dataset

plt.scatter(np.array(data[:50,0]), np.array(data[:50,2]), marker='o', label='setosa') # scatter first 50 data and get columns 1 and 3. Label setosa
plt.scatter(np.array(data[50:100,0]), np.array(data[50:100,2]), marker='o', label='versicolor') # scatter from 50 to 100 data and get columns 1 and 3. Label versicolor
plt.scatter(np.array(data[100:150,0]), np.array(data[100:150,2]), marker='o', label='virginica') # scatter from 100 to 150 data and get columns 1 and 3. Label virginica
plt.xlabel('petal length') # set x label
plt.ylabel('sepal length') # set y lable
plt.legend() # show tags
plt.show() # open diagram

table_t = np.zeros(number_of_patterns)

ans = 'y'
while ans == 'y':
    print('1. Seperation of Iris-Setosa\n2. Seperation of Iris-Virginica\n3. Seperation of Iris-Versicolor ')
    opt = int(input('Chosse from 1 to 3: '))
    if opt == 1:
        option_1(table_t)