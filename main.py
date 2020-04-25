import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Plots():

    def plot_data(self,data):
        plt.scatter(np.array(data[:50,0]), np.array(data[:50,2]), marker='o', label='setosa') # scatter first 50 data and get columns 1 and 3. Label setosa
        plt.scatter(np.array(data[50:100,0]), np.array(data[50:100,2]), marker='o', label='versicolor') # scatter from 50 to 100 data and get columns 1 and 3. Label versicolor
        plt.scatter(np.array(data[100:150,0]), np.array(data[100:150,2]), marker='o', label='virginica') # scatter from 100 to 150 data and get columns 1 and 3. Label virginica
        plt.xlabel('petal length') # set x label
        plt.ylabel('sepal length') # set y lable
        plt.legend() # show tags
        plt.show() # open diagram

class User_Action():

    def option_1(self,t,data):
        #Init dictionary
        map_dict = {
            'Iris-setosa':1,
            'Iris-versicolor':0,
            'Iris-virginica':0
        }
        #Loop through items 
        for counter,data_item in enumerate(data):
            #Replace number with the corresponding from the dictionary
            t[counter] = map_dict[data_item[4]]
        #Return new table
        return t

    def option_2(self,t,data):
        #Init dictionary
        map_dict = {
            'Iris-setosa':0,
            'Iris-versicolor':0,
            'Iris-virginica':1
        }
        #Loop through items 
        for counter,data_item in enumerate(data):
            #Replace number with the corresponding from the dictionary
            t[counter] = map_dict[data_item[4]]
        #Return new table
        return t

    def option_3(self,t,data):
        #Init dictionary
        map_dict = {
            'Iris-setosa':0,
            'Iris-versicolor':1,
            'Iris-virginica':0
        }
        #Loop through items 
        for counter,data_item in enumerate(data):
            #Replace number with the corresponding from the dictionary
            t[counter] = map_dict[data_item[4]]
        #Return new table
        return t

class Data_split():

    def return_xtrain(self,table_x):
        #Init arrat
        xtrain = []
        #Loop 3 times
        for i in range(3):
            #Init the steps
            start = 50 * i
            end = 50 * (i + 1) - 10
            #Extend the array
            xtrain.extend(table_x[start:end])
        return xtrain

    def return_xtest(self,table_x):
        #Init array 
        xtest = []
        #Loop 3 times
        for i in range(3):
            #Init steps
            start = (i + 1) * 50 - 10
            end = start + 10
            #Extend array
            xtest.extend(table_x[start:end])
        return xtest

    def return_ttrain(self,table_t):
        pass

    def return_ttest(self,table_t):
        pass

def load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' # url of data
    data = pd.read_csv(url,header=None) # load data
    return data.values # return data

if __name__ == '__main__':
    data = load_data() # load data
    number_of_patterns, number_of_attributes = data.shape # get the rows and columns
    print(f'Number of patters: {number_of_patterns}\nNumber of attributes: {number_of_attributes}') # print row and columns

    p = Plots()
    d = Data_split()
    a = User_Action()

    table_x = data[:, [0,1,2,3]] # keep only columns 1 2 and 3 from dataset

    p.plot_data(data) # print plot of all data

    table_t = np.zeros(number_of_patterns) # fill array with zeros

    ans = 'y'
    while ans == 'y':
        print('1. Seperation of Iris-Setosa\n2. Seperation of Iris-Virginica\n3. Seperation of Iris-Versicolor ')
        opt = int(input('Chosse from 1 to 3: '))
        if opt == 1:
            new_table_t = a.option_1(table_t,data)
        elif opt == 2:
            new_table_t = a.option_2(table_t,data)
        elif opt == 3:
            new_table_t = a.option_3(table_t,data)
        else:
            print('Invalid option.Please give another one')
            continue
        xtrain = d.return_xtrain(table_x)
        xtest = d.return_xtest(table_x)
