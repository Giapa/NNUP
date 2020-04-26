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

class Data():

    def extend_x(self,table_x):
        new_table_x = []
        for table in table_x:
            new_table = np.hstack([table,1])
            new_table_x.append(new_table)
        return new_table_x

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
        #Init arrat
        ttrain = []
        #Loop 3 times
        for i in range(3):
            #Init the steps
            start = 50 * i
            end = 50 * (i + 1) - 10
            #Extend the array
            ttrain.extend(table_t[start:end])
        return ttrain

    def return_ttest(self,table_t):
        #Init array 
        ttest = []
        #Loop 3 times
        for i in range(3):
            #Init steps
            start = (i + 1) * 50 - 10
            end = start + 10
            #Extend array
            ttest.extend(table_t[start:end])
        return ttest

class Menu():

    def attribute_menu(self,a,table_t,data):
        while True:
            print('1. Seperation of Iris-Setosa\n2. Seperation of Iris-Virginica\n3. Seperation of Iris-Versicolor ')
            opt = int(input('Chosse from 1 to 3: '))
            if opt == 1:
                new_table_t = a.option_1(table_t,data)
                return new_table_t
            elif opt == 2:
                new_table_t = a.option_2(table_t,data)
                return new_table_t
            elif opt == 3:
                new_table_t = a.option_3(table_t,data)
                return new_table_t
            else:
                print('Invalid option.Please give another one\n')

def load_data():
    #Get irish data
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' # url of data
    data = pd.read_csv(url,header=None) # load data
    return data.values # return data

if __name__ == '__main__':

    data = load_data() # load data

    number_of_patterns, number_of_attributes = data.shape # get the rows and columns
    print(f'Number of patters: {number_of_patterns}\nNumber of attributes: {number_of_attributes}') # print row and columns

    #Initialize classes
    p = Plots()
    d = Data()
    a = User_Action()
    m = Menu()

    #Init table x
    table_x = data[:, [0,1,2,3]] # keep only columns 1 2 and 3 from dataset

    #Print basic plot
    p.plot_data(data) # print plot of all data

    #Init table t
    table_t = np.zeros(number_of_patterns) # fill array with zeros

    #Show first menu and get new table t
    new_table_t = m.attribute_menu(a,table_t,data)

    #Get new table
    new_table_x = d.extend_x(table_x)

    #Init training-test sets
    xtrain = d.return_xtrain(new_table_x)
    xtest = d.return_xtest(new_table_x)
    ttrain = d.return_xtrain(new_table_t)
    ttest = d.return_ttest(new_table_t)

