import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random
from sklearn.model_selection import train_test_split
from matplotlib.gridspec import GridSpec

class Plots():

    def basic_data(self,data):
        plt.scatter(np.array(data[:50,0]), np.array(data[:50,2]), marker='o', label='setosa') # scatter first 50 data and get columns 1 and 3. Label setosa
        plt.scatter(np.array(data[50:100,0]), np.array(data[50:100,2]), marker='o', label='versicolor') # scatter from 50 to 100 data and get columns 1 and 3. Label versicolor
        plt.scatter(np.array(data[100:150,0]), np.array(data[100:150,2]), marker='o', label='virginica') # scatter from 100 to 150 data and get columns 1 and 3. Label virginica
        plt.xlabel('sepal length') # set x label
        plt.ylabel('petal length') # set y lable
        plt.legend() # show tags
        plt.show() # open diagram

    def train_test(self,xtrain,xtest):
        # casting to np arrays for viable slicing
        xtrain = np.asarray(xtrain) 
        xtest = np.asarray(xtest)
        plt.scatter(np.array(xtrain[:,0]),np.array(xtrain[:,2]), marker='o', color='blue', label='training')
        plt.scatter(np.array(xtest[:,0]),np.array(xtest[:,2]), marker='o', color='red', label='test')
        plt.legend()
        plt.show()

    def show_guessing(self,ttrain,guesses):
        fig, axs = plt.subplots(2)
        fig.suptitle('Correct results vs guesses')
        for index in range(len(guesses)):
            axs[0].scatter(index,'Choosen class' if ttrain[index] == 1 else 'Other class', marker='o', color='blue', label='correct')
            axs[1].scatter(index,'Choosen class' if guesses[index] == 1 else 'Other class',marker='o', color='red', label='guesses')
        plt.show()

    def fold_results(self, fold_targets,fold_guesses):
        fig, axs = plt.subplots(3, 3)
        for i in range (3):
            for j in range(len(fold_guesses[i])):
                axs[0,i].scatter(j, np.array(fold_guesses)[i][j],c='tab:blue', marker='o')
                axs[0,i].scatter(j, np.array(fold_targets)[i][j],c='tab:red', marker='.')
        for i in range (3):
            for j in range(len(fold_guesses[i])):
                axs[1,i].scatter(j, np.array(fold_guesses)[i][j],c='tab:blue', marker='o')
                axs[1,i].scatter(j, np.array(fold_targets)[i][j],c='tab:red', marker='.')
        for i in range (3):
            for j in range(len(fold_guesses[i])):
                axs[2,i].scatter(j, np.array(fold_guesses)[i][j],c='tab:blue', marker='o')
                axs[2,i].scatter(j, np.array(fold_targets)[i][j],c='tab:red', marker='.')
        plt.legend(['Correct class','Guesses'])
        plt.show()

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

    def option_2_1(self,xtrain,xtest,ttrain,ttest,table_t,table_x):
        ans1 = int(input('Give max epochs: '))
        ans2 = float(input('Give a beta value: '))
        perc = Perceptron(ans1,ans2)
        perc.train_loop(xtrain,ttrain)
        guesses = perc.get_correct_results(xtest)
        p.show_guessing(ttest,guesses)
        print('Now testing with 9 folds')
        fold_guesses = []
        fold_t = []
        for _ in range(9):
            xtrain,xtest,ttrain,ttest = d.return_splits(table_t,table_x)
            perc2 = Perceptron(ans1,ans2)
            perc2.train_loop(xtrain,ttrain)
            guesses = perc2.get_correct_results(xtest)
            fold_guesses.append(guesses)
            fold_t.append(ttest)
        p.fold_results(fold_t,fold_guesses)

    def option_2_2(self):
        pass

    def option_2_3(self):
        pass

    def option_2_4(self):
        pass

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

    def return_splits(self,table_t,table_x):
        xtrain, xtest, ttrain, ttest = train_test_split(table_x, table_t, test_size=0.1)
        return xtrain,xtest,ttrain,ttest



class Menu():

    def attribute_menu(self,a,table_t,data):
        while True:
            print('1. Seperation of Iris-Setosa\n2. Seperation of Iris-Virginica\n3. Seperation of Iris-Versicolor ')
            opt = int(input('Choose from 1 to 3: '))
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

    def method_menu(self,a,xtrain,xtest,ttrain,ttest,table_t,table_x):
        while True:
            print('1. Perceptron\n2. Adaline\n3. Method of least squares\n4. Return to main menu')
            opt = int(input('Choose from 1 to 4: '))
            if opt == 1:
                a.option_2_1(xtrain,xtest,ttrain,ttest,table_t,table_x)
                break
            elif opt == 2:
                a.option_2_2()
                break
            elif opt == 3:
                a.option_2_3()
                break
            elif opt == 4:
                a.option_2_4()
                break
            else:
                print('Invalid option.Please give another one\n')


class Perceptron:
    
    def __init__(self,maxepochs,beta):
        self.epochs = maxepochs
        self.beta = beta
        self.weights = []

        for i in range(5):
            self.weights.append(0.0)

    def correction(self,x,target,prediction):
        new_weights = []
        for index,weight in enumerate(self.weights):
            new_weights.append(self.weights[index] + self.beta*(target - prediction) * x[index])
        return new_weights

    def normalize(self,val):
        if val < 0:
            return 0.0
        else:
            return 1.0

    def guess(self,xtrain):
        weighted_sum = 0.0
        for i in range(len(xtrain)):
            weighted_sum += xtrain[i]*self.weights[i]
        return self.normalize(weighted_sum)

    def train_loop(self,xtrain,ttrain):
        for counter in range(self.epochs):
            flag = True
            for i,x in enumerate(xtrain):
                g = self.guess(x)
                if g != ttrain[i]:
                    self.weights = self.correction(x,ttrain[i],g)
                    flag = False
            if flag == True:
                break

    def get_correct_results(self,xtest):
        guesses = []
        for x in xtest:
            weighted_sum = 0.0
            for i in range(len(x)):
                weighted_sum += x[i]*self.weights[i]
            if weighted_sum < 0:
                guesses.append(0)
            else:
                guesses.append(1)
        return guesses

class Adaline:
    pass

class LeastSquares:
    pass


def load_data():
    #Get irish data
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' # url of data
    data = pd.read_csv(url,header=None) # load data
    return data.values # return data

def run():
    #Show first menu and get new table t
    new_table_t = m.attribute_menu(a,table_t,data)
    #Get new table
    new_table_x = d.extend_x(table_x)

    #Init training-test sets
    xtrain = d.return_xtrain(new_table_x)
    xtest = d.return_xtest(new_table_x)
    ttrain = d.return_xtrain(new_table_t)
    ttest = d.return_ttest(new_table_t)

    #Show plot of training and test set
    p.train_test(xtrain,xtest)

    m.method_menu(a,xtrain,xtest,ttrain,ttest,new_table_t,new_table_x)




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
    p.basic_data(data) # print plot of all data

    #Init table t
    table_t = np.zeros(number_of_patterns) # fill array with zeros

    run()
   
