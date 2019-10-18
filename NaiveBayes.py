"""
Description: All the work of training the algorithm and predicting the labels for the test data happens
             here.
Author:Tanjuma H
Date: 9/29/19

"""
# imports
import math
from Partition import *
from fractions import Fraction

class NaiveBayes:

    def __init__ (self, partition):
        self.partition = partition #the partition for the train/test data in question

        self.k = partition.K #number of classes
        self.n = partition.n #numbeer of total number of examples
        self.F = partition.F #Feature dictionary with feature values
        self.examples = partition.data #gets the examples


        #finds the probabilities of each class values
        classes = set() #makes a list of all the class labels
        classdict ={}
        nK = [0]*(self.k)
        for x in self.examples: #iterates through list of examples
            nK[int(x.label)] += 1 #keeps track of the count for each label
            classes.add(int(x.label))
            y = int(x.label)    #gets the int of the label of x example
            if y in classdict:  #if label already exists as key then increment value by 1
                classdict[y] += 1
            else:
                classdict[y] = 1 #else make a new key and make 1 as value
        for x in range(self.k):
            y = classdict.get(x)  #gets the count for x class
            classdict[x] = math.log(y + 1) - math.log(len(self.examples) + self.k)  #computes the probabilities for each class

        self.classes = classes

        counts ={}   #to keep track of Nk (count of data points where y has a certain label)
        probabilities = {} #to keep track of the probabilities for each j,v,k

        for j in self.F: #iterates through the feature dictionary
            for ex in self.examples: #loops through the example list
                v = ex.features.get(j) #gets the feature value from the example for the current feature
                k = (int(ex.label))  #gets the class lbl of the example
                if (j, v, k) in counts: #checks if key exists
                    counts[(j, v, k)] += 1 #if yes, then increment the value by 1
                else:
                    counts[(j, v, k)] = 1 #else, make a new key with 1 as the value


        for j in self.F: #loops through the feature dictionary to get j
            for v in self.F.get(j): #lopps through the feature value set to get v
                for k in classes:  #loops through the class label list to get current k
                    if (j,v,k) not in counts: #checks if (j,v,k) key exist in counts
                        njvk = 0              #if not, then Njvk becomes zero
                    else:
                        njvk = counts[(j,v,k)] #else we get Njvk from counts
                    probabilities[(j,v,k)] = math.log(njvk + 1) - math.log(nK[k] + len(self.F[j]))

        self.probabilities = probabilities
        self.classdict = classdict


    def classify(self, example):
        classprobability = {} #dictionary to hold all the probabilities for each class for the example
        for k in self.classes: #loops through the classes
            x = 1  #to hold the product of probabilities
            for j in example: #iterating through the features
                v = example.get(j) #getting the feature value
                x = x + self.probabilities[(j,v,k)] #computing the total probabilities
            x = x + self.classdict[k] #multiplies the class probability
            classprobability[k] = x #adds the class label as key and as the class probability as value

        return (max(classprobability, key=classprobability.get)) #returns the class label with maximum probability
