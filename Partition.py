"""
Example and Partition data structures.
Authors: Sara Mathieson + Tanjuma H.
Date: 10/28/19
"""
import math
import numpy as np
from numpy import argsort

class Example:

    def __init__(self, features, label, weight):
        """
        Class to hold an individual example (data point) and its weight.
        features -- dictionary of features for this example (key: feature name,
            value: feature value for this example)
        label -- label of the example (-1 or 1)
        weight -- weight of the example (starts out as 1/n)
        """
        self.features = features
        self.label = label
        self.weight = weight

    def set_weight(self, new):
        """Change the weight on an example (used for AdaBoost)."""
        self.weight = new

class Partition:

    def __init__(self, data, F):
        """
        Class to hold a set of examples and their associated features, as well
        as compute information about this partition (i.e. entropy).
        data -- list of Examples
        F -- dictionary (key: feature name, value: list of feature values)
        """
        self.data = data
        self.F = F
        self.n = len(self.data)

    def prob(self):
        """
         Function to find the probability of the classes
        """
        i = 0 #to hold weights for -1 labels
        j = 0 #to hold weights for +1
        for a in self.data: #goes through each example
            if a.label == -1:
                i = i + a.weight #adds the weights for -1
            else:
                j = j + a.weight #does the same for +1

        i = i/ (i+j)  #weighted probability of -1
        j = j/ (i+j)  #weighted probability of 1
        return [i,j]  #returns set of probability of -1 and +1

    def find_entropy(self):
        """
        Function to find the entropy of classes
        """
        p = self.prob()                                          #gets the list of probabilities
        e = -1 * (p[0]*math.log(p[0],2) + p[1]*math.log(p[1],2)) #computes the entropy
        return e

    def cond_entro(self, featrname):
        """
        Function to find the conditional entropy of all the feature values of a feature
        """
        valset = self.F.get(featrname)              #gets the set of feature values
        ftdict = {}                                 #initializes a dictionary that is going to store the feature values
                                                    # as keys and their corresponding counts as values
        for val in valset:                          #loops through each feature value from the value set
            ftdict[val] = [0,0]                     #adds the feature value as key and (to initialize) [0,0] is [weighted c for 1, weighted c for -1]
        totalweights = 0                            #to hold the total weights for all examples
        for x in self.data:                         #loops through the data to go through each example
            totalweights += x.weight
            example_val = x.features.get(featrname) #finds the feature value for the current example and assigns it to a variable
            if (x.label == 1):                      #checks if label is 1
                ftdict[example_val][0] += x.weight  #adds the weight for 1
            else:
                ftdict[example_val][1] += x.weight  #adds the weight for -1

        ent = 0                                     #variable will hold the entropy at the end
        for key in ftdict:                          #iterates through the dictionary of feature values and their
                                                    #corresponding counts for computing probability
            f = ftdict[key]                         #gets the count list
            f0 = f[0]                               #gets the first count (i.e. - sunny & class 1)
            f1 = f[1]                               #gets the 2nd count (i.e.- sunny & class -1)
            y = (f1+f0)/totalweights                #weighted probability of that feature value over total
            if (f0+f1) == 0:                        #in case if p f1 is zero, it will give an error so checking
                p = 0
            else:
                p = f0/(f0+f1)
            p1 = 1-p                                #computes (sunny+ class -1)/total sunny
            if p == 0 or p1 == 0:                   #if either probability is zero then entropy for that feature value, we can just
                                                    #conclude (short-circuit) that entropy will be zero.
                e1 = 0
                e2 = 0
            else:
                e1 = -(p*math.log(p,2))             # calculates entropy for the feature value
                e2 = -(p1*(math.log(p1,2)))
            ent += y*(e1+e2)                        #adds entropy of each feature value

        return [featrname,ent]                      #returns final entropy for the feature

    def info_gain(self):
        """
        Function to compute Information Gain
        """
        x = self.find_entropy()
        i = 0
        z = len(self.F.keys())
        n1 = [None]*z
        for key in self.F:
            y = self.cond_entro(key)
            a = x - y[1]
            n1[i] = (key, a)
            i+=1
        n1.sort(key=lambda tup: tup[1], reverse = True)
        # print("Info Gain:")
        # for x in n1:
        #     print (x[0], ",", x[1])
        return n1

    def best_feature(self):
        """
        Main function for finding best feature
        """
        return (self.info_gain()[0][0])      # gets best feature from numpy array

    def prob_pos(self):
        """
        Function for finding the probability for positive label in the leaf node
        """
        w = 0                 #to hold the weights for y =1
        tot = 0               #to hold total weights
        for x in self.data:   #loops through the partition's examples
            tot += x.weight   #adds all the examples' weights
            if x.label == 1:
                w += x.weight #adds weight to w only if example's label is 1
        if tot == 0:
            return 0
        else:
            return w/tot      #returns the weighted probability
