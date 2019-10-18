"""
Partition class (holds feature information, feature values, and labels for a
dataset). Includes helper class Example.
Author: Sara Mathieson + Tanjuma Haque
Date:
"""
import math
import numpy as np
from numpy import argsort


class Example:

    def __init__(self, features, label):
        """Helper class (like a struct) that stores info about each example."""
        # dictionary. key=feature name: value=feature value for this example
        self.features = features
        self.label = label # in {-1, 1}

class Partition:

    def __init__(self, data, F):
        """Store information about a dataset"""
        self.data = data # list of examples
        # dictionary. key=feature name: value=set of possible values
        self.F = F
        self.n = len(self.data)

    # TODO: implement entropy and information gain methods here!

    """ Function to find the probability of the classes"""
    def prob(self):
        i = 0
        for a in self.data:
            if a.label == -1:
                i = i + 1                #goes through each example's class and increments i (count for -1) accordingly

        i = i/(len(self.data))
        j = 1 - i                        #probability of +1
        return [i,j]                     #returns set of probability of -1 and +1

    """ Function to find the entropy of classes"""
    def find_entropy(self):
        p = self.prob()                                          #gets the list of probabilities
        e = -1 * (p[0]*math.log(p[0],2) + p[1]*math.log(p[1],2)) #computes the entropy
        return e

    """Function to find the conditional entropy of all the feature values of a feature"""
    def cond_entro(self, featrname):
        valset = self.F.get(featrname)              #gets the set of feature values
        ftdict = {}                                 #initializes a dictionary that is going to store the feature values as keys and their corresponding counts as values
        for val in valset:                          #loops through each feature value from the value set
            ftdict[val] = [0,0]                     #adds the feature value as key and (to initialize) 0 as values (i is the feature value + when class is 1 count, j is just feature value count) to the dictionary
        for x in self.data:                         #loops through the data to go through each example
            example_val = x.features.get(featrname) #finds the feature value for the current example and assigns it to a variable
            if (x.label == 1):                      #checks if label is 1
                ftdict[example_val][0] += 1         #increments both the counts
                ftdict[example_val][1] += 1
            else:
                ftdict[example_val][1] += 1         #increments the total count for that feature value
        ent = 0                                     #variable will hold the entropy at the end
        x = len(self.data)
        for key in ftdict:                          #iterates through the dictionary of feature values and their corresponding counts for computing probability
            f = ftdict[key]                         #gets the count list
            f0 = f[0]                               #gets the first count (i.e. - sunny + class 1) for the feature value
            f1 = f[1]                               #gets the 2nd count (i.e.- total sunny) for the feature value
            y = f1/x                                #probability of that feature value over total
            if f1 == 0:                             #in case if p f1 is zero, it will give an error so checking
                p = 0
            else:
                p = f0/f1
            p1 = 1-p                                #computes (sunny+ class -1)/total sunny
            if p == 0 or p1 == 0:                   #if either probability is zero then entropy for that feature value, we can just conclude (short-circuit) that entropy will be zero.
                e1 = 0
                e2 = 0
            else:
                e1 = -(p*math.log(p,2))             # calculates entropy for the feature value
                e2 = -(p1*(math.log(p1,2)))
            ent += y*(e1+e2)                        #adds entropy of each feature value

        return [featrname,ent]                       #returns final entropy for the feature

    """ Function to computer Information Gain"""
    def info_gain(self):
        x = self.find_entropy()
        i = 0
        z = len(self.F.keys())
        n1 = [None]*z
        for key in self.F:
            y = self.cond_entro(key)
            a = x - y[1]
            n1[i] = [y[0],a]
            i+=1


        n1 = np.asarray(n1)
        sortedn1 = n1[n1[:,1].argsort()[::-1]]    #sorts the array based on a column
        print("Info Gain:")
        for x in sortedn1:
            print (x[0], ",", x[1])
        print("___________________________________________")
        return sortedn1

    def best_feature(self):
        return (self.info_gain()[0][0])      # gets best feature from numpy array
