"""
Decision tree data structure (recursive).
Author: Tanjuma Haque
Date:
"""
from Partition import *

"""Node class for allowing a Node object to be initialized later"""
class Node:

    def __init__(self):
        self.children = {}
        self.name = None
        self.labels = {}

"""Decision Tree class to hold all the related functions and constructor"""
class DecisionTree:

    def __init__(self, prtition, maxdepth):
        self.totexamples = prtition.data
        self.totfeatures = prtition.F


    """Recursive function to fill in the Decision tree that takes in the
       updated Partition(p), maximum depth, and updated current depth"""
    def constructsubtree (self, p, maxdepth, currentdepth):
        root = Node() #initializes an empty Node

        if self.stop(p, currentdepth, maxdepth) == True: #checks for stopping criteria
            majorlabel = self.mostlabels(p.data) #gets the majority label
            root.name = majorlabel #assigns the root name to majority label

        else:
            currentdepth += 1  #increments current depth
            ftr_dict = p.F     #stores the feature & feature value dictionary
            bestfeature = p.best_feature()  #gets the current best feature
            root.name = bestfeature #assigns the best feature to the root name (e.g.- Outlook)
            fval_list = ftr_dict[bestfeature] #gets the feature values for the best feature

            for k in fval_list: #loops through the f value list
                examplesubset = [] #to store example subset for f val
                copydict = ftr_dict.copy() #makes copy of main feature dictionary
                i = 0; #to count the 1 labels
                j = 0; #to count the -1 labels
                for ex in p.data: #loops through the list of examples
                    if ex.features[bestfeature] == k: #checks if fval in example and from fval list match
                        examplesubset.append(ex) #if they match, we add example to subset
                        if ex.label == 1:  #this is to keep track of the label distribution for current fval
                            i += 1
                        else:
                            j += 1

                        root.labels[k] = [j, i] #stores the fvals and their corresponding label distribution
                                                # for the feature name (root)

                root.labels["currentdepth"] = currentdepth #stores the current depth in the label attribute
                                                           # to keep track of the depth at each root node

                del copydict[bestfeature] #deletes the feature name from the copied dictionary

                pnew = Partition(examplesubset, copydict) #makes new Partition by passing in
                                                          #fval example subset and updated f dictionary

                root.children[k] = self.constructsubtree(pnew, maxdepth, currentdepth) #calls this function recursively
                                                                                       #on new partition etc
                                                                                       #with current fval as key
        return root #returns root (which essentially holds the entire tree)

    """Function to determine stopping criteria"""
    def stop(self, p, currentdepth, maxdepth):
        if (currentdepth == maxdepth or   #checks if maxdepth been reached
            len(p.F) == 0 or              #checks if feature list is empty
            len(p.data) == 0 or           #checks if example list has ran out
            self.samelabel(p) == True):   #checks if all the labels are same
            return True                   #returns True to stop

    """Function to check if labels in the example subset for fval is the same"""
    def samelabel(self, p):
        b = []
        for a in p.data:
             b.append(a.label)  #makes a list of all the labels in the examples
        if (len(set(b))) == 1:  #makes a set to see if all have same value or not
            return True         #returns True if all same

    """To check what the majority label is in an example subset"""
    def mostlabels (self, examples):
        i = 0
        j = 0
        for x in examples:
            if x.label == 1:
                i += 1
            else:
                j +=1
        if i > j:
            return 1
        else:
            return -1

    """Prints tree recursively"""
    def printtree(self, root):
            for x in root.children:
                if len(root.children[x].children) ==0 :
                    print ('\t'*(root.labels["currentdepth"]-1), root.name, "=", x, root.labels[x]," : ", root.children[x].name )
                else:
                    print('\t'*(root.labels["currentdepth"]-1), root.name, "=" , x, root.labels[x])
                self.printtree(root.children[x])
