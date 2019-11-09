"""
PART 2: Random Forests
Purpose: Implements Random Forests with decision stumps.
Authors:Tanjuma H
Date: 10/29/19
"""

import util
import numpy as np
from Partition import *
from DecisionStump import *

def construct_ensemble(T, train_partition):
    """
    Constructs the ensemble of decision stumps
    """
    #initalizes a list to hold the decision stumps later
    ensemble_d = [None]*T
    for i in range(T):
        # gets a partition with data (random+ w/ replacement) and F(random + without replacement)
        random_partition = util.bootstrap_partition(train_partition)
        #passes the newly generated partition into DecisionStump class
        one_d = DecisionStump(random_partition)
        #adds the current decision stump to the ensemble list
        ensemble_d[i] = one_d
    return ensemble_d

def testing(test_partition, ensemble, threshold):
    """
    Function to predict and store all the test examples' labels for each classifer
    """
    testexamples = test_partition.data
    #gets the length of the list of examples
    len_t = len(testexamples)
    #gets the number of decision stumps in the ensemble
    len_d = len(ensemble)
    #initializes a list to hold the final list of lists of predicted labels for a classifer
    ds_list = [None]*len_d
    #loops to use every classifier/decision stump
    for j in range(len_d):
        #initalizes the list that will hold the predicted labels for the current classifer
        y_preds = [None]*len_t
        #loops through all the test examples
        for i in range(len_t):
            #adds the predicted label for the current example
            y_preds[i] = ensemble[j].classify(testexamples[i], threshold)
        #adds the predicted labels list for the current classifer
        ds_list[j] = y_preds
    #returns something like [[1,1,-1], [1,1,1],..]
    return ds_list

def finaloutput(ds_list, test_partition):
    """
    Function to compute majority vote for each example and return a list of final predicted labels
    """
    len_t = len(test_partition.data)
    #initializes a list to hold the final predicted labels
    final_pred = [None]*len_t
    #loops through every test example
    for i in range(len_t):
        tot = 0
        #computes the sum of all the predicted labels corresponding to the current example, i.e.- decided by i
        tot = sum([x[i] for x in ds_list])
        #if the sum is greater than 0, then we know that there is more 1 than -1
        if tot > 0:
            #predict final label for current example to be 1
            final_pred[i] = 1
        else:
            #otherwise predict -1
            final_pred[i] = -1
    #returns the list of final predicted labels for test data
    return final_pred

def main():
    """
    Calls every function to implement Rain Forests
    """

    opts = util.parse_args()
    train_partition = util.read_arff(opts.train_filename)
    test_partition  = util.read_arff(opts.test_filename)
    #constructs our ensemble of decision stumps
    ds_ensemble = construct_ensemble(opts.T, train_partition)
    #constructs a list of lists of (predicted labels for all examples) for each classifer
    ds_list = testing(test_partition, ds_ensemble, opts.threshold)
    #gets the final predicted labels for test data by majority vote
    finalpred_lst = finaloutput(ds_list,test_partition)
    #contructs confusion matrix
    confusion_matrix = util.construct_cm(finalpred_lst, test_partition)
    #computes the true positive and false positive rates for the confusion matrix
    (true_pos, false_pos) = util.rates(confusion_matrix)
    #print statements
    print ("T:",opts.T, ", thresh: ", opts.threshold)
    print("        prediction    ")
    print("       -1        1")
    print("-1","|  ", confusion_matrix[0,0],"  ", confusion_matrix[0,1])
    print(" 1","|  ", confusion_matrix[1,0],"   ", confusion_matrix[1,1])
    print(" ")
    print ("false positive: ", false_pos)
    print ("true positive: ", true_pos)


if __name__ == "__main__":
    main()
