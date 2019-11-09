"""
Implements AdaBoost algorithm with decision stumps.
Authors:Tanjuma H
Date:10/31/19
"""

import util
import numpy as np
import math
from Partition import *
from DecisionStump import *

def weighted_error(ds, train_partition):
    """
    Function to compute the weighted error for a classifier
    """
    train_data = train_partition.data
    len_t = train_partition.n
    #initializes a list to hold the predicted labels for train data
    train_preds = [None]*len_t
    #initializes variable as 0 to hold the total weighted error
    w_error = 0
    #loops through the train data
    for i in range(len_t):
        #gets the predicted label for current train example
        y_pred = ds.classify(train_data[i], 0.5)
        #adds it to the list we will return
        train_preds[i] = y_pred
        #if predicted label doesn't match with real one
        if y_pred != train_data[i].label:
            #we add the data point's weight to the error
            w_error += train_data[i].weight
    return (w_error,train_preds)

def compute_score(w_error):
    """
    Computes the score for current classifer
    """
    x = (1-w_error)/w_error
    return (.5*math.log(x))

def update_weights(train_partition, train_preds, score):
    """
    Function to update the weights of each data point
    after using each classifer
    """
    train_data = train_partition.data
    len_t = train_partition.n
    #variable to hold the normalizing constant
    ct = 0
    #loops through the training data set
    for i in range(len_t):
        ex = train_data[i]
        #adds to the normalizing constant
        ct += (ex.weight*math.exp((-1*ex.label)*score*train_preds[i]))
    #computes the final constant
    ct = 1/ct
    #loops through train data again to update weights for each example
    for i in range(len_t):
        ex = train_data[i]
        new_weight = (ct*ex.weight*math.exp((-1*ex.label)*score*train_preds[i]))
        ex.set_weight(new_weight)

def construct_ensemble(opts, train_partition):
    """
    Constructs the ensemble of decision stumps
    """
    #initalizes a list to hold the decision stumps later
    ensemble_d = [None]*opts.T
    scorelist = [None]*opts.T
    for i in range(opts.T):
        #passes the current weighted partition into DecisionStump class
        one_d = DecisionStump(train_partition)
        (w_error, train_preds) = weighted_error(one_d, train_partition)
        scorelist[i] = compute_score(w_error)
        update_weights(train_partition, train_preds, scorelist[i])
        #adds the current decision stump to the ensemble list
        ensemble_d[i] = one_d
    return (ensemble_d, scorelist)

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

def finaloutput(ds_list, test_partition, scorelist, threshold):
    """
    Function to compute majority vote for each example and return a list of final predicted labels
    """
    len_t = len(test_partition.data)
    #initializes a list to hold the final predicted labels
    len_d = len(ds_list)
    final_pred = [None]*len_t
    #loops through every test example
    for i in range(len_t):
        tot = 0
        for j in range(len_d):
            tot += (scorelist[j]*ds_list[j][i])
        #if the sum is greater than threshold, then we know label should be 1
        if tot > threshold:
            final_pred[i] = 1
        else:
            #otherwise predict -1
            final_pred[i] = -1
    #returns the list of final predicted labels for test data
    return final_pred

def main():

    opts = util.parse_args()
    train_partition = util.read_arff(opts.train_filename)
    test_partition  = util.read_arff(opts.test_filename)
    #training happens to get ensemble list and corresponding score list
    (ds_ensemble, scorelist) = construct_ensemble(opts, train_partition)
    #testing starts..to get list of predicted labels from each classifier
    tested_list = testing(test_partition, ds_ensemble, opts.threshold)
    #gets the final predicted labels for test data
    finalpred_lst = finaloutput(tested_list,test_partition, scorelist, opts.threshold)
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
