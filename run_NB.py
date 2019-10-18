"""
Description: The main run happens here, where all the other classes are imported
Author:Tanjuma Haque
Date: 9/29/19
"""
# python imports
import util
import sys
import numpy as np
from NaiveBayes import *
from Partition import *


def main():
    # Process the data
    opts = util.parse_args()
    train_partition = util.read_arff(opts.train_filename)
    test_partition  = util.read_arff(opts.test_filename)

    # sanity check
    print("num train =", train_partition.n, ", num classes =", train_partition.K)
    print("num test  =", test_partition.n, ", num classes =", test_partition.K)

    nb_model = NaiveBayes(train_partition)

    y_real = [] #list of real y's
    y_h = [] #list of predicted y's
    for example in test_partition.data: #loops through test example list
        y_hat = nb_model.classify(example.features) #calls classify on each example's feature
        y_real.append(int(example.label)) #appends the test data's label to y_real
        y_h.append(y_hat) #appends the predicted label to y_h\

    ln = len(nb_model.classes)
    l = len(test_partition.data)
    confusion_matrix = np.zeros((ln,ln)) #makes a confusion matrix of zeroes of the right size first
    for i in range(l):
        y_r = y_real[i]
        pred_y = y_h[i]
        confusion_matrix[y_r][pred_y] += 1  #adds one to diagonal elements of the numpy array

    n = 0 #keeps track of number of accurate data points
    for i in range(ln):
        n += confusion_matrix[i][i] #sums the diagonal


    accuracy = n / (l) #computes accuracy

    #printing here
    print("Accuracy", round(accuracy, 7), "(", int(n), " out of ", l , " correct)")
    print("Confusion Matrix:")
    print(confusion_matrix)

main()
