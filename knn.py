# """
# Name: Umme Tanjuma Haque
# Purpose: Implementing K-nearest neighbours

# """

from scipy import stats
import numpy as np
import math

#loading the text files with the training and test data
train_data = np.loadtxt("zip.train")
test_data = np.loadtxt("zip.test")

#Filtering the training and test data for 2 and 3 labels
train_data = np.concatenate([[x] for x in train_data if x[0] == 2 or x[0] == 3])
test_data =  np.concatenate([[x] for x in test_data if x[0] == 2 or x[0] == 3])

##Separating the labels from the data
train_labels = train_data[:,0]
traindata_unlabeled = train_data[:,1:]

test_labels = test_data[:,0]
testdata_unlabeled = test_data[:,1:]
##

## Main function is where the Quantify the Accuracy function is being called
## and the output print statements are made
def main():
    print("Nearest Neighbours:")
    for i in range(1,11):
        y = accuracy_func(i,testdata_unlabeled)
        print("K=",i,",", y)
##

## Classify function takes a test data (one row from the entire test data, without the label)
## and the k value
def classify_func(test,k):
    dist_list = np.zeros(train_labels.size)           # initializes a list for carrying the distances from the test case to each training data row
    i = 0
    for x in traindata_unlabeled:                     # loops through each row of the unlabeled training data and calls the distance measuring function on it and the test case
        dist_list[i] = dist_func(x, test)             # adds each distance to the distance list with each loop
        i = i+1
    sorted_indices = (np.argsort(dist_list))[:k]      # sorts the distance list, produces the retained indices for the sorted distance list and takes the first k-1 indices
    k_nearest = np.take(train_labels, sorted_indices) # takes and returns the train data labels associated with the sorted indices
    return k_nearest

## Distance function measures the Euclidean distance between the test data case and training data
def dist_func(x, example):
    sub = np.subtract(x, example)
    mul = np.multiply(sub,sub)
    added = np.sum(mul)
    return math.sqrt(added)


## The Accuracy function goes through the k-nearest neighbours and finds the mode (predicted label) for all the test cases
def accuracy_func(kv, examples):
    mode_list = np.zeros(test_labels.size)
    i = 0
    for x in examples:
        k_neighbours = classify_func(x,kv)
        unique_neighbours = np.unique(k_neighbours)
        if unique_neighbours.size != k_neighbours.size: # the if statement checks that if there are repeats then there must be a mode. If not, then the predicted label is the average.
            a = stats.mode(k_neighbours)
            mode_data = a[0]
        else:
            mode_data = np.average(k_neighbours)
        mode_list[i] = mode_data
        i = i+1
    truth_num = np.sum(mode_list == test_labels)
    percentage = (np.divide(truth_num,test_labels.size))*100
    return (np.around(percentage, decimals = 3))
##

##calling main
main()
