"""
Utils for ensemble methods.
Authors: Sara Mathieson + Tanjuma Haque
Date: 10/27/19
"""

from collections import OrderedDict
import optparse
import random
import math
# my files
from Partition import *


def parse_args():
    """Parse command line arguments (train and test arff files)."""
    parser = optparse.OptionParser(description='run ensemble methods')

    parser.add_option('-r', '--train_filename', type='string', help='path to' +\
        ' train arff file', default = "data/mushroom_train.arff")
    parser.add_option('-e', '--test_filename', type='string', help='path to' +\
        ' test arff file', default = "data/mushroom_test.arff")
    parser.add_option('-T', '--T', type='int', help='the number of classifiers to use in our ensemble', default=10)
    parser.add_option('-p', '--threshold', type='float', help='the probability threshold required to classify a test example as possible', default=0.5)
    (opts, args) = parser.parse_args()

    mandatories = ['train_filename', 'test_filename', 'T', 'threshold',]
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()
    return opts

"""Function that is essentially the same as parse_args, but only T is mandatory"""
def read_arff(filename):
    """Read arff file into Partition format."""
    arff_file = open(filename,'r')
    data = [] # list of Examples
    F = OrderedDict() # dictionary

    header = arff_file.readline()
    line = arff_file.readline().strip()

    # read the attributes
    while line != "@data":
        line = line.replace('{','').replace('}','').replace(',','')
        tokens = line.split()
        name = tokens[1][1:-1]
        features = tokens[2:]

        # label
        if name != "class":
            F[name] = features
        else:
            first = tokens[2]
        line = arff_file.readline().strip()

    # read the examples
    for line in arff_file:
        tokens = line.strip().split(",")
        X_dict = {}
        i = 0
        for key in F:
            val = tokens[i]
            X_dict[key] = val
            i += 1
        label = -1 if tokens[-1] == first else 1
        # set weight to None for now
        data.append(Example(X_dict,label,None))

    arff_file.close()

    # set weights on each example
    n = len(data)
    for i in range(n):
        data[i].set_weight(1/n)

    partition = Partition(data, F)
    return partition

def bootstrap_partition(partition):
    """
    For creating bootstrap training data set and random subset of features
    """
    F = partition.F
    data = partition.data
    #randomly chooses k many elements from the list with replacement
    random_data = random.choices(data, k = len(data))
    #gets n, which is the size of our subset of features
    n = round(math.sqrt(len(F)))
    #randomly creates a list with n many featues and f vals
    random_Flst = random.sample(list(F.items()),k = n)
    #initialize an empty dictionary to hold our new F
    random_F = {}
    #gets each tuple from random_Flst, which is (feature name, [feature vals])
    for (a,b) in random_Flst:
        #adds feature vals as values to feature name as key
        random_F[a] = b
    #generates our new partition
    partition = Partition(random_data, random_F)
    return partition

def construct_cm(final_pred, test_partition):
    """
    Constructs confusion matrix
    """
    #makes a confusion matrix of zeroes of (2,2) since binary labels
    matrix = np.zeros((2,2))
    testexamples = test_partition.data
    len_t = len(testexamples)
    #goes through example list to get each example
    for i in range(len_t):
        #to "convert" -1 to 0 for later use
        if testexamples[i].label == -1:
            y_r = 0
        else:
            y_r = 1
        if final_pred[i] == -1:
            pred_y = 0
        else:
            pred_y = 1
        #depending on the combination of predicted and real label, increments count
        matrix[y_r][pred_y] += 1
    #returns confusion matrix
    return matrix

def rates(matrix):
    """
    Function to compute true and false positive rates from confusion matrix
    """
    #gets count for true positive
    tp = matrix[1][1]
    #gets count for false negative
    fn = matrix[1][0]
    #gets count for false positive
    fp = matrix[0][1]
    #gets count for true negative
    tn = matrix[0][0]
    #computes true positive rate
    true_pos = tp/(tp+fn)
    #computes false positive rate
    false_pos = fp/(fp+tn)
    #returns tuple of true and false positive rates
    return (true_pos, false_pos)
