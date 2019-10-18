"""
Lab 4 ARFF file reader
Author: Sara Mathieson + Tanjuma Haque
Date: 9/29/19
"""

# python imports
from collections import OrderedDict
import math
import numpy as np
import optparse
import sys

# my file imports
from Partition import *

def parse_args():
    """Parse command line arguments (train and test arff files)."""
    parser = optparse.OptionParser(description='run naive bayes method')

    parser.add_option('-r', '--train_filename', type='string', help='path to' +\
        ' train arff file')
    parser.add_option('-e', '--test_filename', type='string', help='path to' +\
        ' test arff file')
    (opts, args) = parser.parse_args()

    mandatories = ['train_filename', 'test_filename',]
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts

def read_arff(filename):
    """
    Read arff file into Partition format. Params:
    * filename (str), the path to the arff file
        and False if test dataset (don't convert continuous features)
    """
    arff_file = open(filename,'r')
    data = [] # list of Examples
    F = OrderedDict() # key: feature name, value: list of feature values

    header = arff_file.readline()
    line = arff_file.readline().strip()
    classes = []

    # read the attributes
    while line != "@data":

        clean = line.replace('{','').replace('}','').replace(',','')
        tokens = clean.split()
        name = tokens[1][1:-1]

        # discrete feature
        feature_values = tokens[2:]

        # record features or label
        if name != "class":
            F[name] = feature_values
        else:
            # first will be label -1, second will be +1
            first = tokens[2]
            classes = tokens[2:]
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
        # add the labels
        label = tokens[-1]

        # add to list of Examples
        data.append(Example(X_dict,label))

    arff_file.close()
    partition = Partition(data, F, len(classes))

    return partition
