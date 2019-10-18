"""
Tanjuma H, 9/19/19
"""

import util
from DecisionTree import *
from Partition import *

def main():

    opts = util.parse_args()
    train_partition = util.read_arff(opts.train_filename, True)
    test_partition  = util.read_arff(opts.test_filename, False)


    # create an instance of the DecisionTree class from the train_partition
    tree = DecisionTree(train_partition, (vars(opts)).get("depth"))
    rootnode = tree.constructsubtree(train_partition, (vars(opts)).get("depth"), 0)

    #print text representation of the DecisionTree
    tree.printtree(rootnode)

    # TODO: evaluate the decision tree on the test_partition

main()
