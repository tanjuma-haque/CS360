"""
Decision stump data structure (i.e. tree with depth=1), non-recursive.
Authors: Sara Mathieson + Tanjuma Haque
Date:10/28/19
"""
from Partition import *
import util

class DecisionStump:

    def __init__(self, partition):
        """
        Create a DecisionStump from the given partition of data by choosing one
        best feature and splitting the data into leaves based on this feature.
        """
        # key: edge, value: probability of positive (i.e. 1)
        self.children = {}

        # use weighted conditional entropy to select the best feature
        feature = partition.best_feature()
        self.name = feature

        # divide data into separate partitions based on this feature
        values = partition.F[feature]
        groups = {}
        for v in values:
            groups[v] = []
        for i in range(partition.n):
            v = partition.data[i].features[feature]
            groups[v].append(partition.data[i])

        # add a child for each possible value of the feature
        for v in values:
            new_partition = Partition(groups[v], partition.F)
            # weighted probability of a positive result
            prob_pos = new_partition.prob_pos()
            self.add_child(v,prob_pos)

    def get_name(self):
        """Getter for the name of the best feature (root)."""
        return self.name

    def add_child(self, edge, prob):
        """
        Add a child with edge as the feature value, and prob as the probability
        of a positive result.
        """
        self.children[edge] = prob

    def get_child(self, edge):
        """Return the probability of a positive result, given feature value."""
        return self.children[edge]

    def __str__(self):
        """Returns a string representation of the decision stump."""
        s = self.name + " =\n"
        for v in self.children:
            s += "  " + v + ", " + str(self.children[v]) + "\n"
        return s

    def classify(self, test_features, thresh):
        """
        Classify the test example (using features only) as +1 (positive) or -1
        (negative), using the provided threshold.
        """
         #get the best feature's value from the test example
        featureval = test_features.features[self.name]
        #get the probability of the feature value
        probability = self.children[featureval]

        #if probability for positive label is >= threshold then return +1 otherwise return -1
        if probability >= thresh:
            return 1
        else:
            return -1
# def main():
#
#     train_partition = util.read_arff("data/tennis_train.arff")
#     test_partition  = util.read_arff("data/tennis_test.arff")
#
#     for i in range(train_partition.n):
#         example = train_partition.data[i]
#         if i == 0 or i == 8:
#             example.set_weight(0.25)
#         else:
#             example.set_weight(0.5/(train_partition.n-2))
#
#     s = DecisionStump(train_partition)
#     print("D-STUMP")
#     print (s.__str__())
#
#     error = 0
#     print("TESTING")
#     print("i      true      pred")
#     i = 0
#     for x in test_partition.data:
#         y_pred = s.classify(x, 0.5)
#         if (y_pred != x.label):
#             error += 1
#         print('{:<8}{: d}{:<8}{: d}'.format(i, x.label, '', y_pred))
#         i+=1
#
#     print ("error", (error/len(test_partition.data)))
#
# if __name__ == "__main__":
#     main()
