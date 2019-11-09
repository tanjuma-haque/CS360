"""
Run ensemble methods to create ROC curves.
Authors: Tanjuma H
Date:11/1/19
"""
import random_forest
import ada_boost
import util
import matplotlib.pyplot as plt

def test_random(test_partition, ds_ensemble, threshold ):
    """
    Function to run the testing part for Random Forests
    """
    ds_list = random_forest.testing(test_partition, ds_ensemble, threshold)
    #gets the final predicted labels for test data by majority vote
    finalpred_lst = random_forest.finaloutput(ds_list,test_partition)
    #contructs confusion matrix
    confusion_matrix = util.construct_cm(finalpred_lst, test_partition)
    #computes the true positive and false positive rates from the confusion matrix
    (true_pos, false_pos) = util.rates(confusion_matrix)
    return (false_pos, true_pos)


def test_adaboost(test_partition, ds_ensemble, scorelist, threshold):
    """
    Function to run the testing part for AdaBoost
    """
    #gets the predicted ys from each classifer
    tested_list = ada_boost.testing(test_partition, ds_ensemble, threshold)
    #gets the final predicted ys for the test data
    finalpred_lst = ada_boost.finaloutput(tested_list,test_partition, scorelist, threshold)
    #contructs confusion matrix
    confusion_matrix = util.construct_cm(finalpred_lst, test_partition)
    #computes the true positive and false positive rates from the confusion matrix
    (true_pos, false_pos) = util.rates(confusion_matrix)
    return (false_pos, true_pos)

def plot_data(ab_list, rf_list, T):
    """
    Function to plot the tp and fp rates data for Random Forest and AdaBoost
    """
    #unzipping the lists to separate tp and fp for easier plotting
    ab_false, ab_true = zip(*ab_list)
    rf_false, rf_true = zip(*rf_list)
    plt.plot(ab_false, ab_true, label = "AdaBoost", marker = "o")
    plt.plot(rf_false, rf_true, label = "Random forest", marker = "x")

    # naming the x  and y axis
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.legend()
    # giving a title
    heading = "ROC Curve for Mushroom Dataset T =" + str(T)
    plt.title(heading)
    plt.show()

def main():
    opts = util.parse_args()
    train_partition = util.read_arff(opts.train_filename)
    test_partition  = util.read_arff(opts.test_filename)

    #training random forest first
    print(opts.T)
    rf_ensemble = random_forest.construct_ensemble(opts.T, train_partition)
    #training AdaBoost next
    (ad_ensemble, scorelist) = ada_boost.construct_ensemble(opts, train_partition)
    #initializing threshold that will be changed in the loop
    thresh = -0.1
    #initializes two lists of size 20 to hold true and false positive rates for
    #both the ensemble methods for each threshold value
    rm_forest = [None]*20
    adaboost = [None]*20
    #loops to increment threshold
    for i in range(20):
        rm_forest[i] = test_random(test_partition,rf_ensemble, thresh)
        adaboost[i] = test_adaboost(test_partition, ad_ensemble, scorelist, thresh)
        thresh += 0.06
        print(rm_forest[i],adaboost[i],thresh)

    #plots the roc curves
    plot_data(adaboost,rm_forest, opts.T)
if __name__ == '__main__':
    main()
