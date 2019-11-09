## CS 360 Lab 6 - Ensemble Methods

Name: Tanjuma Haque

userId:

Number of Late Days Using for this lab: 0

---

### Analysis

1. Based on your ROC curves, which method would you choose for this application?
Which threshold would you choose? Does your answer depend on how much training
time you have available?

For the given data set, I would use AdaBoost since the true positive rate approaches 1.0 faster, so we have 1.0 as tp with a lower fp. Also, the difference in run time isn't that huge.
I would choose a threshold of 0.45 because it roughly gives 1.0 as tp
with the lowest possible fp. If there is more training time, Random Forest could be better
because if tp also reached 1.0 with a low enough fp value with more training, then the run time would be the deciding factor between RF and AB, because I think RF has a shorter run time.

2. `T` can be thought of as a measure of model complexity. Do these methods seem
to suffer from overfitting as the model complexity increases? What is the
intuition behind your answer?

I think they are not prone to overfitting as model complexity increases. Random forests use
randomly selected data sets and features from the given training data, so the intuition is more
like averaging the decision stumps, which cannot lead to overfitting. It limits overfitting.
For AdaBoost, the use of more and more classifers (T) would only cause the model to fix the mistakes of the previous classifiers, which basically would smooth out the noise.

---

### Lab Questionnaire

(None of your answers below will affect your grade; this is to help refine lab assignments in the future)

1. Approximately, how many hours did you take to complete this lab? (provide your answer as a single integer on the line below)
10 hours
2. How difficult did you find this lab? (1-5, with 5 being very difficult and 1 being very easy)
I would say 3 because debugging was difficult because there is so much going on, but the implementation wasn't that difficult.
3. Describe the biggest challenge you faced on this lab:
 Debugging as I said before
