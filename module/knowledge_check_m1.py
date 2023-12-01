
check_1_q1 = """
A survey of 25 people recorded each person’s family size and type of car.
1. Which of these is a data point?
-The 14th person’s family size and car type
-The 14th person’s family size
-The car type of each person
"""

check_1_q2 = """
2. Which of these is structured data?
-The contents of a person’s Twitter feed
-The amount of money in a person’s bank account

3. Which of these is time series data?
-The average cost of a house in the United States every year since 1820
-The height of each professional basketball player in the NBA at the start of the season
"""
check_1_q3 = """
When comparing models, if we use the same data to pick the best model as we do to estimate how good the best one is, what is likely to happen?
-The model will appear to be better than it really is.
-The model will appear to be worse than it really is.
-The model will appear to be just as good as it really is.
"""

check_1_q4 = """
In k-fold cross-validation, how many times is each part of the data used for training, and for validation?
-k times for training, and k times for validation
-1 time for training, and k-1 times for validation
-k-1 times for training, and 1 time for validation

"""

check_1__q5 = """
The k-means algorithm for clustering is a “heuristic” because…
…it runs quickly.
…it never gets the best answer.
…it isn’t guaranteed to get the best answer.
"""

check_1__q6 = """
Which of these is generally a good reason to remove an outlier from your data set?
The outlier is an incorrectly-entered data, not real data.
Outliers like this only happen occasionally.
"""

check_1_q7 = """
In the CUSUM model, having a higher threshold T makes it…
…detect changes faster, and less likely to falsely detect changes.
…detect changes faster, and more likely to falsely detect changes.
…detect changes slower, and less likely to falsely detect changes.
…detect changes slower, and more likely to falsely detect changes.

"""

check_1_q8 = """
A multiplicative seasonality, like in the Holt-Winters method, means that the seasonal effect is…
- The same regardless of the baseline value.
- Proportional to the baseline value.

"""
check_1_q9 = """
Is exponential smoothing better for short-term forecasting or long-term forecasting?
Short-term
Long-term

"""
check_1_q10 = """
Why is GARCH different from ARIMA and exponential smoothing?
- GARCH uses time series data
- GARCH is autoregressive
- GARCH estimates variance
"""
check_1_q11 = """
When would regression be used instead of a time series model?
- When there are other factors or predictors that affect the response
- When only previous values of the response affect its current value

"""
check_1_q12 = """
If two models are approximately equally good, measures like AIC and BIC will favor the simpler model. Simpler models are often better because…
-Simple models are easier to explain and “sell” to managers and executives
-The effects observed in simple models are easier for everyone, including analytics professionals, to understand
-Simple models are less likely to be over-fit to random effects
-All of the above


"""
check_1_q13 = """
What does “heteroscedasticity” mean?
-The variance is different in different ranges of the data correct
-The variances of two samples of data are different from each other

"""
check_1_q14= """
A model is built to determine whether data points belong to a category or not. A “true negative” result is:
- A data point that is in the category, but the model incorrectly says it isn’t.
- A data point that is not in the category, but the model incorrectly says it is.
- A data point that is in the category, and the model correctly says it is.
- A data point that is not in the category, and the model correctly says so.
- A “Debbie Downer” (someone who often says negative things that bring down everyone’s mood).

"""
check_1_q15= """
A common rule of thumb is to stop branching if a leaf would contain less than 5 percent of the data points. Why not keep branching and allow models to find very close fits to each very small subset of data?
-Actually, that sounds like a great idea – we should keep branching and let models find very close fits to very small subsets of data!
-Fitting to very small subsets of data will cause overfitting.
-Fitting to very small subsets of data will make the tree have too many leaves.

"""


KNOWLEDGE_1_QUESTIONS = [check_1_q1, check_1_q2, check_1_q3, check_1_q4, check_1__q5, check_1__q6,
                        check_1_q7, check_1_q8, check_1_q9, check_1_q10, check_1_q11, check_1_q12,
                        check_1_q13, check_1_q14, check_1_q15]
