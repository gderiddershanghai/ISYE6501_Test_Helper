
check_2_q1 = """
Building simpler models with fewer factors helps avoid which problems?
-Overfitting
-Low prediction quality
-Bias in the most important factors.
-Difficulty of interpretation

"""
check_2_q2 = """
When two predictors are highly correlated, which of the following statements is true?
-Lasso regression will usually have positive coefficients for both predictors.
-Ridge regression will usually have positive coefficients for both predictors.

"""
check_2_q3 = """
If we’re testing to see whether red cars sell for higher prices than blue cars, we need to account for the type and age of the cars in our data set. This is called
- Controlling
- Comparing
- Combining
"""

check_2_q4 = """
In which of these situations is a factorial design more appropriate than A/B testing?
- Choosing between two different banner ad designs for orange juice.
- Picking the best-tasting of four different brands of orange juice.
- Finding the best combination of factors in orange juice to maximize sales.

"""

check_2_q5 = """
Which of these is a way that multi-armed bandit models deal with balancing exploration and exploitation?
- As we get more sure of the best answer, we’re more likely to choose to use it.
- If we’re unsure of the best answer, we should pick one and stick with it.
- As we get more sure of the best answer, we’re more likely to try many different ones just to make sure.

"""

check_2_q6 = """
1. If the time between customer arrivals to a restaurant at lunchtime fits the exponential distribution, then which of the following is true?
- The number of arrivals per unit time follows the Weibull distribution.
- The number of arrivals per unit time follows the Poisson distribution.

"""

check_2_q7 = """
Which of these is a queuing model not appropriate for?
- Determining the average wait time on a customer service hotline.
- Estimating the length of the checkout lines at a grocery store.
- Predicting the number of customers who will come to a restaurant tomorrow.

"""

check_2_q8 = """
In analytics modeling, we need to…
…try to avoid our biases affecting our models.
…report our results honestly.
…be open to other people’s models, even if they’re different from ours.
…develop our own intuition and artistry.
"""

check_2_q9 = """
Which of these is a common reason that data sets are missing values?
-A person accidentally typed in the wrong value.
-A person did not want to reveal the true value.
-An automated system did not work correctly to record the value.
-All of the above.
"""

check_2_q10 = """
Which statements are true about data imputation?
Imputing more than 5 percent of values is usually not recommended.
Imputation of more than one factor value for a data point is also possible.
Both answers above are correct.
"""

check_2_q11 = """
Which of these statistical models does not have an underlying optimization model to find the best fit?
Linear regression
Logistic regression
Lasso regression
Exponential smoothing
k-means clustering
None of the above

"""

check_2_q12 = """
The two main steps of most optimization algorithms are:
- Find the most important remaining variable, and assign its value to be as large as possible.
- Find a good direction to move from the current solution, and determine how far to go in that direction.
- Pick a set of variables to be zero, and find the best values of the remaining variables.

"""
check_2_q13 = """
Nonparametric tests are useful when…
- we don’t know much about the form of the underlying distribution the data comes from, or it doesn’t fit a nice distribution.
- it’s important to have information about the median.
- we don’t have much data.
- we want to know whether the means of two distributions are similar.

"""
check_2_q14= """
Suppose we have a graph where all edge weights are equal to 1. In the video, we saw how to split a graph up into highly-interconnected communities. Now, instead we want to split the nodes into large groups that have very few connections between them (for example, if a marketer wants to find sets of people in a social network who probably have very different sets of friends). How might you do that?
- Change the Louvain algorithm to minimize modularity instead of maximizing it.
- Change the graph: for every pair of nodes i and j, if there’s an edge between i and j then remove it; and if there’s not an edge between i and j, then add it. Then run the Louvain algorithm on the new graph.

"""
check_2_q15= """
Deep learning is one of the current best approaches for which of these?
Image recognition
Splitting graphs into highly-connected communities
Demand forecasting
"""


KNOWLEDGE_2_QUESTIONS = [check_2_q1, check_2_q2, check_2_q3, check_2_q4, check_2_q5, check_2_q6,
                        check_2_q7, check_2_q8, check_2_q9, check_2_q10, check_2_q11, check_2_q12,
                        check_2_q13, check_2_q14, check_2_q15]
