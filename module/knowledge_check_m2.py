
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
If the time between customer arrivals to a restaurant at lunchtime fits the exponential distribution, then which of the following is true?
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

check_2_q16= """
Which formula represents the objective of Ridge Regression?
A. Minimize: ∑(yi−∑βjxij)〖^2〗 subject to ∑∣βj∣≤t
B. Minimize: ∑(yi−∑βjxij)〖^2〗+λ1∑∣βj∣+λ2∑βj〖^2〗
C. Minimize: ∑(yi−∑βjxij)〖^2〗2+λ∑βj〖^2〗
D. Minimize: ∑(yi−∑βjxij)〖^2〗2+λ∑βj
"""

check_2_q17= """
Which variable selection method combines both Lasso and Ridge approaches?
A. Forward Selection
B. Backward Elimination
C. Stepwise Regression
D. Elastic Net

"""

check_2_q18= """
In Ridge Regression, what does the tuning parameter λ control?
A. The size of coefficients.
B. The sum of the absolute values of the coefficients.
C. The number of factors in the model.
D. The p-value thresholds.
"""

check_2_q19= """
What does "blocking" involve in the context of Design of Experiments (DOE)?
A. Selecting a random sample of data
B. Identifying and accounting for factors that could introduce variability in the results
C. Collecting data without any controls
D. Analyzing data without statistical techniques
"""

check_2_q20= """
In which field is the Weibull distribution frequently used, and what does it model?
A. The Weibull distribution is used in finance to model stock prices.
B. The Weibull distribution is used in manufacturing to model production rates.
C. The Weibull distribution is used in reliability engineering and models various types of data, particularly failure rates and survival times.
D. The Weibull distribution is used in sports analytics to model player performance.
"""

check_2_q21= """
When is the Poisson distribution commonly used, and what does the parameter
λ represent?
A. The Poisson distribution is used for continuous data, and
λ represents the average value.

B. The Poisson distribution is used for counting events in a fixed interval, and
λ represents the average number of events.

C. The Poisson distribution is used for modeling geometric shapes, and
λ represents the shape parameter.

D. The Poisson distribution is used in finance, and
λ represents the interest rate.
"""

check_2_q22= """
How does the Binomial distribution differ from the Bernoulli distribution in terms of modeling?
A. The Binomial distribution predicts the number of successes in a fixed number of trials, while the Bernoulli distribution models individual events with two outcomes.
B. The Binomial distribution is used for continuous data, while the Bernoulli distribution is used for discrete data.
C. The Binomial distribution can have more than two possible outcomes, while the Bernoulli distribution is limited to two outcomes.
D. The Binomial distribution assumes no probability of success, while the Bernoulli distribution assumes a fixed probability of success.
"""

check_2_q23= """
In optimization models, what are the three key components?
A. Data, constraints, and objectives
B. Variables, constraints, and an objective function
C. Variables, equations, and parameters
D. Decisions, goals, and constraints
"""


check_2_q24= """
What is a dynamic program in optimization modeling?
A. A program that adapts to changing constraints in real-time.
B. A program that uses complex mathematical equations to optimize solutions.
C. A program that divides the system into states, decisions, and uses Bellman's equation to determine optimal decisions.
D. A program that focuses on optimizing linear functions.
"""

check_2_q25= """
McNemar's Test is primarily designed for:
A. Comparing the means of two independent samples.
B. Analyzing correlations between two continuous variables.
C. Analyzing categorical data in a 2x2 contingency table.
D. Predicting future outcomes based on historical data.
"""

check_2_q26= """
When should the Wilcoxon Signed Rank Test be used?
A. When analyzing normally distributed data.
B. When comparing two independent samples.
C. When working with non-normal data in paired samples.
D. When conducting before-and-after studies.
"""

check_2_q27= """
The Mann-Whitney Test is particularly useful for comparing:
A. Means from two independent samples.
B. Medians from two independent samples.
C. Variances from two paired samples.
D. Proportions from two dependent samples.
"""

check_2_q28= """
The Louvain Algorithm is primarily used for:
A. Solving linear equations.
B. Detecting communities in large networks.
C. Analyzing textual data.
D. Predicting stock prices.
"""

check_2_q29= """
In the context of game theory, what does Nash Equilibrium refer to?
A. The point where all players lose.
B. The point where no player can improve their outcome by changing their strategy.
C. The point where one player dominates all others.
D. The point where all players cooperate fully.
"""

check_2_q30= """
In Bayesian modeling, what is the purpose of the posterior distribution?
A. To represent the prior beliefs and knowledge of the modeler.
B. It describes the likelihood of observed data given the model parameters.
C. To summarize the historical data used in the model.
D. It quantifies the uncertainty about model parameters after considering both prior beliefs and observed data.
"""
KNOWLEDGE_2_QUESTIONS = [check_2_q1, check_2_q2, check_2_q3, check_2_q4, check_2_q5, check_2_q6,
                        check_2_q7, check_2_q8, check_2_q9, check_2_q10, check_2_q11, check_2_q12,
                        check_2_q13, check_2_q14, check_2_q15,check_2_q16, check_2_q17, check_2_q18,
                        check_2_q19, check_2_q20, check_2_q21, check_2_q22, check_2_q23, check_2_q24,
                        check_2_q25, check_2_q26, check_2_q27, check_2_q28, check_2_q29, check_2_q30,]
