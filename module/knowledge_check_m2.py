
check_2_q1 = """
\nBuilding simpler models with fewer factors helps avoid which problems?
\n-Overfitting
\n-Low prediction quality
\n-Bias in the most important factors.
\n-Difficulty of interpretation

"""
check_2_q2 = """
\nWhen two predictors are highly correlated, which of the following statements is true?
\n-Lasso regression will usually have positive coefficients for both predictors.
\n-Ridge regression will usually have positive coefficients for both predictors.

"""
check_2_q3 = """
\nIf we’re testing to see whether red cars sell for higher prices than blue cars, we need to account for the type and age of the cars in our data set. This is called
\n- Controlling
\n- Comparing
\n- Combining
"""

check_2_q4 = """
\nIn which of these situations is a factorial design more appropriate than A/B testing?
\n- Choosing between two different banner ad designs for orange juice.
\n- Picking the best-tasting of four different brands of orange juice.
\n- Finding the best combination of factors in orange juice to maximize sales.

"""

check_2_q5 = """
\nWhich of these is a way that multi-armed bandit models deal with balancing exploration and exploitation?
\n- As we get more sure of the best answer, we’re more likely to choose to use it.
\n- If we’re unsure of the best answer, we should pick one and stick with it.
\n- As we get more sure of the best answer, we’re more likely to try many different ones just to make sure.

"""

check_2_q6 = """
\nIf the time between customer arrivals to a restaurant at lunchtime fits the exponential distribution, then which of the following is true?
\n- The number of arrivals per unit time follows the Weibull distribution.
\n- The number of arrivals per unit time follows the Poisson distribution.

"""

check_2_q7 = """
\nWhich of these is a queuing model not appropriate for?
\n- Determining the average wait time on a customer service hotline.
\n- Estimating the length of the checkout lines at a grocery store.
\n- Predicting the number of customers who will come to a restaurant tomorrow.

"""

check_2_q8 = """
\nIn analytics modeling, we need to…
\n…try to avoid our biases affecting our models.
\n…report our results honestly.
\n…be open to other people’s models, even if they’re different from ours.
\n…develop our own intuition and artistry.
"""

check_2_q9 = """
\nWhich of these is a common reason that data sets are missing values?
\n-A person accidentally typed in the wrong value.
\n-A person did not want to reveal the true value.
\n-An automated system did not work correctly to record the value.
\n-All of the above.
"""

check_2_q10 = """
\nWhich statements are true about data imputation?
\nImputing more than 5 percent of values is usually not recommended.
\nImputation of more than one factor value for a data point is also possible.
\nBoth answers above are correct.
"""

check_2_q11 = """
\nWhich of these statistical models does not have an underlying optimization model to find the best fit?
\nLinear regression
\nLogistic regression
\nLasso regression
\nExponential smoothing
\nk-means clustering
\nNone of the above

"""

check_2_q12 = """
\nThe two main steps of most optimization algorithms are:
\n- Find the most important remaining variable, and assign its value to be as large as possible.
\n- Find a good direction to move from the current solution, and determine how far to go in that direction.
\n- Pick a set of variables to be zero, and find the best values of the remaining variables.

"""
check_2_q13 = """
\nNonparametric tests are useful when…
\n- we don’t know much about the form of the underlying distribution the data comes from, or it doesn’t fit a nice distribution.
\n- it’s important to have information about the median.
\n- we don’t have much data.
\n- we want to know whether the means of two distributions are similar.

"""
check_2_q14= """
\nSuppose we have a graph where all edge weights are equal to 1. In the video, we saw how to split a graph up into highly-interconnected communities. Now, instead we want to split the nodes into large groups that have very few connections between them (for example, if a marketer wants to find sets of people in a social network who probably have very different sets of friends). How might you do that?
\n- Change the Louvain algorithm to minimize modularity instead of maximizing it.
\n- Change the graph: for every pair of nodes i and j, if there’s an edge between i and j then remove it; and if there’s not an edge between i and j, then add it. Then run the Louvain algorithm on the new graph.

"""
check_2_q15= """
\nDeep learning is one of the current best approaches for which of these?
\nImage recognition
\nSplitting graphs into highly-connected communities
\nDemand forecasting
"""

check_2_q16= """
\nWhich formula represents the objective of Ridge Regression?
\nA. Minimize: ∑(yi−∑βjxij)〖^2〗 subject to ∑∣βj∣≤t
\nB. Minimize: ∑(yi−∑βjxij)〖^2〗+λ1∑∣βj∣+λ2∑βj〖^2〗
\nC. Minimize: ∑(yi−∑βjxij)〖^2〗2+λ∑βj〖^2〗
\nD. Minimize: ∑(yi−∑βjxij)〖^2〗2+λ∑βj
"""

check_2_q17= """
\nWhich variable selection method combines both Lasso and Ridge approaches?
\nA. Forward Selection
\nB. Backward Elimination
\nC. Stepwise Regression
\nD. Elastic Net

"""

check_2_q18= """
\nIn Ridge Regression, what does the tuning parameter λ control?
\nA. The size of coefficients.
\nB. The sum of the absolute values of the coefficients.
\nC. The number of factors in the model.
\nD. The p-value thresholds.
"""

check_2_q19= """
\nWhat does "blocking" involve in the context of Design of Experiments (DOE)?
\nA. Selecting a random sample of data
\nB. Identifying and accounting for factors that could introduce variability in the results
\nC. Collecting data without any controls
\nD. Analyzing data without statistical techniques
"""

check_2_q20= """
\nIn which field is the Weibull distribution frequently used, and what does it model?
\nA. The Weibull distribution is used in finance to model stock prices.
\nB. The Weibull distribution is used in manufacturing to model production rates.
\nC. The Weibull distribution is used in reliability engineering and models various types of data, particularly failure rates and survival times.
\nD. The Weibull distribution is used in sports analytics to model player performance.
"""

check_2_q21= """
When is the Poisson distribution commonly used, and what does the parameter
λ represent?\n
A. The Poisson distribution is used for continuous data, and
λ represents the average value.
\n
B. The Poisson distribution is used for counting events in a fixed interval, and
λ represents the average number of events.
\n
C. The Poisson distribution is used for modeling geometric shapes, and
λ represents the shape parameter.
\n
D. The Poisson distribution is used in finance, and
λ represents the interest rate.
"""

check_2_q22= """
\nHow does the Binomial distribution differ from the Bernoulli distribution in terms of modeling?
\nA. The Binomial distribution predicts the number of successes in a fixed number of trials, while the Bernoulli distribution models individual events with two outcomes.
\nB. The Binomial distribution is used for continuous data, while the Bernoulli distribution is used for discrete data.
\nC. The Binomial distribution can have more than two possible outcomes, while the Bernoulli distribution is limited to two outcomes.
\nD. The Binomial distribution assumes no probability of success, while the Bernoulli distribution assumes a fixed probability of success.
"""

check_2_q23= """
\nIn optimization models, what are the three key components?
\nA. Data, constraints, and objectives
\nB. Variables, constraints, and an objective function
\nC. Variables, equations, and parameters
\nD. Decisions, goals, and constraints
"""


check_2_q24= """
\nWhat is a dynamic program in optimization modeling?
\nA. A program that adapts to changing constraints in real-time.
\nB. A program that uses complex mathematical equations to optimize solutions.
\nC. A program that divides the system into states, decisions, and uses Bellman's equation to determine optimal decisions.
\nD. A program that focuses on optimizing linear functions.
"""

check_2_q25= """
\nMcNemar's Test is primarily designed for:
\nA. Comparing the means of two independent samples.
\nB. Analyzing correlations between two continuous variables.
\nC. Analyzing categorical data in a 2x2 contingency table.
\nD. Predicting future outcomes based on historical data.
"""

check_2_q26= """
\nWhen should the Wilcoxon Signed Rank Test be used?
\nA. When analyzing normally distributed data.
\nB. When comparing two independent samples.
\nC. When working with non-normal data in paired samples.
\nD. When conducting before-and-after studies.
"""

check_2_q27= """
\nThe Mann-Whitney Test is particularly useful for comparing:
\nA. Means from two independent samples.
\nB. Medians from two independent samples.
\nC. Variances from two paired samples.
\nD. Proportions from two dependent samples.
"""

check_2_q28= """
\nThe Louvain Algorithm is primarily used for:
\nA. Solving linear equations.
\nB. Detecting communities in large networks.
\nC. Analyzing textual data.
\nD. Predicting stock prices.
"""

check_2_q29= """
\nIn the context of game theory, what does Nash Equilibrium refer to?
\nA. The point where all players lose.
\nB. The point where no player can improve their outcome by changing their strategy.
\nC. The point where one player dominates all others.
\nD. The point where all players cooperate fully.
"""

check_2_q30= """
\nIn Bayesian modeling, what is the purpose of the posterior distribution?
\nA. To represent the prior beliefs and knowledge of the modeler.
\nB. It describes the likelihood of observed data given the model parameters.
\nC. To summarize the historical data used in the model.
\nD. It quantifies the uncertainty about model parameters after considering both prior beliefs and observed data.
"""
KNOWLEDGE_2_QUESTIONS = [check_2_q1, check_2_q2, check_2_q3, check_2_q4, check_2_q5, check_2_q6,
                        check_2_q7, check_2_q8, check_2_q9, check_2_q10, check_2_q11, check_2_q12,
                        check_2_q13, check_2_q14, check_2_q15,check_2_q16, check_2_q17, check_2_q18,
                        check_2_q19, check_2_q20, check_2_q21, check_2_q22, check_2_q23, check_2_q24,
                        check_2_q25, check_2_q26, check_2_q27, check_2_q28, check_2_q29, check_2_q30,]
