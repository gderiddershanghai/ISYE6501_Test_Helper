
check_1_q1 = """
\nA survey of 25 people recorded each person’s family size and type of car.
\n1. Which of these is a data point?
\n-The 14th person’s family size and car type
\n-The 14th person’s family size
\n-The car type of each person
"""

check_1_q2 = """
\na. Which of these is structured data?
\n-The contents of a person’s Twitter feed
\n-The amount of money in a person’s bank account

\nb. Which of these is time series data?
\n-The average cost of a house in the United States every year since 1820
\n-The height of each professional basketball player in the NBA at the start of the season
"""
check_1_q3 = """
\nWhen comparing models, if we use the same data to pick the best model as we do to estimate how good the best one is, what is likely to happen?
\n-The model will appear to be better than it really is.
\n-The model will appear to be worse than it really is.
\n-The model will appear to be just as good as it really is.
"""

check_1_q4 = """
\nIn k-fold cross-validation, how many times is each part of the data used for training, and for validation?
\n-k times for training, and k times for validation
\n-1 time for training, and k-1 times for validation
\n-k-1 times for training, and 1 time for validation

"""

check_1__q5 = """
\nThe k-means algorithm for clustering is a “heuristic” because…
\n…it runs quickly.
\n…it never gets the best answer.
\n…it isn’t guaranteed to get the best answer.
"""

check_1__q6 = """
\nWhich of these is generally a good reason to remove an outlier from your data set?
\nThe outlier is an incorrectly-entered data, not real data.
\nOutliers like this only happen occasionally.
"""

check_1_q7 = """
\nIn the CUSUM model, having a higher threshold T makes it…
\n…detect changes faster, and less likely to falsely detect changes.
\n…detect changes faster, and more likely to falsely detect changes.
\n…detect changes slower, and less likely to falsely detect changes.
\n…detect changes slower, and more likely to falsely detect changes.

"""

check_1_q8 = """
\nA multiplicative seasonality, like in the Holt-Winters method, means that the seasonal effect is…
\n- The same regardless of the baseline value.
\n- Proportional to the baseline value.

"""
check_1_q9 = """
\nIs exponential smoothing better for short-term forecasting or long-term forecasting?
\nShort-term
\nLong-term

"""
check_1_q10 = """
\nWhy is GARCH different from ARIMA and exponential smoothing?
\n- GARCH uses time series data
\n- GARCH is autoregressive
\n- GARCH estimates variance
"""
check_1_q11 = """
\nWhen would regression be used instead of a time series model?
\n- When there are other factors or predictors that affect the response
\n- When only previous values of the response affect its current value

"""
check_1_q12 = """
\nIf two models are approximately equally good, measures like AIC and BIC will favor the simpler model. Simpler models are often better because…
\n-Simple models are easier to explain and “sell” to managers and executives
\n-The effects observed in simple models are easier for everyone, including analytics professionals, to understand
\n-Simple models are less likely to be over-fit to random effects
\n-All of the above


"""
check_1_q13 = """
\nWhat does “heteroscedasticity” mean?
\n-The variance is different in different ranges of the data correct
\n-The variances of two samples of data are different from each other

"""
check_1_q14= """
\nA model is built to determine whether data points belong to a category or not. A “true negative” result is:
\n- A data point that is in the category, but the model incorrectly says it isn’t.
\n- A data point that is not in the category, but the model incorrectly says it is.
\n- A data point that is in the category, and the model correctly says it is.
\n- A data point that is not in the category, and the model correctly says so.
\n- A “Debbie Downer” (someone who often says negative things that bring down everyone’s mood).

"""
check_1_q15= """
\nA common rule of thumb is to stop branching if a leaf would contain less than 5 percent of the data points. Why not keep branching and allow models to find very close fits to each very small subset of data?
\n-Actually, that sounds like a great idea – we should keep branching and let models find very close fits to very small subsets of data!
\n-Fitting to very small subsets of data will cause overfitting.
\n-Fitting to very small subsets of data will make the tree have too many leaves.
"""

check_1_q16 = """
\nIn K-fold cross-validation, what does 'k' represent?

\nA. The number of data points in the test set.
\nB. The number of features in the dataset.
\nC. The number of parts the data is split into.
\nD. The number of attributes in the training set.
"""

check_1_q17 = """
\nWhat is the primary purpose of using a separate test data set in model validation?

\nA. To optimize model parameters.
\nB. To assess model effectiveness on random patterns.
\nC. To choose the best-performing model.
\nD. To estimate true model performance.
"""

check_1_q18 = """
\nWhich of the following best describes the concept of classification in analytics?

\nA. Identifying the most common category among neighbors.
\nB. Separating different categories with maximum margin.
\nC. Assessing the cost of different types of errors.
\nD. Measuring distances between data points.
"""

check_1_q19 = """
\nWhat is the main iterative process in the K-means clustering algorithm?
\nA. Calculating distances between data points.
\nB. Assigning data points to the nearest cluster center.
\nC. Calculating p-norms for distance measurements.
\nD. Creating Voronoi diagrams.
"""

check_1_q20 = """
\nQuestion 4: Practical K-means
\nWhat is the purpose of the "elbow diagram" in K-means clustering?
\nA. To show the shape of the clusters.
\nB. To visualize the data distribution.
\nC. To decide the number of clusters.
\nD. To identify outliers in the data.
"""

check_1_q21 = """
\nWhat are the different types of outliers?
\nA. Single outliers and double outliers.
\nB. Point outliers, contextual outliers, and collective outliers.
\nC. Inliers and outliers.
\nD. Clean data and noisy data.
"""

check_1_q22 = """
\nWhat is the primary purpose of change detection in data analysis?
\nA. Identifying historical data patterns
\nB. Detecting changes in data over time
\nC. Smoothing out data fluctuations
\nD. Estimating baseline values
"""

check_1_q23 = """
\nwhat do different parameter values in CUSUM affect?
\nA. The nature of the data
\nB. The sensitivity of change detection
\nC. The data collection process
\nD. The data visualization
"""

check_1_q24 = """
\nWhat does exponential smoothing primarily help in determining when analyzing time series data?
\nA. The latest observation
\nB. Baseline data value
\nC. Seasonality factors
\nD. Random fluctuations
"""

check_1_q25 = """
\nIn exponential smoothing, what does the coefficient alpha balance in the model?
\nA. Baseline and trend estimates
\nB. Latest observation and previous estimate
\nC. Seasonality and cyclical effects
\nD. Forecasting errors
"""

check_1_q26 = """
\nWhat is the primary purpose of the Box-Cox transformation in data preparation?
\nA. Reduce dimensionality of data
\nB. Normalize data distributions
\nC. Remove outliers from data
\nD. Adjust data for seasonal effects
"""

check_1_q27 = """
\nWhat is the main objective of Principal Component Analysis (PCA)?
\nA. Increase the number of predictors
\nB. Remove all variance in data
\nC. Simplify complex datasets
\nD. Introduce correlation among variables
"""

check_1_q28 = """
\nIn Principal Component Analysis (PCA), what do eigenvalues and eigenvectors determine?
\nA. The number of principal components
\nB. The orthogonality of data
\nC. The transformation of data
\nD. The correlation among variables
"""

check_1_q29 = """
\nWhat is the primary advantage of Classification and Regression Trees (CART) in regression analysis?
\nA. They use a single regression model for all data points.
\nB. They allow the use of separate coefficients for different data segments.
\nC. They are less prone to overfitting compared to other methods.
\nD. They do not require pruning.
"""

check_1_q30 = """
\nIn logistic regression, what is the range of predicted probabilities?
\nA. -1 to 1
\nB. 0 to 1
\nC. 0 to infinity
\nD. -infinity to infinity
"""

KNOWLEDGE_1_QUESTIONS = [check_1_q1, check_1_q2, check_1_q3, check_1_q4, check_1__q5, check_1__q6,
                        check_1_q7, check_1_q8, check_1_q9, check_1_q10, check_1_q11, check_1_q12,
                        check_1_q13, check_1_q14, check_1_q15,check_1_q16, check_1_q17, check_1_q18,
                        check_1_q19, check_1_q20, check_1_q21, check_1_q22, check_1_q23, check_1_q24,
                        check_1_q25, check_1_q26, check_1_q27, check_1_q28, check_1_q29, check_1_q30,]
