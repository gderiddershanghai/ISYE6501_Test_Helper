midterm_1_q1 =     """
    A1. \nFor each of the models (a-m) below, circle one type of question (i-viii) it is commonly used for.
    For models that have more than one correct answer, choose any one correct answer;
    for models that have no correct answer listed, do not circle anything.
\n
    Models:
        \na. ARIMA
        \nb. CART
        \nc. Cross validation
        \nd. CUSUM
        \ne. Exponential smoothing
        \nf. GARCH
        \ng. k-means
        \nh. k-nearest-neighbor
        \ni. Linear regression
        \nj. Logistic regression
        \nk. Principal component analysis
        \nl. Random forest
        \nm. Support vector machine
\n
    Question Types:
        \ni. Change detection
        \nii. Classification
        \niii. Clustering
        \niv. Feature-based prediction of a probability
        \nv. Feature-based prediction of a value
        \nvi. Time-series-based prediction
        \nvii. Validation
        \nviii. Variance estimation

    """

midterm_1_q2 = """
A1. \nSelect all of the following models that are designed for use with attribute/feature data (i.e., not time-series data):

\n- CUSUM
\n- Logistic regression
\n- Support vector machine
\n- GARCH
\n- Random forest
\n- k-means
\n- Linear regression
\n- k-nearest-neighbor
\n- ARIMA
\n- Principal component analysis
\n- Exponential smoothing
"""

midterm_1_q3 = """
Question A1\n
In the soft classification SVM model where we select coefficients to minimize the following formula:
Σ_{j=1}^n max{0, 1 - (Σ_{i=1}^m a_ix_ij + a_0)y_j} + C Σ_{i=1}^m a_i^2
Select all of the following statements that are correct.

\n- Decreasing the value of C could decrease the margin.
\n- Allowing a larger margin could decrease the number of classification errors in the training set.
\n- Decreasing the value of C could increase the number of classification errors in the training set.

Question A2
In the hard classification SVM model, it might be desirable to not put the classifier in a location that has equal margin on both sides... (select all correct answers):

\n- ...because moving the classifier will usually result in fewer classification errors in the validation data.
\n- ...because moving the classifier will usually result in fewer classification errors in the test data.
\n- ...when the costs of misclassifying the two types of points are significantly different.
"""

midterm_1_q4 = """
Question A1\n
Select whether a supervised learning model (like regression) is more directly appropriate than an unsupervised learning model (like dimensionality reduction).

Definition:\n

Supervised Learning: Machine learning where the "correct" answer or outcome is known for each data point in the training set.
Regression: A type of supervised learning where the model predicts a continuous outcome.
Unsupervised Learning Model: Machine learning where the "correct" answer is not known for the data points in the training set.
Dimensionality Reduction: A process in unsupervised learning of reducing the number of random variables under consideration, and can be divided into feature selection and feature extraction.
\n- In a dataset of residential property sales, for each property, the sale price is known, and the goal is to predict prices for new listings based on various attributes like location, size, and amenities:
\n- In a large dataset of customer reviews, there is no specific response variable, but the goal is to understand underlying themes and patterns in the text data:
\n- In a clinical trial dataset, for each participant, the response to a medication is known, and the task is to predict patient outcomes based on their medical history and trial data:
"""


# midterm_1_q4 = """
# The table below shows the Akaike Information Criterion (AIC), Corrected AIC, and Bayesian Information Criterion (BIC) for each of the models.

# Model       AIC     Corrected AIC    BIC
# 1           -5.58   -5.32            2.07
# 2           -5.67   -5.15            3.89
# 3           -6.51   -5.62            4.96
# 4           -4.77   -3.41            8.61
# 5           -2.80   -0.85            12.49
# 6           -1.31   1.35             15.90
# 7           0.19    3.71             19.31

# Question

# Based on the table above and the figure , select all of the following statements that are correct.

# - Adjusted R_squared and BIC give qualitatively opposite evaluations of Model 1.
# - Among Models 1 and 3, AIC suggests that Model 1 is e^(-6.51-(5.58))/2 = 62.8 percent as likely as Model 3 to be better.
# - Among Models 1 and 3, AIC suggests that Model 3 is e^(-6.51-(5.58))/2 = 62.8 percent as likely as Model 1 to be better.
# - BIC suggests that Model 3 is very likely to be better than Model 4.
# """

midterm_1_q5 =     """
Question A1\n
    An airline wants to predict airline passenger traffic for the upcoming year.
    For each of the specific questions (a-e) listed below, identify the question type (i-viii) it corresponds to.
    If a question does not match any of the listed types, leave it uncircled.

    Question Types:
        \ni. Change detection
        \nii. Classification
        \niii. Clustering
        \niv. Feature-based prediction of a value
        \nv. Feature-based prediction of a probability
        \nvi. Time-series-based prediction
        \nvii. Validation
        \nviii. Variance estimation

    Questions:
        \na. What is the probability that the airline will exceed 1 million passengers next year, considering current travel trends and economic factors?
        \nb. Among various forecasting models for airline passenger traffic, which one is likely to be the most accurate for the upcoming year?
        \nc. Based on the past decade's data, how many passengers are expected to travel via the airline next year?
        \nd. Analyzing the past fifteen years of data, has there been a significant change in passenger traffic during holiday seasons?
        \ne. Considering economic indicators and travel trends over the past 25 years, which years had the most similar passenger traffic patterns?
    """


midterm_1_q6 = """
Question A1\n
Information for all parts of the Question
Atlanta’s main library has collected the following day-by-day data over the past six years (more than 2000 data points):

\nx1 = Number of books borrowed from the library on that day
\nx2 = Day of the week
\nx3 = Temperature
\nx4 = Amount of rainfall
\nx5 = Whether the library was closed that day
\nx6 = Whether public schools were open that day
\nx7 = Number of books borrowed the day before
\nt = Time

a.

Select all data that are categorical (including binary data):\n

\n- Number of books borrowed from the library on that day
\n- Day of the week
\n- Temperature
\n- Amount of rainfall
\n- Whether the library was closed that day
\n- Whether public schools were open that day

Questions 1b and 1c\n

The library believes that there is a day-by-day word-of-mouth marketing effect: if more books were borrowed yesterday, then more books will be borrowed today (and if fewer books were borrowed yesterday, fewer books will be borrowed today), so they add a new predictor:

x7 = number of books borrowed the day before\n

b. If the library is correct that on average, if more books were borrowed yesterday, more books will be borrowed today (and vice versa), what sign (positive or negative) would you expect the new predictor's coefficient β to have?

\n- Negative, because higher values of x7 decrease the response (books borrowed today)
\n- Negative, because on average the number of books borrowed each day is decreasing
\n- Positive, higher values of x7 increase the response (books borrowed today) (correct)

c. Does x7 make the model autoregressive?\n

\n- Yes, because the model does not use any day t data to predict day t+1 borrowing.
\n- Yes, because the model uses day t-1 borrowing data to predict day t borrowing. (correct)
\n- No, because the model does not use previous response data to predict the day t response.
"""


midterm_1_q7 = """
Question A1\n
Select all of the following statements that are correct:

\n- It is likely that the first principal component has much more predictive power than each of the other principal components.
\n- It is likely that the first original covariate has much more predictive power than each of the other covariates.
\n- It is likely that the last original covariate has much less predictive power than each of the other covariates.
\n- The first principal component cannot contain information from all 7 original covariates. (correct)
"""

midterm_1_q8 = """

For each scenario, identify the most relevant statistical measure: AIC (Akaike Information Criterion), R-squared, Specificity, or Variance. Remember, Variance is included as a distractor and may not be the correct answer.

Definitions:\n

\nAIC (Akaike Information Criterion): This is a criterion for model selection, balancing the model's fit with the complexity of the model by penalizing the number of parameters.
\nR-squared: A statistical measure in regression models that quantifies the proportion of variance in the dependent variable explained by the independent variables.
\nSpecificity: Not relevant in this context.
\nVariance: A measure of the dispersion of a set of data points.

Choices:\n
\nA. AIC
\nB. R-squared
\nC. Specificity
\nD. Variance

Scenarios:\n
\n
Question A1\n
A researcher is assessing various linear regression models to predict future profits of a company, aiming to find a balance between model complexity and fit.
\n
Question A2\n
In a study evaluating the effect of advertising on sales, the analyst seeks to understand how changes in advertising budgets correlate with variations in sales figures.
\n
Question A3\n
An economist is choosing among different models to forecast economic growth, with a focus on avoiding overfitting in the presence of many potential predictor variables.
\n
Select the most appropriate statistical measure for each scenario.

"""

# midterm_1_q8 = """
# Recall the equations for triple exponential smoothing (Winters’/Holt-Winters method):
# S_t = α * (x_t / C_(t-L)) + (1 - α) * (S_(t-1) + T_(t-1))
# T_t = β * (S_t - S_(t-1)) + (1 - β) * T_(t-1)
# C_t = γ * (x_t / S_t) + (1 - γ) * C_(t-L)

# A construction vehicle manufacturer wants to use this model to analyze a production process
# where construction vehicles are produced in batches of exactly 170, and a batch takes an
# average of 9 days to be completed (usually between 8 and 10). Our data includes the day each
# vehicle’s production is completed, its sequence in the batch, the day within the batch that it
# was completed, and the number of hours the vehicle operated before its first breakdown.

# Based on this data, the manufacturer wants to use a triple exponential smoothing model to
# determine whether any patterns exist in the number of hours before the first breakdown, based
# on a vehicle’s sequence number in its batch.

# For each of the mathematical terms on the left, pick the appropriate number or description
# from the right.
# a. x_t
#     i. 170
#     ii. 9
#     iii. Sequence in batch
#     iv. Day within batch that vehicle was produced
#     v. Hours of operation before first breakdown

# b. L
#     i. 170
#     ii. 9
#     iii. Sequence in batch
#     iv. Day within batch that vehicle was produced
#     v. Hours of operation before first breakdown

# c. If the manufacturer observes that the values of C are generally close to 1, except that
# they are significantly lower than 1 for vehicles built near the beginning of batches, what
# can be concluded?
# CHOICES
#     i. There is no effect of sequence in batch on the number of hours before the first
#        breakdown.
#     ii. Vehicles built early in a batch tend to break down more quickly.
#     iii. Vehicles built early in a batch tend to break down more quickly, because
#          workers are adjusting to the different specifications in a each new batch.
#     iv. Vehicles built early in a batch tend to take longer to break down.
#     v. Vehicles built early in a batch tend to take longer to break down, because
#        workers are paying closer attention to their work early in each new batch.

# d. If the values of T tend to be slightly positive, what can be concluded?
# CHOICES
#     i. Vehicles built more recently tend to take longer to break down.
#     ii. Vehicles built more recently tend to break down more quickly.

# e. Suppose the manufacturer wanted to use a regression model to answer the same
# question, using the same data: two predictors (sequence in batch and day within batch)
# and one response (hours of operation before first breakdown).
# If the manufacturer first used principal component analysis on the data, what would you
# expect?
# CHOICES
#     i. The first component would be much more important than the second.
#     ii. The second component would be much more important than the first.
#     iii. The two components would have approximately the same importance.
# """

midterm_1_q9 = """
Question A1: Model Suitability Analysis\n

For each statistical and machine learning model listed below, select the type of analysis it is best suited for.
There may be more than one correct answer for each model, but you need only choose one. Assume 1 of 1 attempt for each part.
\n
A1. Time Series Analysis (e.g., ARMA, ARIMA)
\n- Predicting future values in a time-series dataset.
\n- Classifying items based on time-dependent features.
\n- Analyzing the seasonal components of time-series data.
\n- Estimating the probability of an event occurring in the future.
\n
A2. k-Nearest-Neighbor Classification (kNN)
\n- Using feature data to predict the amount of something two time periods in the future.
\n- Using feature data to predict the probability of something happening two time periods in the future.
\n- Using feature data to predict whether or not something will happen two time periods in the future.
\n- Using time-series data to predict the amount of something two time periods in the future.
\n- Using time-series data to predict the variance of something two time periods in the future.
\n
A3. Exponential Smoothing
\n- Using feature data to predict the amount of something two time periods in the future.
\n- Using feature data to predict the probability of something happening two time periods in the future.
\n- Using feature data to predict whether or not something will happen two time periods in the future.
\n- Using time-series data to predict the amount of something two time periods in the future.
\n- Using time-series data to predict the variance of something two time periods in the future.

"""

midterm_1_q10 = """
Question A1: Model Suitability Analysis\n

For each statistical and machine learning model listed below, select the type of analysis it is best suited for.
There may be more than one correct answer for each model, but you need only choose one. Assume 1 of 1 attempt for each part.

\nA1. Ridge Regression
\n- Predicting a continuous response variable with feature data.
\n- Dealing with multicollinearity in regression analysis.
\n- Forecasting future values in a time-series dataset.
\n- Classifying binary outcomes.

\nA2. Lasso Regression
\n- Selecting important features in a large dataset.
\n- Predicting a numerical outcome based on feature data.
\n- Analyzing patterns in time-series data.
\n- Identifying categories in unstructured data.

\nA3. Principal Component Analysis (PCA)
\n- Reducing the dimensionality of a large dataset.
\n- Forecasting trends in a time-series dataset.
\n- Classifying items into categories based on feature data.
\n- Detecting changes in the variance of a dataset over time.

"""
midterm_1_q11 = """
Question A1: Model Suitability Analysis\n

For each statistical and machine learning model listed below, select the type of analysis it is best suited for.
There may be more than one correct answer for each model, but you need only choose one. Assume 1 of 1 attempt for each part.

\nA1. Decision Trees (e.g., CART)
\n- Predicting the category of an item based on feature data.
\n- Forecasting numerical values two time periods in the future.
\n- Identifying clusters in feature data.
\n- Analyzing the variance in time-series data.

\nA2. Random Forest
\n- Predicting the likelihood of an event based on feature data.
\n- Classifying items into categories based on feature data.
\n- Estimating the amount of a variable two time periods in the future using time-series data.
\n- Detecting patterns in large datasets with many variables.

\nA3. Naive Bayes Classifier
\n- Classifying text data into predefined categories.
\n- Predicting future trends based on historical time-series data.
\n- Estimating the probability of an event occurring in the future.
\n- Analyzing variance in feature data.
"""

midterm_1_q12 = """
Question A1\n
Select all of the following reasons that data should not be scaled until point outliers are removed:
\n- If data is scaled first, the range of data after outliers are removed will be narrower than intended.
\n- If data is scaled first, the range of data after outliers are removed will be wider than intended.
\n- Point outliers would appear to be valid data if not removed before scaling.
\n- Valid data would appear to be outliers if data is scaled first.

Question A2\n
Select all of the following situations in which using a variable selection approach like lasso or stepwise regression would be important:
\n- It is too costly to create a model with a large number of variables.
\n- There are too few data points to avoid overfitting if all variables are included.
\n- Time-series data is being used.
\n- There are fewer data points than variables.
"""

midterm_1_q13 = """
Confusion Matrix for Shoplifting Prediction Model:
=======================================================================
                       Predicted Not Shoplifting   Predicted Shoplifting
Actual Not Shoplifting            1200                       300
Actual Shoplifting                 150                       350
=======================================================================

This confusion matrix represents the outcomes of a shoplifting prediction model. The model predicts whether an individual is likely to commit shoplifting ('Predicted Shoplifting')
or not ('Predicted Not Shoplifting'), and the results are compared against the actual occurrences ('Actual Shoplifting' and 'Actual Not Shoplifting').

Questions about the Shoplifting Prediction Model's Confusion Matrix:

Question A1:\n
Calculate the model's accuracy (the proportion of true results among the total number of cases examined).
\nA) (1200 + 350) / (1200 + 300 + 150 + 350)
\nB) (1200 + 150) / (1200 + 300 + 150 + 350)
\nC) (300 + 350) / (1200 + 300 + 150 + 350)

Question A2:\n
Determine the model's precision for shoplifting predictions (the proportion of correctly predicted shoplifting incidents to the total predicted as shoplifting).
\nA) 350 / (300 + 350)
\nB) 1200 / (1200 + 150)
\nC) 350 / (1200 + 350)

Question A3:\n
Calculate the model's recall for shoplifting predictions (the ability of the model to identify actual shoplifting incidents).
\nA) 350 / (150 + 350)
\nB) 300 / (1200 + 300)
\nC) 1200 / (1200 + 150)

Question A4:\n
Based on the confusion matrix, which statement is true regarding the model's predictions?
\nA) The model is more accurate in predicting non-shoplifting incidents than shoplifting incidents.
\nB) The model has the same accuracy for predicting shoplifting and non-shoplifting incidents.
\nC) The model is more accurate in predicting shoplifting incidents than non-shoplifting incidents.

"""
midterm_1_q14 = """
A1\n
Matching\n
Choices:\n
\nA. Classification
\nB. Clustering
\nC. Dimensionality Reduction
\nD. Outlier Detection

\nA1. Astronomers have a collection of long-exposure CCD images of distant celestial objects. They are unsure about the types of these objects and seek to group similar ones together. Which method is more suitable?
\nA2. An astronomer has manually categorized hundreds of images and now wishes to use analytics to automatically categorize new images. Which approach is most fitting?
\nA3. A data scientist wants to reduce the complexity of a high-dimensional dataset to visualize it more effectively, while preserving as much information as possible. Which technique should they use?
\nA4. A financial analyst is examining a large set of transaction data to identify unusual transactions that might indicate fraudulent activity. Which method is most appropriate?
"""

midterm_1_q15 = """
Question A1:\n

A retail company operates in various regions and is interested in optimizing its inventory management. The company has historical sales data from multiple stores and wants to predict future sales volumes for each product in different locations. This forecasting will help in efficient stock allocation and reducing inventory costs. The company also wants to understand the factors influencing sales to make strategic decisions. Which of the following models/approaches could the company use to predict future sales and understand sales dynamics?

CUSUM:\n
\nDiscrete event simulation: A simulation that models a system that changes when specific events occur.
\nLinear Regression:
\nLogistic Regression Tree:
\nRandom Linear Regression Forest:
"""

# midterm_1_q15 = """
# A modeler built a support vector machine (SVM) model for a problem, and found that it correctly
# predicted 86 percent of the training set and 76 percent of the validation set.
# a. When evaluated on the test data set, the expected correct prediction percent for the
# SVM model is…
#     i. …greater than 86 percent.
#     ii. …equal to 86 percent.
#     iii. …greater than 76 percent and less than 86 percent
#     iv. …equal to 76 percent
#     v. …less than 76 percent

# Later, the modeler created a second SVM model and a k‐nearest‐neighbor (kNN) model. The
# performance of each model on the training and validation data sets is shown in the table below.

# Correct prediction percent (training set)
# Correct prediction percent (validation set)
# SVM model 1          86 percent                       76 percent
# SVM model 2          84 percent                       45 percent
# kNN model            85 percent                       76 percent
# b. Which model is most likely to be overfit?

# c. (Based on the table above, which model should you select?
#     i. SVM model 1
#     ii. SVM model 2
#     iii. kNN model
#     iv. Either SVM model 1 or kNN model, but not SVM model 2
#     v. There’s not much difference between the three models

# d. Suppose SVM model 1 is selected as best, and its correct prediction percent on the test
# data set is 72 percent. An unbiased estimate of SVM model 1’s correct prediction percent on a new test
# data set is
#     i. Greater than 72 percent
#     ii. Equal to 72 percent
#     iii. Less than 72 percent
# """

midterm_1_q16 = """
Question A1\n
A company has noticed an increasing trend in customer service calls on Mondays over the past 15 years. The company wants to analyze whether there has been a significant change in this Monday trend in customer service calls during this period. Select all of the approaches that might reasonably be correct.
\n
i. Develop 15 separate logistic regression models, one for each year, with "is it Monday?" as one of the predictor variables; then apply a CUSUM analysis on the yearly coefficients for the Monday variable.
\n
ii. Implement time series forecasting using ARIMA, focusing on Mondays for the 780 weeks, and then use CUSUM on the forecasted values to identify any significant shifts.
\n
iii. Apply CUSUM directly on the volume of customer service calls received each of the 780 Mondays over the past 15 years.
"""

midterm_1_q17 = """
Question A1\n
A regional supermarket chain has collected day-to-day data over the last five years (approximately 1800 data points):

\nx1 = Number of customers visiting the store that day
\nx2 = Day of the week
\nx3 = Whether the day was part of a promotional event
\nx4 = Local unemployment rate on that day
\nx5 = Average temperature on that day
\nx6 = Local sports team win or loss on the previous day
\na. (3 points) Select all data that are categorical.

The supermarket has built three models using the linear formula b0 + b1x1 + b2x2 + b3x3 + b4x4 + b5x5 + b6x6:

\nA linear regression model
\nA logistic regression model
\nA k-nearest neighbors model
\nb.  For each of the following scenarios (i-iii), which model (1, 2, or 3) would you suggest using?
\ni. The supermarket wants to estimate the total number of customers visiting the store each day.
\nii. The supermarket aims to predict the likelihood of having more than 500 customers in the store each day.
\niii. The supermarket seeks to classify days into high or low customer traffic based on a threshold of 500 customers.

A regional supermarket chain has implemented a triple exponential smoothing (Holt-Winters) model to forecast the number of customers visiting the store each day. The model includes a multiplicative seasonal pattern with a weekly cycle (i.e., L=7).

Given that the supermarket experiences regular customer patterns with minimal random day-to-day variation, they are determining the optimal value for α (the level smoothing constant).

i. What should they expect the best value of α to be, considering the consistency in customer visits?

\nα < 0
\n0 < α < ½
\n½ < α < 1
\nα > 1
"""

midterm_1_q18 = """
Question A1\n
A regional healthcare provider has collected extensive data on patient visits over the years, including patient demographics, symptoms, diagnoses, treatments, and outcomes. The organization now wants to leverage this data to predict patient readmission risks and identify key factors that contribute to higher readmission rates. This insight will help them in allocating resources more effectively and improving patient care strategies. Choose the appropriate models/approaches from the list below that the healthcare provider could use for predicting patient readmissions and understanding the underlying factors.

\n-CUSUM: A method for change detection that compares the mean of an observed distribution with a threshold level for change, abbreviated as cumulative sum.
\n-K-nearest-neighbor classification: A non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space.
\n-Logistic Regression: A statistical model that in its basic form uses a logistic function to model a binary dependent variable, often used for binary classification.
\n-Multi-armed bandit: A problem in which a fixed limited set of resources must be allocated between competing (alternative) choices in a way that maximizes their expected gain.
\n-Support Vector Machine: A supervised learning model with associated learning algorithms that analyze data used for classification and regression analysis.
"""

# midterm_1_q18 = """
# A city park has been collecting the following day-by-day data over the past five years (around 1800 data points):

# x1 = Number of visitors to the park on that day
# x2 = Day of the week
# x3 = Weather condition (Sunny, Cloudy, Rainy, etc.)
# x4 = Temperature
# x5 = Whether there was a special event in the park that day
# x6 = Whether local schools were in session that day
# x7 = Number of visitors the day before
# t = Time
# Question a

# Select all data that are categorical (including binary data):

# Number of visitors to the park on that day
# Day of the week
# Weather condition
# Temperature
# Whether there was a special event in the park that day
# Whether local schools were in session that day
# Questions b and c

# The park management believes that there is a day-by-day effect related to visitor patterns: if more people visited the park yesterday, then more are likely to visit today (and if fewer people visited yesterday, fewer are likely to visit today), leading them to add a new predictor:

# x7 = number of visitors the day before

# b. If the park's assumption is correct that on average, more visitors yesterday leads to more visitors today (and vice versa), what sign (positive or negative) would you expect the new predictor's coefficient β to have?

# Negative, because higher values of x7 decrease the response (number of visitors today)
# Negative, because on average the number of visitors each day is decreasing
# Positive, higher values of x7 increase the response (number of visitors today)
# c. Does x7 make the model autoregressive?

# Yes, because the model does not use any day t data to predict day t+1 visitor numbers.
# Yes, because the model uses day t-1 visitor data to predict day t visitor numbers.
# No, because the model does not use previous response data to predict the day t response.
# """
midterm_1_q19 = midterm_1_q18
# midterm_1_q19 = """
# Energy Efficiency Prediction Models Analysis

# A team of researchers has evaluated several models for predicting the energy efficiency of buildings. The performance of these models is measured using different criteria: R-squared, Adjusted R-squared, Akaike Information Criterion (AIC), and Bayesian Information Criterion (BIC). The following table summarizes the results:

# Model 1: R-squared = 0.75, Adjusted R-squared = 0.74, AIC = -150, BIC = -140
# Model 2: R-squared = 0.78, Adjusted R-squared = 0.77, AIC = -155, BIC = -145
# Model 3: R-squared = 0.72, Adjusted R-squared = 0.71, AIC = -148, BIC = -138
# Model 4: R-squared = 0.76, Adjusted R-squared = 0.75, AIC = -152, BIC = -142
# Model 5: R-squared = 0.79, Adjusted R-squared = 0.78, AIC = -157, BIC = -147
# Questions:

# a. Which model demonstrates the best overall fit based on the combination of all four criteria (R-squared, Adjusted R-squared, AIC, and BIC)?

# b. If the primary concern is to avoid overfitting while still achieving a good fit, which model would be the most suitable choice?

# c. If the primary focus is on maximizing the explained variance of the model, which model should be selected? Choose one:

# Model 1
# Model 2
# Model 3
# Model 4
# Model 5
# This question format assesses the understanding of key statistical metrics used in model evaluation and selection, with a particular emphasis on the trade-offs between model fit, complexity, and the risk of overfitting in the context of linear regression modeling for energy efficiency prediction.
# """

# midterm_1_q20 = """
# Recall the equations for triple exponential smoothing (Winters’/Holt-Winters method):
# S_t = α * (x_t / C_(t-L)) + (1 - α) * (S_(t-1) + T_(t-1))
# T_t = β * (S_t - S_(t-1)) + (1 - β) * T_(t-1)
# C_t = γ * (x_t / S_t) + (1 - γ) * C_(t-L)

# A pharmaceutical company produces medications in batches. Each batch consists of exactly 200 units, and it takes an average of 7 days to complete a batch. The company has recorded data including the completion date of each unit, its sequence number in the batch, the day within the batch when it was completed, and the time until the first reported efficacy drop in patients.

# The company wants to use a triple exponential smoothing model to investigate if there are any patterns in the time until efficacy drop, based on a unit’s sequence number in its batch.

# For each of the mathematical terms on the left, pick the appropriate number or description from the right.

# a. y_t

# 200
# 7
# Sequence number in batch
# Day within batch that unit was completed
# Time until first reported efficacy drop
# b. L

# 200
# 7
# Sequence number in batch
# Day within batch that unit was completed
# Time until first reported efficacy drop

# c. If the company notices that the values of I are generally close to 1, but significantly lower for units produced at the end of the batches, what can be concluded?

# There is no effect of sequence in batch on the time until efficacy drop.
# Units produced later in a batch tend to show efficacy drop more quickly.
# Units produced later in a batch tend to show efficacy drop more quickly, possibly due to rushed production processes.
# Units produced later in a batch tend to maintain efficacy longer.
# Units produced later in a batch tend to maintain efficacy longer, possibly due to increased proficiency in production toward the end of the batch.

# d. If the values of T tend to be slightly negative, what can be concluded?

# Units produced more recently tend to maintain efficacy longer.
# Units produced more recently tend to show efficacy drop more quickly.

# e. Suppose the company wanted to use a regression model to investigate the same question, using the same data: two predictors (sequence in batch and day within batch) and one response (time until efficacy drop).
# If the company first applied principal component analysis to the data, what would you expect?

# The first component would be much more important than the second.
# The second component would be much more important than the first.
# The two components would have approximately the same importance.
# This question assesses understanding of time series analysis and its application in production quality control, as well as the integration of statistical techniques like principal component analysis in predictive modeling.

# """

midterm_1_q20 = """
Question A1\n
A pharmaceutical company produces medications in batches of 200 units, with each batch taking an average of 7 days to complete. They have data on the sequence number of each unit in the batch, the day it was completed within the batch, and the time until the first reported efficacy drop in patients. The company plans to use triple exponential smoothing to analyze patterns in the time until efficacy drop based on a unit’s sequence number in its batch.

For each description below, choose the most appropriate option:

\na. The observed variable (y_t) in the context of this study:
\n-Sequence number in batch
\n-Day within batch that unit was completed
\n-Time until first reported efficacy drop

\nb. The seasonal length (L) in this analysis:
\n-200 (number of units in a batch)
\n-7 (days to complete a batch)

\nc. If the seasonal component (C_t) values are consistently lower towards the end of the batch, it suggests:
\n-Units produced later in a batch tend to show efficacy drop more quickly.
\n-Units produced later in a batch tend to maintain efficacy longer.
\n-This question is designed to assess the application of time series analysis in a production context and the interpretation of its results.

"""

midterm_1_q21 = """
Question A1\n
Data Analysis and Modeling in Healthcare
\n
A healthcare analytics team is working on various models to analyze patient data for improving treatment outcomes. They have collected extensive patient data over the years, including demographics, treatment details, and health outcomes.
\n
Classification and Clustering in Patient Segmentation
\n
The team wants to segment patients into groups for targeted treatment approaches. They have the following data points for each patient: age, gender, diagnosis, treatment type, and recovery time.
\n
A1. Which model would be best for classifying patients into high-risk and low-risk categories based on their treatment outcomes?
\nCusum
\nK-Nearest Neighbors
\nSupport Vector Machines

A2.For clustering patients based on similarities in their diagnosis and treatment types, which algorithm would be most effective?
\nK-Means Clustering
\nPCA
\nGARCH Variance Clustering
\n
Question b: Time Series Analysis for Predicting Treatment Efficacy

The team is also interested in predicting the efficacy of treatments over time.

A3. If the team wants to forecast treatment efficacy based on past trends and seasonal variations, which model should they use?
\nARIMA
\nExponential Smoothing
\nRandom Forests

A4.To detect significant changes in treatment efficacy over time, which method would be most suitable?
\nCUSUM
\nPrincipal Component Analysis
\nBox-Cox Transformation
"""

midterm_1_q22 = """
Question A1: \nSupport Vector Machines (SVM) and K-Nearest Neighbor (KNN) in Classification

a. A bank is developing a model to classify loan applicants as high-risk or low-risk. Given the importance of minimizing the misclassification of high-risk applicants, which model would be more suitable?
\nSVM
\nKNN

b. In a medical diagnosis system, which model would be preferable for classifying patients based on a dataset with many overlapping characteristics?
\nSVM
\nKNN

Question A2: Validation and Model Assessment
\na. A marketing team has developed several predictive models for customer behavior. To avoid overfitting, which approach should they use for model assessment?
\nCross-validation
\nTraining on the entire dataset
\nb. When choosing between two different models for predicting sales, one with a lower AIC and one with a higher BIC, which model should be preferred considering both simplicity and likelihood?
\nModel with lower AIC
\nModel with higher BIC
"""

midterm_1_q23 = """
Question A1: \nClustering and Outlier Detection in Data Analysis

\na. A retailer wants to segment their customer base for targeted marketing. Which clustering method would be best for a dataset with well-defined, separate customer groups?
\nK-means Clustering
\nPCA

\nb. In analyzing customer purchase data, the team identifies several extreme values. What is the most appropriate initial step in handling these outliers?
\nRemoving them from the dataset
\nInvestigating their source and context

\nQuestion A2: Time Series Analysis and Exponential Smoothing
\na. A utility company is analyzing electricity usage patterns over time. To forecast future usage that exhibits both trend and seasonality, which method would be most appropriate?
\nARIMA
\nExponential Smoothing with trend and seasonality

\nb. If the company wants to smooth out short-term fluctuations in daily usage data while giving more weight to recent observations, what should be the approach to setting the alpha value in exponential smoothing?
\nA high alpha value
\nA low alpha value
"""

midterm_1_q24 = """
Question A1: \nClassification Techniques

\nA financial institution is implementing a new system to classify loan applicants based on risk.

\na. Which classifier would be more effective for categorizing applicants into 'high risk' and 'low risk', considering the cost of misclassification?
\nLinear Regression
\nK-Nearest Neighbor (KNN)
\nSupport Vector Machine (SVM)
\nRandom Forest

\nb. In a scenario where the bank needs to identify potential fraudulent transactions, which approach should they use, given the transactions data is highly imbalanced?
\nHard Classifiers
\nSoft Classifiers
\nDecision Trees
\nBayesian Classifiers

\nQuestion A2: Model Validation and Testing
An e-commerce company is evaluating different models for predicting customer purchase behavior.
\na. To ensure the chosen model is not overfitting, which method should be used for validating the model's effectiveness?
\nCross-Validation
\nTraining on Entire Dataset
\nAIC/BIC Comparison
\nHoldout Method

\nb. If the model performs well on the training data but poorly on the validation data, what might this indicate?
\nThe model is underfitting
\nThe model is overfitting
\nThe model is perfectly fitted
\nThe model is not complex enough
"""

midterm_1_q25 = """
Question A1: \nClustering in Market Segmentation

A marketing agency is segmenting its audience for targeted advertising campaigns.
\na. For creating customer segments based on shopping behavior and preferences, which clustering method would be most suitable?
\nK-means Clustering
\nKNN Clustering
\nPCA
\nPoisson Variance Classification

Question A2: Regression Analysis for Sales Forecasting
A retail chain is analyzing factors affecting its sales performance.
\na. To predict future sales based on factors like store location, advertising spend, and local demographics, which regression method should be employed?
\nLinear Regression
\nPoisson Regression
\nBayesian Regression
\nLasso Regression

\nb. The retailer needs to understand the relationship between temperature and outdoor sales. If the relationship is non-linear, what should they consider in their regression model?
\nTransformation and Interaction Terms
\nLogistic Regression
\nPolynomial Regression
\nRidge Regression
"""




MIDTERM_1_QUESTIONS = [midterm_1_q1,midterm_1_q2,midterm_1_q3,midterm_1_q4,midterm_1_q5,midterm_1_q6,
                       midterm_1_q7,midterm_1_q8,midterm_1_q9,midterm_1_q10,midterm_1_q11,midterm_1_q12,
                       midterm_1_q13,midterm_1_q14,midterm_1_q15,midterm_1_q16,midterm_1_q17,midterm_1_q18,
                       midterm_1_q19,midterm_1_q20,midterm_1_q21,midterm_1_q22,midterm_1_q23,midterm_1_q24,
                       midterm_1_q25]
