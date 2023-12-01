
midterm_1_q1 =     """
    For each of the models (a-m) below, circle one type of question (i-viii) it is commonly used for.
    For models that have more than one correct answer, choose any one correct answer;
    for models that have no correct answer listed, do not circle anything.

    Models:
        a. ARIMA
        b. CART
        c. Cross validation
        d. CUSUM
        e. Exponential smoothing
        f. GARCH
        g. k-means
        h. k-nearest-neighbor
        i. Linear regression
        j. Logistic regression
        k. Principal component analysis
        l. Random forest
        m. Support vector machine

    Question Types:
        i. Change detection
        ii. Classification
        iii. Clustering
        iv. Feature-based prediction of a probability
        v. Feature-based prediction of a value
        vi. Time-series-based prediction
        vii. Validation
        viii. Variance estimation

    Returns:
        A dictionary mapping each model to its associated question type.
    """

midterm_1_q2 = """
Select all of the following models that are designed for use with attribute/feature data (i.e., not time-series data):

- CUSUM
- Logistic regression
- Support vector machine
- GARCH
- Random forest
- k-means
- Linear regression
- k-nearest-neighbor
- ARIMA
- Principal component analysis
- Exponential smoothing
"""

midterm_1_q3 = """
Question 1
In the soft classification SVM model where we select coefficients to minimize the following formula:
Σ_{j=1}^n max{0, 1 - (Σ_{i=1}^m a_ix_ij + a_0)y_j} + C Σ_{i=1}^m a_i^2
Select all of the following statements that are correct.

- Decreasing the value of C could decrease the margin.
- Allowing a larger margin could decrease the number of classification errors in the training set.
- Decreasing the value of C could increase the number of classification errors in the training set.

You have used 1 of 1 attempt. Some problems have options such as save, reset, hints, or show answer. These options follow the Submit button.

Question 2
In the hard classification SVM model, it might be desirable to not put the classifier in a location that has equal margin on both sides... (select all correct answers):

- ...because moving the classifier will usually result in fewer classification errors in the validation data.
- ...because moving the classifier will usually result in fewer classification errors in the test data.
- ...when the costs of misclassifying the two types of points are significantly different.
"""

midterm_1_q4 = """
The table below shows the Akaike Information Criterion (AIC), Corrected AIC, and Bayesian Information Criterion (BIC) for each of the models.

Model       AIC     Corrected AIC    BIC
1           -5.58   -5.32            2.07
2           -5.67   -5.15            3.89
3           -6.51   -5.62            4.96
4           -4.77   -3.41            8.61
5           -2.80   -0.85            12.49
6           -1.31   1.35             15.90
7           0.19    3.71             19.31

Question

Based on the table above and the figure , select all of the following statements that are correct.

- Adjusted R_squared and BIC give qualitatively opposite evaluations of Model 1.
- Among Models 1 and 3, AIC suggests that Model 1 is e^(-6.51-(5.58))/2 = 62.8 percent as likely as Model 3 to be better.
- Among Models 1 and 3, AIC suggests that Model 3 is e^(-6.51-(5.58))/2 = 62.8 percent as likely as Model 1 to be better.
- BIC suggests that Model 3 is very likely to be better than Model 4.
"""

midterm_1_q5 =     """
    An airline wants to predict airline passenger traffic for the upcoming year.
    For each of the specific questions (a-e) listed below, identify the question type (i-viii) it corresponds to.
    If a question does not match any of the listed types, leave it uncircled.

    Question Types:
        i. Change detection
        ii. Classification
        iii. Clustering
        iv. Feature-based prediction of a value
        v. Feature-based prediction of a probability
        vi. Time-series-based prediction
        vii. Validation
        viii. Variance estimation

    Questions:
        a. What is the probability that the airline will exceed 1 million passengers next year, considering current travel trends and economic factors?
        b. Among various forecasting models for airline passenger traffic, which one is likely to be the most accurate for the upcoming year?
        c. Based on the past decade's data, how many passengers are expected to travel via the airline next year?
        d. Analyzing the past fifteen years of data, has there been a significant change in passenger traffic during holiday seasons?
        e. Considering economic indicators and travel trends over the past 25 years, which years had the most similar passenger traffic patterns?
    """


midterm_1_q6 = """
Information for all parts of the Question
Atlanta’s main library has collected the following day-by-day data over the past six years (more than 2000 data points):

x1 = Number of books borrowed from the library on that day
x2 = Day of the week
x3 = Temperature
x4 = Amount of rainfall
x5 = Whether the library was closed that day
x6 = Whether public schools were open that day
x7 = Number of books borrowed the day before
t = Time

Question a

Select all data that are categorical (including binary data):

- Number of books borrowed from the library on that day
- Day of the week (correct)
- Temperature
- Amount of rainfall
- Whether the library was closed that day (correct)
- Whether public schools were open that day

Questions b and c

The library believes that there is a day-by-day word-of-mouth marketing effect: if more books were borrowed yesterday, then more books will be borrowed today (and if fewer books were borrowed yesterday, fewer books will be borrowed today), so they add a new predictor:

x7 = number of books borrowed the day before

b. If the library is correct that on average, if more books were borrowed yesterday, more books will be borrowed today (and vice versa), what sign (positive or negative) would you expect the new predictor's coefficient β to have?

- Negative, because higher values of x7 decrease the response (books borrowed today)
- Negative, because on average the number of books borrowed each day is decreasing
- Positive, higher values of x7 increase the response (books borrowed today) (correct)

c. Does x7 make the model autoregressive?

- Yes, because the model does not use any day t data to predict day t+1 borrowing.
- Yes, because the model uses day t-1 borrowing data to predict day t borrowing. (correct)
- No, because the model does not use previous response data to predict the day t response.
"""


midterm_1_q7 = """
Select all of the following statements that are correct:

- It is likely that the first principal component has much more predictive power than each of the other principal components.
- It is likely that the first original covariate has much more predictive power than each of the other covariates.
- It is likely that the last original covariate has much less predictive power than each of the other covariates.
- The first principal component cannot contain information from all 7 original covariates. (correct)
"""

midterm_1_q8 = """
Recall the equations for triple exponential smoothing (Winters’/Holt-Winters method):
S_t = α * (x_t / C_(t-L)) + (1 - α) * (S_(t-1) + T_(t-1))
T_t = β * (S_t - S_(t-1)) + (1 - β) * T_(t-1)
C_t = γ * (x_t / S_t) + (1 - γ) * C_(t-L)

A construction vehicle manufacturer wants to use this model to analyze a production process
where construction vehicles are produced in batches of exactly 170, and a batch takes an
average of 9 days to be completed (usually between 8 and 10). Our data includes the day each
vehicle’s production is completed, its sequence in the batch, the day within the batch that it
was completed, and the number of hours the vehicle operated before its first breakdown.

Based on this data, the manufacturer wants to use a triple exponential smoothing model to
determine whether any patterns exist in the number of hours before the first breakdown, based
on a vehicle’s sequence number in its batch.

For each of the mathematical terms on the left, pick the appropriate number or description
from the right.
a. x_t
    i. 170
    ii. 9
    iii. Sequence in batch
    iv. Day within batch that vehicle was produced
    v. Hours of operation before first breakdown

b. L
    i. 170
    ii. 9
    iii. Sequence in batch
    iv. Day within batch that vehicle was produced
    v. Hours of operation before first breakdown

c. If the manufacturer observes that the values of C are generally close to 1, except that
they are significantly lower than 1 for vehicles built near the beginning of batches, what
can be concluded?
CHOICES
    i. There is no effect of sequence in batch on the number of hours before the first
       breakdown.
    ii. Vehicles built early in a batch tend to break down more quickly.
    iii. Vehicles built early in a batch tend to break down more quickly, because
         workers are adjusting to the different specifications in a each new batch.
    iv. Vehicles built early in a batch tend to take longer to break down.
    v. Vehicles built early in a batch tend to take longer to break down, because
       workers are paying closer attention to their work early in each new batch.

d. If the values of T tend to be slightly positive, what can be concluded?
CHOICES
    i. Vehicles built more recently tend to take longer to break down.
    ii. Vehicles built more recently tend to break down more quickly.

e. Suppose the manufacturer wanted to use a regression model to answer the same
question, using the same data: two predictors (sequence in batch and day within batch)
and one response (hours of operation before first breakdown).
If the manufacturer first used principal component analysis on the data, what would you
expect?
CHOICES
    i. The first component would be much more important than the second.
    ii. The second component would be much more important than the first.
    iii. The two components would have approximately the same importance.
"""

midterm_1_q9 = """
Question 1: Model Suitability Analysis

For each statistical and machine learning model listed below, select the type of analysis it is best suited for.
There may be more than one correct answer for each model, but you need only choose one. Assume 1 of 1 attempt for each part.

A. Time Series Analysis (e.g., ARMA, ARIMA)
- Predicting future values in a time-series dataset.
- Classifying items based on time-dependent features.
- Analyzing the seasonal components of time-series data.
- Estimating the probability of an event occurring in the future.

B. k-Nearest-Neighbor Classification (kNN)
- Using feature data to predict the amount of something two time periods in the future.
- Using feature data to predict the probability of something happening two time periods in the future.
- Using feature data to predict whether or not something will happen two time periods in the future.
- Using time-series data to predict the amount of something two time periods in the future.
- Using time-series data to predict the variance of something two time periods in the future.

C. Exponential Smoothing
- Using feature data to predict the amount of something two time periods in the future.
- Using feature data to predict the probability of something happening two time periods in the future.
- Using feature data to predict whether or not something will happen two time periods in the future.
- Using time-series data to predict the amount of something two time periods in the future.
- Using time-series data to predict the variance of something two time periods in the future.

"""

midterm_1_q10 = """
Question 1: Model Suitability Analysis

For each statistical and machine learning model listed below, select the type of analysis it is best suited for.
There may be more than one correct answer for each model, but you need only choose one. Assume 1 of 1 attempt for each part.

A. Ridge Regression
- Predicting a continuous response variable with feature data.
- Dealing with multicollinearity in regression analysis.
- Forecasting future values in a time-series dataset.
- Classifying binary outcomes.

B. Lasso Regression
- Selecting important features in a large dataset.
- Predicting a numerical outcome based on feature data.
- Analyzing patterns in time-series data.
- Identifying categories in unstructured data.

C. Principal Component Analysis (PCA)
- Reducing the dimensionality of a large dataset.
- Forecasting trends in a time-series dataset.
- Classifying items into categories based on feature data.
- Detecting changes in the variance of a dataset over time.

"""
midterm_1_q11 = """
Question 1: Model Suitability Analysis

For each statistical and machine learning model listed below, select the type of analysis it is best suited for.
There may be more than one correct answer for each model, but you need only choose one. Assume 1 of 1 attempt for each part.

A. Decision Trees (e.g., CART)
- Predicting the category of an item based on feature data.
- Forecasting numerical values two time periods in the future.
- Identifying clusters in feature data.
- Analyzing the variance in time-series data.

B. Random Forest
- Predicting the likelihood of an event based on feature data.
- Classifying items into categories based on feature data.
- Estimating the amount of a variable two time periods in the future using time-series data.
- Detecting patterns in large datasets with many variables.

C. Naive Bayes Classifier
- Classifying text data into predefined categories.
- Predicting future trends based on historical time-series data.
- Estimating the probability of an event occurring in the future.
- Analyzing variance in feature data.
"""

midterm_1_q12 = """
Question 1
Select all of the following reasons that data should not be scaled until point outliers are removed:
- If data is scaled first, the range of data after outliers are removed will be narrower than intended.
- If data is scaled first, the range of data after outliers are removed will be wider than intended.
- Point outliers would appear to be valid data if not removed before scaling.
- Valid data would appear to be outliers if data is scaled first.

Question 2
Select all of the following situations in which using a variable selection approach like lasso or stepwise regression would be important:
- It is too costly to create a model with a large number of variables.
- There are too few data points to avoid overfitting if all variables are included.
- Time-series data is being used.
- There are fewer data points than variables.
"""

midterm_1_q13 = """
Confusion Matrix for Shoplifting Prediction Model:

                       Predicted Not Shoplifting   Predicted Shoplifting
Actual Not Shoplifting            1200                       300
Actual Shoplifting                 150                       350

This confusion matrix represents the outcomes of a shoplifting prediction model. The model predicts whether an individual is likely to commit shoplifting ('Predicted Shoplifting')
or not ('Predicted Not Shoplifting'), and the results are compared against the actual occurrences ('Actual Shoplifting' and 'Actual Not Shoplifting').

Questions about the Shoplifting Prediction Model's Confusion Matrix:

Question 1:
(1 point) Calculate the model's accuracy (the proportion of true results among the total number of cases examined).
A) (1200 + 350) / (1200 + 300 + 150 + 350)
B) (1200 + 150) / (1200 + 300 + 150 + 350)
C) (300 + 350) / (1200 + 300 + 150 + 350)

Question 2:
(1 point) Determine the model's precision for shoplifting predictions (the proportion of correctly predicted shoplifting incidents to the total predicted as shoplifting).
A) 350 / (300 + 350)
B) 1200 / (1200 + 150)
C) 350 / (1200 + 350)

Question 3:
(1 point) Calculate the model's recall for shoplifting predictions (the ability of the model to identify actual shoplifting incidents).
A) 350 / (150 + 350)
B) 300 / (1200 + 300)
C) 1200 / (1200 + 150)

Question 4:
(1 point) Based on the confusion matrix, which statement is true regarding the model's predictions?
A) The model is more accurate in predicting non-shoplifting incidents than shoplifting incidents.
B) The model has the same accuracy for predicting shoplifting and non-shoplifting incidents.
C) The model is more accurate in predicting shoplifting incidents than non-shoplifting incidents.

"""
midterm_1_q14 = """
1. A group of astronomers has a set of long-exposure CCD images of various distant objects. They do not know yet which types of object each one is and would like your help using analytics to determine which ones look similar. Which is more appropriate: classification or clustering?
- Classification
- Clustering

2. Suppose one astronomer has categorized hundreds of the images by hand, and now wants your help using analytics to automatically determine which category each new image belongs to. Which is more appropriate: classification or clustering?
- Classification
- Clustering
"""



midterm_1_q15 = """
A modeler built a support vector machine (SVM) model for a problem, and found that it correctly
predicted 86 percent of the training set and 76 percent of the validation set.
a. When evaluated on the test data set, the expected correct prediction percent for the
SVM model is…
    i. …greater than 86 percent.
    ii. …equal to 86 percent.
    iii. …greater than 76 percent and less than 86 percent
    iv. …equal to 76 percent
    v. …less than 76 percent

Later, the modeler created a second SVM model and a k‐nearest‐neighbor (kNN) model. The
performance of each model on the training and validation data sets is shown in the table below.

Correct prediction percent (training set)
Correct prediction percent (validation set)
SVM model 1          86 percent                       76 percent
SVM model 2          84 percent                       45 percent
kNN model            85 percent                       76 percent
b. Which model is most likely to be overfit?

c. (Based on the table above, which model should you select?
    i. SVM model 1
    ii. SVM model 2
    iii. kNN model
    iv. Either SVM model 1 or kNN model, but not SVM model 2
    v. There’s not much difference between the three models

d. Suppose SVM model 1 is selected as best, and its correct prediction percent on the test
data set is 72 percent. An unbiased estimate of SVM model 1’s correct prediction percent on a new test
data set is
    i. Greater than 72 percent
    ii. Equal to 72 percent
    iii. Less than 72 percent
"""
MIDTERM_1_QUESTIONS = [midterm_1_q1,midterm_1_q2,midterm_1_q3,midterm_1_q4,midterm_1_q5,midterm_1_q6,
                       midterm_1_q7,midterm_1_q8,midterm_1_q9,midterm_1_q10,midterm_1_q11,midterm_1_q12,
                       midterm_1_q13,midterm_1_q14,midterm_1_q15]
