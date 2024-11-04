


check_1_q1 = {
    'question': "A survey of 25 people recorded each person’s family size and type of car. Which of these is a data point?",
    'options_list': [
        'The 14th person’s family size and car type',
        'The 14th person’s family size',
        'The car type of each person'
    ],
    'correct_answer': 'The 14th person’s family size and car type',
    'explanation': "A data point is a single observation in a dataset. In this context, a complete record containing both the family size and car type represents a data point.",
    'chapter_information': 'midterm 1'
}

check_1_q2 = {
    'question': (
        "a. Which of these is structured data?"
        "\n- The contents of a person’s Twitter feed"
        "\n- The amount of money in a person’s bank account"
        "\n\n"
        "b. Which of these is time series data?"
        "\n- The average cost of a house in the United States every year since 1820"
        "\n- The height of each professional basketball player in the NBA at the start of the season"
    ),
    'correct_answer': (
        "a: The amount of money in a person’s bank account"
        "\nb: The average cost of a house in the United States every year since 1820"
    ),
    'explanation': (
        "Structured data is highly organized and easily searchable, such as numeric data in a bank account. Time series data is collected or recorded over time, like the average cost of a house over many years.",
    ),
    'chapter_information': 'midterm 1'
}

check_1_q3 = {
    'question': "When comparing models, if we use the same data to pick the best model as we do to estimate how good the best one is, what is likely to happen?",
    'options_list': [
        'The model will appear to be better than it really is.',
        'The model will appear to be worse than it really is.',
        'The model will appear to be just as good as it really is.'
    ],
    'correct_answer': 'The model will appear to be better than it really is.',
    'explanation': "Using the same data to select and evaluate a model can lead to overfitting, causing the model to perform better on the training data than it does in practice.",
    'chapter_information': 'midterm 1'
}

check_1_q4 = {
    'question': "In k-fold cross-validation, how many times is each part of the data used for training, and for validation?",
    'options_list': [
        'k times for training, and k times for validation',
        '1 time for training, and k-1 times for validation',
        'k-1 times for training, and 1 time for validation'
    ],
    'correct_answer': 'k-1 times for training, and 1 time for validation',
    'explanation': "In k-fold cross-validation, the data is split into k parts. Each part is used once for validation and k-1 times for training, providing a robust assessment of model performance.",
    'chapter_information': 'midterm 1'
}


check_1__q5 = {
    'question': "The k-means algorithm for clustering is a “heuristic” because...",
    'options_list': [
        '...it runs quickly.',
        '...it never gets the best answer.',
        '...it isn’t guaranteed to get the best answer.'
    ],
    'correct_answer': '...it isn’t guaranteed to get the best answer.',
    'explanation': "A heuristic algorithm like k-means doesn't guarantee an optimal solution. It uses a practical approach to find a solution within a reasonable time, but it may not always achieve the best answer.",
    'chapter_information': 'midterm 1'
}

check_1__q6 = {
    'question': "Which of these is generally a good reason to remove an outlier from your data set?",
    'options_list': [
        'The outlier is an incorrectly-entered data, not real data.',
        'Outliers like this only happen occasionally.'
    ],
    'correct_answer': 'The outlier is an incorrectly-entered data, not real data.',
    'explanation': "An outlier should be removed if it's due to incorrect data entry or isn't representative of actual data. Removing these kinds of outliers helps ensure data quality and validity.",
    'chapter_information': 'midterm 1'
}

check_1_q7 = {
    'question': "In the CUSUM model, having a higher threshold T makes it...",
    'options_list': [
        '...detect changes faster, and less likely to falsely detect changes.',
        '...detect changes faster, and more likely to falsely detect changes.',
        '...detect changes slower, and less likely to falsely detect changes.',
        '...detect changes slower, and more likely to falsely detect changes.'
    ],
    'correct_answer': '...detect changes slower, and less likely to falsely detect changes.',
    'explanation': "In the CUSUM model, a higher threshold T requires a larger deviation for change detection, leading to slower change detection but reducing the likelihood of false positives.",
    'chapter_information': 'midterm 1'
}

check_1_q8 = {
    'question': "A multiplicative seasonality, like in the Holt-Winters method, means that the seasonal effect is...",
    'options_list': [
        'The same regardless of the baseline value.',
        'Proportional to the baseline value.'
    ],
    'correct_answer': 'Proportional to the baseline value.',
    'explanation': "In multiplicative seasonality, the seasonal effect varies with the baseline value. Higher baselines lead to larger seasonal effects, while lower baselines lead to smaller seasonal effects.",
    'chapter_information': 'midterm 1'
}

check_1_q9 = {
    'question': "Is exponential smoothing better for short-term forecasting or long-term forecasting?",
    'options_list': [
        'Short-term',
        'Long-term'
    ],
    'correct_answer': 'Short-term',
    'explanation': "Exponential smoothing gives more weight to recent observations, making it ideal for short-term forecasting, where trends and patterns are influenced by recent data.",
    'chapter_information': 'midterm 1'
}

check_1_q10 = {
    'question': "Why is GARCH different from ARIMA and exponential smoothing?",
    'options_list': [
        'GARCH uses time series data',
        'GARCH is autoregressive',
        'GARCH estimates variance'
    ],
    'correct_answer': 'GARCH estimates variance',
    'explanation': "GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models variance and is used for forecasting volatility, unlike ARIMA and exponential smoothing, which focus on time-series forecasting.",
    'chapter_information': 'midterm 1'
}

check_1_q11 = {
    'question': "When would regression be used instead of a time series model?",
    'options_list': [
        'When there are other factors or predictors that affect the response',
        'When only previous values of the response affect its current value'
    ],
    'correct_answer': 'When there are other factors or predictors that affect the response',
    'explanation': "Regression is used when multiple factors or predictors affect a response, allowing the model to consider these additional influences. Time series models focus primarily on previous values of the response to predict future values.",
    'chapter_information': 'midterm 1'
}


check_1_q12 = {
    'question': "If two models are approximately equally good, measures like AIC and BIC will favor the simpler model. Simpler models are often better because...",
    'options_list': [
        'Simple models are easier to explain and “sell” to managers and executives',
        'The effects observed in simple models are easier for everyone, including analytics professionals, to understand',
        'Simple models are less likely to be over-fit to random effects',
        'All of the above'
    ],
    'correct_answer': 'All of the above',
    'explanation': "Simpler models have several advantages: they are easier to understand and explain, less prone to overfitting, and more straightforward to communicate to stakeholders.",
    'chapter_information': 'midterm 1'
}

check_1_q13 = {
    'question': "What does “heteroscedasticity” mean?",
    'options_list': [
        'The variance is different in different ranges of the data',
        'The variances of two samples of data are different from each other'
    ],
    'correct_answer': 'The variance is different in different ranges of the data',
    'explanation': "Heteroscedasticity refers to a condition where the variance of a variable changes across the data range. This can indicate that a model's assumptions may not hold true across all data points.",
    'chapter_information': 'midterm 1'
}

check_1_q14 = {
    'question': "A model is built to determine whether data points belong to a category or not. A “true negative” result is:",
    'options_list': [
        'A data point that is in the category, but the model incorrectly says it isn’t.',
        'A data point that is not in the category, but the model incorrectly says it is.',
        'A data point that is in the category, and the model correctly says it is.',
        'A data point that is not in the category, and the model correctly says so.',
        'A “Debbie Downer” (someone who often says negative things that bring down everyone’s mood).'
    ],
    'correct_answer': 'A data point that is not in the category, and the model correctly says so.',
    'explanation': "A true negative is when a data point that doesn't belong to a category is correctly identified as not belonging to that category. It indicates a correct negative classification.",
    'chapter_information': 'midterm 1'
}

check_1_q15 = {
    'question': "A common rule of thumb is to stop branching if a leaf would contain less than 5 percent of the data points. Why not keep branching and allow models to find very close fits to each very small subset of data?",
    'options_list': [
        'Actually, that sounds like a great idea – we should keep branching and let models find very close fits to very small subsets of data!',
        'Fitting to very small subsets of data will cause overfitting.',
        'Fitting to very small subsets of data will make the tree have too many leaves.'
    ],
    'correct_answer': 'Fitting to very small subsets of data will cause overfitting.',
    'explanation': "Overfitting occurs when a model is excessively complex, fitting to specific noise or variations in the training data, which reduces its ability to generalize to new data.",
    'chapter_information': 'midterm 1'
}

check_1_q16 = {
    'question': "In K-fold cross-validation, what does 'k' represent?",
    'options_list': [
        'The number of data points in the test set.',
        'The number of features in the dataset.',
        'The number of parts the data is split into.',
        'The number of attributes in the training set.'
    ],
    'correct_answer': 'The number of parts the data is split into.',
    'explanation': "In K-fold cross-validation, 'k' refers to the number of parts into which the data is split for training and validation. Each part serves as a validation set, while the others are used for training.",
    'chapter_information': 'midterm 1'
}

check_1_q17 = {
    'question': "What is the primary purpose of using a separate test data set in model validation?",
    'options_list': [
        'A. To optimize model parameters.',
        'B. To assess model effectiveness on random patterns.',
        'C. To choose the best-performing model.',
        'D. To estimate true model performance.'
    ],
    'correct_answer': 'D. To estimate true model performance.',
    'explanation': "A test data set provides an unbiased evaluation of a model's performance on unseen data, allowing for a more accurate assessment of how the model will perform in real-world applications.",
    'chapter_information': 'midterm 1'
}

check_1_q18 = {
    'question': "Which of the following best describes the concept of classification in analytics?",
    'options_list': [
        'A. Identifying the most common category among neighbors.',
        'B. Separating different categories with maximum margin.',
        'C. Assessing the cost of different types of errors.',
        'D. Measuring distances between data points.'
    ],
    'correct_answer': 'B. Separating different categories with maximum margin.',
    'explanation': "Classification involves assigning data points to predefined categories. A common approach is to separate categories with a decision boundary, typically maximizing the margin between them to reduce misclassification.",
    'chapter_information': 'midterm 1'
}

check_1_q19 = {
    'question': "What is the main iterative process in the K-means clustering algorithm?",
    'options_list': [
        'A. Calculating distances between data points.',
        'B. Assigning data points to the nearest cluster center.',
        'C. Calculating p-norms for distance measurements.',
        'D. Creating Voronoi diagrams.'
    ],
    'correct_answer': 'B. Assigning data points to the nearest cluster center.',
    'explanation': "The K-means algorithm involves iteratively assigning data points to the nearest cluster center based on calculated distances, and then recalculating the cluster centers based on the new assignments.",
    'chapter_information': 'midterm 1'
}

check_1_q20 = {
    'question': "What is the purpose of the 'elbow diagram' in K-means clustering?",
    'options_list': [
        'A. To show the shape of the clusters.',
        'B. To visualize the data distribution.',
        'C. To decide the number of clusters.',
        'D. To identify outliers in the data.'
    ],
    'correct_answer': 'C. To decide the number of clusters.',
    'explanation': "The elbow diagram in K-means clustering is used to determine the optimal number of clusters. It plots the within-cluster sum of squares against the number of clusters, with the 'elbow' indicating the point where adding more clusters yields diminishing returns.",
    'chapter_information': 'midterm 1'
}

check_1_q21 = {
    'question': "What are the different types of outliers?",
    'options_list': [
        'A. Single outliers and double outliers.',
        'B. Point outliers, contextual outliers, and collective outliers.',
        'C. Inliers and outliers.',
        'D. Clean data and noisy data.'
    ],
    'correct_answer': 'B. Point outliers, contextual outliers, and collective outliers.',
    'explanation': "Outliers can be classified into three types: point outliers (single unusual data points), contextual outliers (outliers within a specific context or condition), and collective outliers (anomalies within a group of data points).",
    'chapter_information': 'midterm 1'
}

check_1_q22 = {
    'question': "What is the primary purpose of change detection in data analysis?",
    'options_list': [
        'A. Identifying historical data patterns',
        'B. Detecting changes in data over time',
        'C. Smoothing out data fluctuations',
        'D. Estimating baseline values'
    ],
    'correct_answer': 'B. Detecting changes in data over time',
    'explanation': "Change detection focuses on identifying significant shifts or variations in data over time, helping to recognize trends, patterns, or anomalies in a dataset.",
    'chapter_information': 'midterm 1'
}

check_1_q23 = {
    'question': "What do different parameter values in CUSUM affect?",
    'options_list': [
        'A. The nature of the data',
        'B. The sensitivity of change detection',
        'C. The data collection process',
        'D. The data visualization'
    ],
    'correct_answer': 'B. The sensitivity of change detection',
    'explanation': "CUSUM parameters determine how sensitive the model is to changes in data. Adjusting these parameters affects how quickly and accurately the model detects changes.",
    'chapter_information': 'midterm 1'
}

check_1_q24 = {
    'question': "What does exponential smoothing primarily help in determining when analyzing time series data?",
    'options_list': [
        'A. The latest observation',
        'B. Baseline data value',
        'C. Seasonality factors',
        'D. Random fluctuations'
    ],
    'correct_answer': 'D. Random fluctuations',
    'explanation': "Exponential smoothing is used to smooth out random fluctuations in time series data, making it easier to identify trends, seasonality, and other underlying patterns.",
    'chapter_information': 'midterm 1'
}

check_1_q25 = {
    'question': "In exponential smoothing, what does the coefficient alpha balance in the model?",
    'options_list': [
        'A. Baseline and trend estimates',
        'B. Latest observation and previous estimate',
        'C. Seasonality and cyclical effects',
        'D. Forecasting errors'
    ],
    'correct_answer': 'B. Latest observation and previous estimate',
    'explanation': "In exponential smoothing, the coefficient alpha controls the weighting between the latest observation and the previous estimate. A higher alpha gives more weight to recent data, while a lower alpha gives more weight to historical trends.",
    'chapter_information': 'midterm 1'
}

check_1_q26 = {
    'question': "What is the primary purpose of the Box-Cox transformation in data preparation?",
    'options_list': [
        'A. Reduce dimensionality of data',
        'B. Normalize data distributions',
        'C. Remove outliers from data',
        'D. Adjust data for seasonal effects'
    ],
    'correct_answer': 'B. Normalize data distributions',
    'explanation': "The Box-Cox transformation is used to normalize data distributions, allowing for more effective analysis and meeting the assumptions of parametric statistical tests.",
    'chapter_information': 'midterm 1'
}

check_1_q27 = {
    'question': "What is the main objective of Principal Component Analysis (PCA)?",
    'options_list': [
        'A. Increase the number of predictors',
        'B. Remove all variance in data',
        'C. Simplify complex datasets',
        'D. Introduce correlation among variables'
    ],
    'correct_answer': 'C. Simplify complex datasets',
    'explanation': "PCA aims to reduce the dimensionality of complex datasets by identifying key components that capture most of the variance, thus simplifying the dataset while retaining essential information.",
    'chapter_information': 'midterm 1'
}

check_1_q28 = {
    'question': "In Principal Component Analysis (PCA), what do eigenvalues and eigenvectors determine?",
    'options_list': [
        'A. The number of principal components',
        'B. The orthogonality of data',
        'C. The transformation of data',
        'D. The correlation among variables'
    ],
    'correct_answer': 'A. The number of principal components',
    'explanation': "Eigenvalues and eigenvectors in PCA help determine the number of principal components by indicating the amount of variance each component captures. The number of principal components to retain is often based on the eigenvalues.",
    'chapter_information': 'midterm 1'
}

check_1_q29 = {
    'question': "What is the primary advantage of Classification and Regression Trees (CART) in regression analysis?",
    'options_list': [
        'A. They use a single regression model for all data points.',
        'B. They allow the use of separate coefficients for different data segments.',
        'C. They are less prone to overfitting compared to other methods.',
        'D. They do not require pruning.'
    ],
    'correct_answer': 'B. They allow the use of separate coefficients for different data segments.',
    'explanation': "CART allows the creation of different regression models for different data segments, providing flexibility and improving accuracy for diverse data subsets.",
    'chapter_information': 'midterm 1'
}

check_1_q30 = {
    'question': "In logistic regression, what is the range of predicted probabilities?",
    'options_list': [
        'A. -1 to 1',
        'B. 0 to 1',
        'C. 0 to infinity',
        'D. -infinity to infinity'
    ],
    'correct_answer': 'B. 0 to 1',
    'explanation': "Logistic regression outputs probabilities ranging from 0 to 1, indicating the likelihood of a binary outcome. The logistic function maps values to this range, ensuring valid probability predictions.",
    'chapter_information': 'midterm 1'
}

check_1_q31 = {
    'question': " (GPT) In the context of data analytics, what is the 'curse of dimensionality'?",
    'options_list': [
        'A. As the number of dimensions increases, the volume of the space increases exponentially, making the data sparse.',
        'B. High-dimensional data always leads to better model performance.',
        'C. Increasing dimensions reduces computational complexity.',
        'D. It refers to the overfitting problem when the number of observations is larger than the number of features.'
    ],
    'correct_answer': 'A. As the number of dimensions increases, the volume of the space increases exponentially, making the data sparse.',
    'explanation': "The 'curse of dimensionality' describes how as the number of features (dimensions) increases, the data becomes sparse, and algorithms struggle to find patterns, leading to decreased performance.",
    'chapter_information': 'midterm 1'
}

check_1_q32 = {
    'question': " (GPT) Which of the following is true regarding overfitting in machine learning models?",
    'options_list': [
        'A. Overfitting occurs when a model performs well on both training and unseen data.',
        'B. Regularization techniques can help prevent overfitting by adding a penalty term to the loss function.',
        'C. Overfitting can be completely eliminated by increasing the complexity of the model.',
        'D. Cross-validation cannot detect overfitting.'
    ],
    'correct_answer': 'B. Regularization techniques can help prevent overfitting by adding a penalty term to the loss function.',
    'explanation': "Regularization adds a penalty for complexity to the loss function, discouraging overly complex models that might overfit the training data.",
    'chapter_information': 'midterm 1'
}

check_1_q34 = {
    'question': " (GPT) Which of the following statements about Principal Component Analysis (PCA) is TRUE?",
    'options_list': [
        'A. PCA maximizes the variance along the new axes.',
        'B. PCA can only be applied to categorical data.',
        'C. PCA increases the dimensionality of the dataset.',
        'D. The principal components are always aligned with the original features.'
    ],
    'correct_answer': 'A. PCA maximizes the variance along the new axes.',
    'explanation': "PCA seeks to find the directions (principal components) that maximize the variance in the data, thereby reducing dimensionality while retaining the most important information.",
    'chapter_information': 'midterm 1'
}

check_1_q35 = {
    'question': " (GPT) In the context of time series forecasting, what is 'stationarity'?",
    'options_list': [
        'A. A time series with constant mean and variance over time.',
        'B. A time series that has no seasonal patterns.',
        'C. A time series that shows a linear trend over time.',
        'D. A time series that is unaffected by external factors.'
    ],
    'correct_answer': 'A. A time series with constant mean and variance over time.',
    'explanation': "Stationarity refers to statistical properties of a time series being constant over time, meaning the mean, variance, and autocorrelation structure do not change.",
    'chapter_information': 'midterm 1'
}

check_1_q36 = {
    'question': " (GPT) Which of the following methods can be used to transform a non-stationary time series into a stationary one?",
    'options_list': [
        'A. Differencing the data.',
        'B. Applying exponential smoothing.',
        'C. Calculating the moving average.',
        'D. Using higher-degree polynomial fitting.'
    ],
    'correct_answer': 'A. Differencing the data.',
    'explanation': "Differencing is a common technique to stabilize the mean of a time series by removing changes in the level of a time series, and thus eliminating (or reducing) trend and seasonality.",
    'chapter_information': 'midterm 1'
}

check_1_q37 = {
    'question': " (GPT) Which of the following is NOT an assumption of linear regression?",
    'options_list': [
        'A. Linearity between the independent and dependent variables.',
        'B. Homoscedasticity of residuals.',
        'C. Multicollinearity among independent variables.',
        'D. Independence of errors.'
    ],
    'correct_answer': 'C. Multicollinearity among independent variables.',
    'explanation': "Linear regression assumes that there is no perfect multicollinearity among independent variables. Multicollinearity violates the assumption that the predictors are independent.",
    'chapter_information': 'midterm 1'
}

check_1_q38 = {
    'question': " (GPT) Which evaluation metric is most appropriate for imbalanced classification problems?",
    'options_list': [
        'A. Accuracy',
        'B. Precision',
        'C. Recall',
        'D. F1 Score'
    ],
    'correct_answer': 'D. F1 Score',
    'explanation': "The F1 score balances precision and recall and is a better metric than accuracy for imbalanced datasets where the minority class is more important.",
    'chapter_information': 'midterm 1'
}

check_1_q39 = {
    'question': " (GPT) In random forest models, how is 'bagging' used to improve model performance?",
    'options_list': [
        'A. By reducing variance through averaging multiple decision trees.',
        'B. By reducing bias through increasing model complexity.',
        'C. By selecting the best-performing tree from an ensemble.',
        'D. By pruning trees to prevent overfitting.'
    ],
    'correct_answer': 'A. By reducing variance through averaging multiple decision trees.',
    'explanation': "Bagging (Bootstrap Aggregating) reduces variance by training multiple models on different subsets of data and averaging their predictions, which is fundamental to random forests.",
    'chapter_information': 'midterm 1'
}

check_1_q40 = {
    'question': " (GPT) Which of the following is a disadvantage of using k-nearest neighbors (k-NN) algorithm?",
    'options_list': [
        'A. It requires a lot of storage space.',
        'B. It is not sensitive to irrelevant features.',
        'C. It can model complex nonlinear relationships.',
        'D. It makes strong assumptions about the underlying data distribution.'
    ],
    'correct_answer': 'A. It requires a lot of storage space.',
    'explanation': "k-NN is a lazy learning algorithm that stores all training data, which can be computationally expensive in terms of memory and processing time, especially with large datasets.",
    'chapter_information': 'midterm 1'
}


check_2_q1tt = {
    'question': " (GPT) What is the purpose of the 'validation set' in a machine learning model?",
    'options_list': [
        'A. To optimize hyperparameters',
        'B. To improve training data accuracy',
        'C. To provide an unbiased evaluation during training',
        'D. To select the best features in a model'
    ],
    'correct_answer': 'C. To provide an unbiased evaluation during training',
    'explanation': "The validation set is used during training to provide an unbiased evaluation of a model's performance. It's crucial for tuning hyperparameters without affecting the training process.",
    'chapter_information': 'midterm 2'
}

check_2_q2tt = {
    'question': "(GPT) In ridge regression, increasing the penalty term λ results in:",
    'options_list': [
        'A. Increasing the complexity of the model',
        'B. Decreasing the complexity of the model',
        'C. Increasing the variance of the model',
        'D. No impact on the model’s complexity'
    ],
    'correct_answer': 'B. Decreasing the complexity of the model',
    'explanation': "Ridge regression introduces a penalty term that controls the model's complexity by shrinking the coefficients. Increasing λ leads to a simpler, less complex model with lower variance.",
    'chapter_information': 'midterm 2'
}

check_2_q3tt = {
    'question': "(GPT) Which of the following scenarios would benefit from using a support vector machine (SVM)?",
    'options_list': [
        'A. High-dimensional data with clear separation between classes',
        'B. Data with a significant number of outliers',
        'C. Unlabeled data requiring clustering',
        'D. Time series data with long-term dependencies'
    ],
    'correct_answer': 'A. High-dimensional data with clear separation between classes',
    'explanation': "SVMs are effective in handling high-dimensional data and work well when there is a clear margin of separation between classes, using hyperplanes to maximize the margin.",
    'chapter_information': 'midterm 1'
}


check_2_q7tt = {
    'question': " (GPT) True or False: Lasso regression can set some feature coefficients to zero, effectively selecting features.",
    'options_list': [
        'True',
        'False'
    ],
            'correct_answer': 'True',
    'explanation': "Lasso regression uses L1 regularization, which can shrink some coefficients to zero, effectively selecting important features and excluding irrelevant ones.",
    'chapter_information': 'midterm 1'
}

check_2_q8tt = {
    'question': "(GPT) True or False: Cross-validation is only necessary for small datasets.",
    'options_list': [
        'True',
        'False'
    ],
            'correct_answer': 'False',
    'explanation': "Cross-validation is beneficial for all dataset sizes, as it helps ensure the model generalizes well to unseen data by reducing the risk of overfitting, regardless of dataset size.",
    'chapter_information': 'midterm 1'
}




KC_MPC_QUESTIONS = []
global_items = list(globals().items())
# print(global_items)

for name, value in global_items:
    if not name.startswith('_'):
        KC_MPC_QUESTIONS.append(value)

MIDTERM1_MPC_QUESTIONS = KC_MPC_QUESTIONS[:-1]
