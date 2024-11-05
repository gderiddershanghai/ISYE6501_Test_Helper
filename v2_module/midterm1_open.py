

midterm_1_q18 = {
    'question': (
        "A city's transportation department has collected data on public transit usage, including passenger demographics, travel patterns, peak times, fare categories, and service delays. The department wants to use this data to predict future service delays and identify the factors that contribute to frequent delays."
        "\n\n"
        "Choose the appropriate models/approaches from the list below that the transportation department could use for predicting service delays and understanding the underlying factors."
        "\n"
        "- CUSUM\n"
        "- K-nearest-neighbor classification\n"
        "- Logistic Regression\n"
        "- Multi-armed bandit\n"
        "- Support Vector Machine\n"
    ),
    'correct_answer': (
        "- Logistic Regression\n"
        "- K-nearest-neighbor classification\n"
        "- Support Vector Machine"
    ),
    'explanation': (
        "Logistic Regression is suitable for predicting service delays as it can classify outcomes based on binary data, helping to identify key factors contributing to delays."
        "\n\n"
        "K-nearest-neighbor classification can be used to predict future delays by comparing similar past travel patterns, helping to understand delay trends."
        "\n\n"
        "Support Vector Machine is appropriate for binary classification tasks and provides strong decision boundaries, making it useful for predicting service delays."
    )
}

midterm_1_q18 = {
    'question': (
        "A regional healthcare provider has collected extensive data on patient visits over the years, including patient demographics, symptoms, diagnoses, treatments, and outcomes. The organization now wants to leverage this data to predict patient readmission risks and identify key factors that contribute to higher readmission rates."
        "\n\n"
        "Choose the appropriate models/approaches from the list below that the healthcare provider could use for predicting patient readmissions and understanding the underlying factors."
        "\n"
        "- CUSUM\n"
        "- K-nearest-neighbor classification\n"
        "- Logistic Regression\n"
        "- Multi-armed bandit\n"
        "- Support Vector Machine\n"
    ),
    'correct_answer': (
        "- Logistic Regression\n"
        "- K-nearest-neighbor classification\n"
        "- Support Vector Machine"
    ),
    'explanation': (
        "Logistic Regression is suitable for predicting patient readmission risks due to its binary classification approach. It helps identify key factors contributing to higher readmission rates."
        "\n\n"
        "K-nearest-neighbor classification can be used to predict patient readmissions by finding the closest matches in the dataset, aiding in understanding patterns."
        "\n\n"
        "Support Vector Machine is also suitable for binary classification tasks, providing robust decision boundaries for predicting patient readmission risks."
    )
}

midterm_1_q16 = {
    'question': (
        "Question A1: A company has noticed an increasing trend in customer service calls on Mondays over the past 15 years. The company wants to analyze whether there has been a significant change in this Monday trend in customer service calls during this period. Select all of the approaches that might reasonably be correct. "
        "i. Develop 15 separate logistic regression models, one for each year, with 'is it Monday?' as one of the predictor variables; then apply a CUSUM analysis on the yearly coefficients for the Monday variable. "
        "ii. Implement time series forecasting using ARIMA, focusing on Mondays for the 780 weeks, and then use CUSUM on the forecasted values to identify any significant shifts. "
        "iii. Apply CUSUM directly on the volume of customer service calls received each of the 780 Mondays over the past 15 years."
    ),
    'correct_answer': "ii, iii",
    'explanation': (
        "ii. Using time series forecasting with ARIMA allows for modeling the trend and seasonality in the data, while CUSUM can be applied to detect shifts in the forecasted trend, making this approach suitable for analyzing changes in the pattern. "
        "iii. Applying CUSUM directly on the call volume for each Monday provides a direct way to observe shifts in the data, especially over a long-term period like 15 years."
    )
}


midterm_1_q13 = {
    'question': (
        "Confusion Matrix for Drug Detection Model (Drug-Sniffing Dog at Airport):"
        "\n======================================================================="
        "\n                        Predicted No Drugs   Predicted Drugs"
        "\nActual No Drugs               1000                 200"
        "\nActual Drugs                    100                 300"
        "\n======================================================================="
        "\n\n"
        "This confusion matrix represents the outcomes of a drug detection model where a trained dog predicts whether an individual is carrying drugs ('Predicted Drugs') or not ('Predicted No Drugs'). The results are compared against the actual occurrences ('Actual Drugs' and 'Actual No Drugs')."
        "\n\n"
        "Questions about the Drug Detection Model's Confusion Matrix:"
        "\n\n"
        "Question A1:\n"
        "Calculate the model's accuracy (the proportion of true results among the total number of cases examined)."
        "\n"
        "A) (1000 + 300) / (1000 + 200 + 100 + 300)"
        "\nB) (1000 + 100) / (1000 + 200 + 100 + 300)"
        "\nC) (200 + 300) / (1000 + 200 + 100 + 300)\n\n"
        "Question A2:\n"
        "Determine the model's precision for drug detection (the proportion of correctly predicted drug cases to the total predicted as carrying drugs)."
        "\n"
        "A) 300 / (200 + 300)"
        "\nB) 1000 / (1000 + 100)"
        "\nC) 300 / (1000 + 300)\n\n"
        "Question A3:\n"
        "Calculate the model's recall for drug detection (the ability of the model to identify actual drug carriers)."
        "\n"
        "A) 300 / (100 + 300)"
        "\nB) 200 / (1000 + 200)"
        "\nC) 1000 / (1000 + 100)\n\n"
        "Question A4:\n"
        "Based on the confusion matrix, which statement is true regarding the model's predictions?"
        "\n"
        "A) The model is more accurate in predicting non-drug carriers than drug carriers."
        "\nB) The model has the same accuracy for predicting drug carriers and non-drug carriers."
        "\nC) The model is more accurate in predicting drug carriers than non-drug carriers."
    ),
    'correct_answer': (
        "A1: A) (1000 + 300) / (1000 + 200 + 100 + 300)\n\n"
        "A2: A) 300 / (200 + 300)\n\n"
        "A3: A) 300 / (100 + 300)\n\n"
        "A4: A) The model is more accurate in predicting non-drug carriers than drug carriers."
    ),
    'explanation': (
        "A1: The model's accuracy is the proportion of true results among the total cases, which is calculated as (1000 + 300) / (1000 + 200 + 100 + 300), yielding 1300/1600, or 81.25%."
        "\n\n"
        "A2: Precision for drug detection is the proportion of correctly predicted drug carriers to the total predicted as carrying drugs, which is calculated as 300 / (200 + 300), resulting in 300/500, or 60%."
        "\n\n"
        "A3: Recall for drug detection is the ability of the model to identify actual drug carriers, calculated as 300 / (100 + 300), resulting in 300/400, or 75%."
        "\n\n"
        "A4: The model is more accurate in predicting non-drug carriers, as it correctly identifies 1000 out of 1200 cases, giving a higher accuracy compared to drug detection predictions."
    )
}


midterm_1_q14 = {
    'question': (
        "A1\n"
        "Matching\n"
        "Choices:\n"
        "\nA. Classification"
        "\nB. Clustering"
        "\nC. Dimensionality Reduction"
        "\nD. Outlier Detection\n\n"
        "A1. Astronomers have a collection of long-exposure CCD images of distant celestial objects. They are unsure about the types of these objects and seek to group similar ones together. Which method is more suitable?"
        "\n\n"
        "A2. An astronomer has manually categorized hundreds of images and now wishes to use analytics to automatically categorize new images. Which approach is most fitting?"
        "\n\n"
        "A3. A data scientist wants to reduce the complexity of a high-dimensional dataset to visualize it more effectively, while preserving as much information as possible. Which technique should they use?"
        "\n\n"
        "A4. A financial analyst is examining a large set of transaction data to identify unusual transactions that might indicate fraudulent activity. Which method is most appropriate?"
    ),
    'correct_answer': (
        "A1: B. Clustering\n\n"
        "A2: A. Classification\n\n"
        "A3: C. Dimensionality Reduction\n\n"
        "A4: D. Outlier Detection"
    ),
    'explanation': (
        "A1: Clustering is suitable for grouping similar objects, making it ideal for astronomers seeking to group celestial objects based on their characteristics."
        "\n\n"
        "A2: Classification is appropriate for automatically categorizing new images based on manually categorized examples."
        "\n\n"
        "A3: Dimensionality Reduction helps reduce the complexity of a high-dimensional dataset for visualization, preserving key information."
        "\n\n"
        "A4: Outlier Detection is designed to identify unusual data points or anomalies, which is ideal for identifying potential fraudulent activity in transaction data."
    )
}


midterm_1_q10 = {
    'question': (
        "Model Suitability Analysis\n\n"
        "For each statistical and machine learning model listed below, select the type of analysis it is best suited for."
        " There may be more than one correct answer for each model, but you need only choose one. Assume 1 of 1 attempt for each part."
        "\n\n"
        "A1. Ridge Regression"
        "\n- Predicting a continuous response variable with feature data."
        "\n- Dealing with multicollinearity in regression analysis."
        "\n- Forecasting future values in a time-series dataset."
        "\n- Classifying binary outcomes."
        "\n\n"
        "A2. Lasso Regression"
        "\n- Selecting important features in a large dataset."
        "\n- Predicting a numerical outcome based on feature data."
        "\n- Analyzing patterns in time-series data."
        "\n- Identifying categories in unstructured data."
        "\n\n"
        "A3. Principal Component Analysis (PCA)"
        "\n- Reducing the dimensionality of a large dataset."
        "\n- Forecasting trends in a time-series dataset."
        "\n- Classifying items into categories based on feature data."
        "\n- Detecting changes in the variance of a dataset over time."
    ),
    'correct_answer': (
        "A1: Dealing with multicollinearity in regression analysis\n\n"
        "A2: Selecting important features in a large dataset\n\n"
        "A3: Reducing the dimensionality of a large dataset"
    ),
    'explanation': (
        "A1: Ridge Regression is best suited for dealing with multicollinearity in regression analysis because it adds a regularization term that reduces the impact of correlated variables."
        "\n\n"
        "A2: Lasso Regression is ideal for selecting important features in a large dataset because it adds a regularization term that can shrink some coefficients to zero, effectively removing less important features."
        "\n\n"
        "A3: Principal Component Analysis (PCA) is designed to reduce dimensionality by transforming the original dataset into a smaller set of uncorrelated variables, retaining the most information."
    )
}

midterm_1_q11 = {
    'question': (
        "Model Suitability Analysis\n\n"
        "For each statistical and machine learning model listed below, select the type of analysis it is best suited for."
        " There may be more than one correct answer for each model, but you need only choose one. Assume 1 of 1 attempt for each part."
        "\n\n"
        "A1. Decision Trees (e.g., CART)"
        "\n- Predicting the category of an item based on feature data."
        "\n- Forecasting numerical values two time periods in the future."
        "\n- Identifying clusters in feature data."
        "\n- Analyzing the variance in time-series data."
        "\n\n"
        "A2. Random Forest"
        "\n- Predicting the likelihood of an event based on feature data."
        "\n- Classifying items into categories based on feature data."
        "\n- Estimating the amount of a variable two time periods in the future using time-series data."
        "\n- Detecting patterns in large datasets with many variables."
        "\n\n"
        "A3. Naive Bayes Classifier"
        "\n- Classifying text data into predefined categories."
        "\n- Predicting future trends based on historical time-series data."
        "\n- Estimating the probability of an event occurring in the future."
        "\n- Analyzing variance in feature data."
    ),
    'correct_answer': (
        "A1: Predicting the category of an item based on feature data\n\n"
        "A2: Detecting patterns in large datasets with many variables\n\n"
        "A3: Classifying text data into predefined categories"
    ),
    'explanation': (
        "A1: Decision Trees are well-suited for predicting the category of an item based on feature data, providing a clear visual representation of the decision-making process."
        "\n\n"
        "A2: Random Forest is designed to detect patterns in large datasets with many variables, using an ensemble of decision trees to improve prediction accuracy."
        "\n\n"
        "A3: Naive Bayes Classifier is optimal for classifying text data into predefined categories, given its probabilistic approach that assumes independence among features."
    )
}

midterm_1_q12 = {
    'question': (
        "A1\n"
        "Select all of the following reasons that data should not be scaled until point outliers are removed:"
        "\n"
        "- If data is scaled first, the range of data after outliers are removed will be narrower than intended."
        "\n- If data is scaled first, the range of data after outliers are removed will be wider than intended."
        "\n- Point outliers would appear to be valid data if not removed before scaling."
        "\n- Valid data would appear to be outliers if data is scaled first.\n\n"
        "A2\n"
        "Select all of the following situations in which using a variable selection approach like lasso or stepwise regression would be important:"
        "\n"
        "- It is too costly to create a model with a large number of variables."
        "\n- There are too few data points to avoid overfitting if all variables are included."
        "\n- Time-series data is being used."
        "\n- There are fewer data points than variables."
    ),
    'correct_answer': (
        "A1: If data is scaled first, the range of data after outliers are removed will be narrower than intended"
        "\n\n"
        "A2: It is too costly to create a model with a large number of variables"
        "\n- There are too few data points to avoid overfitting if all variables are included"
    ),
    'explanation': (
        "A1: If data is scaled first, the range of data after outliers are removed will be narrower than intended, affecting the scaling process."
        "\n\n"
        "A2: Using variable selection approaches like lasso or stepwise regression is important when it is too costly to create a model with many variables, or when there are too few data points to avoid overfitting."
    )
}


midterm_1_q9 = {
    'question': (
        "Model Suitability Analysis\n\n"
        "For each statistical and machine learning model listed below, select the type of analysis it is best suited for. "
        "There may be more than one correct answer for each model, but you need only choose one. Assume 1 of 1 attempt for each part."
        "\n\n"
        "A1. Time Series Analysis (e.g., ARMA, ARIMA)"
        "\n- Predicting future values in a time-series dataset."
        "\n- Classifying items based on time-dependent features."
        "\n- Analyzing the seasonal components of time-series data."
        "\n- Estimating the probability of an event occurring in the future."
        "\n\n"
        "A2. k-Nearest-Neighbor Classification (kNN)"
        "\n- Using feature data to predict whether or not something will happen two time periods in the future."
        "\n- Using feature data to predict the probability of something happening two time periods in the future."
        "\n- Using time-series data to predict the amount of something two time periods in the future."
        "\n- Using time-series data to predict the variance of something two time periods in the future."
        "\n\n"
        "A3. Exponential Smoothing"
        "\n- Using time-series data to predict the amount of something two time periods in the future."
        "\n- Analyzing the seasonal components of time-series data."
        "\n- Using time-series data to predict future trends."
    ),
    'correct_answer': (
        "A1: Predicting future values in a time-series dataset\n\n"
        "A2: Using feature data to predict whether or not something will happen two time periods in the future\n\n"
        "A3: Using time-series data to predict the amount of something two time periods in the future"
    ),
    'explanation': (
        "A1: Time Series Analysis, such as ARIMA, is most suitable for predicting future values in a time-series dataset, providing forecasting capabilities with seasonal components."
        "\n\n"
        "A2: k-Nearest-Neighbor Classification (kNN) is best for using feature data to predict binary outcomes, as it relies on the proximity to known data points to classify new data."
        "\n\n"
        "A3: Exponential Smoothing is appropriate for predicting the amount of something in a time-series dataset, offering a smoothing approach to forecast trends and address variability."
    )
}

midterm_1_q8 = {
    'question': (
        "For each scenario, identify the most relevant statistical measure: AIC (Akaike Information Criterion), R-squared, Specificity, or Variance. Variance is included as a distractor and may not be the correct answer."
        "\n\n"
        "Definitions:"
        "\nAIC (Akaike Information Criterion): Balances the model's fit with the complexity by penalizing the number of parameters."
        "\nR-squared: Measures the proportion of variance in the dependent variable explained by the independent variables."
        "\nSpecificity: Not relevant in this context."
        "\nVariance: Measures the dispersion of a set of data points."
        "\n\n"
        "Choices:"
        "\nA. AIC"
        "\nB. R-squared"
        "\nC. Specificity"
        "\nD. Variance"
        "\n\n"
        "Scenarios:"
        "\n\n"
        "Question A1:"
        "A researcher is assessing various linear regression models to predict future profits of a company, aiming to find a balance between model complexity and fit."
        "\n\n"
        "Question A2:"
        "In a study evaluating the effect of advertising on sales, the analyst seeks to understand how changes in advertising budgets correlate with variations in sales figures."
        "\n\n"
        "Question A3:"
        "An economist is choosing among different models to forecast economic growth, with a focus on avoiding overfitting in the presence of many potential predictor variables."
    ),
    'correct_answer': (
        "A1: AIC\n\n"
        "A2: R-squared\n\n"
        "A3: AIC"
    ),
    'explanation': (
        "A1: AIC is the most relevant measure for assessing various linear regression models, as it balances model fit with complexity, helping prevent overfitting."
        "\n\n"
        "A2: R-squared is the appropriate measure for evaluating the effect of advertising on sales, indicating the proportion of variance in the dependent variable explained by the independent variables."
        "\n\n"
        "A3: AIC is suitable for selecting models that avoid overfitting when forecasting economic growth, providing a balance between model complexity and fit."
    )
}

midterm_1_q7 = {
    'question': (
        "Select all of the following statements that are correct:"
        "\n\n"
        "- It is likely that the first principal component has much more predictive power than each of the other principal components."
        "\n- It is likely that the first original covariate has much more predictive power than each of the other covariates."
        "\n- It is likely that the last original covariate has much less predictive power than each of the other covariates."
        "\n- The first principal component cannot contain information from all 7 original covariates. (correct)"
    ),
    'correct_answer': (
        "- The first principal component cannot contain information from all 7 original covariates."
    ),
    'explanation': (
        "The first principal component (PCA) captures the most variance in the dataset but cannot contain all the information from the original covariates. It represents a linear combination of original variables, but not the complete set."
    )
}


midterm_1_q5 = {
    'question': (
        "An airline wants to predict airline passenger traffic for the upcoming year."
        " For each of the specific questions (a-e) listed below, identify the question type (i-viii) it corresponds to."
        " If a question does not match any of the listed types, leave it uncircled."
        "\n\n"
        "Question Types:"
        "\n"
        "i. Change detection"
        "\nii. Classification"
        "\niii. Clustering"
        "\niv. Feature-based prediction of a value"
        "\nv. Feature-based prediction of a probability"
        "\nvi. Time-series-based prediction"
        "\nvii. Validation"
        "\nviii. Variance estimation"
        "\n\n"
        "Questions:"
        "\na. What is the probability that the airline will exceed 1 million passengers next year, considering current travel trends and economic factors?"
        "\nb. Among various forecasting models for airline passenger traffic, which one is likely to be the most accurate for the upcoming year?"
        "\nc. Based on the past decade's data, how many passengers are expected to travel via the airline next year?"
        "\nd. Analyzing the past fifteen years of data, has there been a significant change in passenger traffic during holiday seasons?"
        "\ne. Considering economic indicators and travel trends over the past 25 years, which years had the most similar passenger traffic patterns?"
    ),
    'correct_answer': (
        "a: v. Feature-based prediction of a probability\n\n"
        "b: vii. Validation\n\n"
        "c: vi. Time-series-based prediction\n\n"
        "d: i. Change detection\n\n"
        "e: iii. Clustering"
    ),
    'explanation': (
        "a: This question involves predicting the probability of an event occurring, aligning with 'Feature-based prediction of a probability.'"
        "\n\n"
        "b: This question seeks the most accurate forecasting model, relating to 'Validation.'"
        "\n\n"
        "c: This question requires predicting a future value based on time-series data, which is 'Time-series-based prediction.'"
        "\n\n"
        "d: Analyzing significant changes over time corresponds to 'Change detection.'"
        "\n\n"
        "e: Identifying years with similar passenger traffic patterns corresponds to 'Clustering.'"
    )
}


midterm_1_q4 = {
    'question': (
        "Select whether a supervised learning model (like regression) is more directly appropriate than an unsupervised learning model (like dimensionality reduction)."
        "\n\n"
        "Definitions:"
        "\nSupervised Learning: Machine learning where the 'correct' answer or outcome is known for each data point in the training set."
        "\nRegression: A type of supervised learning where the model predicts a continuous outcome."
        "\nUnsupervised Learning Model: Machine learning where the 'correct' answer is not known for the data points in the training set."
        "\nDimensionality Reduction: A process in unsupervised learning of reducing the number of random variables under consideration, through feature selection and feature extraction."
        "\n\n"
        "Questions:"
        "\n\n"
        "- In a dataset of residential property sales, for each property, the sale price is known, and the goal is to predict prices for new listings based on various attributes like location, size, and amenities."
        "\n\n"
        "- In a large dataset of customer reviews, there is no specific response variable, but the goal is to understand underlying themes and patterns in the text data."
        "\n\n"
        "- In a clinical trial dataset, for each participant, the response to a medication is known, and the task is to predict patient outcomes based on their medical history and trial data."
    ),
    'correct_answer': (
        "- In a dataset of residential property sales, for each property, the sale price is known, and the goal is to predict prices for new listings based on various attributes like location, size, and amenities."
        "\n- In a clinical trial dataset, for each participant, the response to a medication is known, and the task is to predict patient outcomes based on their medical history and trial data."
    ),
    'explanation': (
        "Supervised learning is appropriate when the 'correct' outcome is known, such as predicting property sale prices based on known values and forecasting patient outcomes based on medical history."
        "\n\n"
        "Unsupervised learning is appropriate when the 'correct' outcome is unknown, such as identifying themes and patterns in text data."
    )
}

midterm11_1_q1 = {
    'question': (
        "For each of the models (a-m) below, circle one type of question (i-viii) it is commonly used for."
        " For models that have more than one correct answer, choose any one correct answer."
        " If there is no correct answer listed, do not circle anything."
        "\n\n"
        "Models:"
        "\na. ARIMA"
        "\nb. CART"
        "\nc. Cross validation"
        "\nd. CUSUM"
        "\ne. Exponential smoothing"
        "\nf. GARCH"
        "\ng. k-means"
        "\nh. k-nearest-neighbor"
        "\ni. Linear regression"
        "\nj. Logistic regression"
        "\nk. Principal component analysis"
        "\nl. Random forest"
        "\nm. Support vector machine"
        "\n\n"
        "Question Types:"
        "\ni. Change detection"
        "\nii. Classification"
        "\niii. Clustering"
        "\niv. Feature-based prediction of a probability"
        "\nv. Feature-based prediction of a value"
        "\nvi. Time-series-based prediction"
        "\nvii. Validation"
        "\nviii. Variance estimation"
    ),
    'correct_answer': (
        "a. vi. Time-series-based prediction\n\n"
        "b. ii. Classification\n\n"
        "c. vii. Validation\n\n"
        "d. i. Change detection\n\n"
        "e. vi. Time-series-based prediction\n\n"
        "f. viii. Variance estimation\n\n"
        "g. iii. Clustering\n\n"
        "h. ii. Classification\n\n"
        "i. v. Feature-based prediction of a value\n\n"
        "j. iv. Feature-based prediction of a probability\n\n"
        "k. iii. Clustering\n\n"
        "l. ii. Classification\n\n"
        "m. ii. Classification"
    ),
    'explanation': (
        "a. ARIMA is used for time-series-based prediction, analyzing data with trends and seasonality."
        "\n\n"
        "b. CART (Classification and Regression Trees) is commonly used for classification, creating decision trees to separate data into distinct groups."
        "\n\n"
        "c. Cross validation is used for validation, ensuring model robustness and preventing overfitting."
        "\n\n"
        "d. CUSUM (Cumulative Sum) is designed for change detection, identifying shifts in data over time."
        "\n\n"
        "e. Exponential smoothing is also used for time-series-based prediction, applying a smoothing factor to forecast trends."
        "\n\n"
        "f. GARCH (Generalized Autoregressive Conditional Heteroskedasticity) is used for variance estimation, modeling volatility in financial data."
        "\n\n"
        "g. k-means is a clustering method, creating groups based on data similarities."
        "\n\n"
        "h. k-nearest-neighbor is used for classification, classifying data based on proximity to other data points."
        "\n\n"
        "i. Linear regression is used for feature-based prediction of a value, predicting a continuous outcome."
        "\n\n"
        "j. Logistic regression is for feature-based prediction of a probability, predicting binary outcomes."
        "\n\n"
        "k. Principal component analysis (PCA) is typically used for clustering, reducing dimensionality and identifying key components."
        "\n\n"
        "l. Random forest is often used for classification, employing an ensemble of decision trees."
        "\n\n"
        "m. Support vector machine (SVM) is used for classification, creating decision boundaries to separate data."
    )
}

midter33m_1_q2 = {
    'question': (
        "Select all of the following models that are designed for use with attribute/feature data (i.e., not time-series data):"
        "\n"
        "- CUSUM"
        "\n- Logistic regression"
        "\n- Support vector machine"
        "\n- GARCH"
        "\n- Random forest"
        "\n- k-means"
        "\n- Linear regression"
        "\n- k-nearest-neighbor"
        "\n- ARIMA"
        "\n- Principal component analysis"
        "\n- Exponential smoothing"
    ),
    'correct_answer': (
        "- CUSUM\n"
        "- Logistic regression\n"
        "- Support vector machine\n"
        "- Random forest\n"
        "- k-means\n"
        "- Linear regression\n"
        "- k-nearest-neighbor\n"
        "- Principal component analysis"
    ),
    'explanation': (
        "These models are designed for use with attribute/feature data, not time-series data:"
        "\n\n"
        "- CUSUM is for change detection in attribute/feature data."
        "\n\n"
        "- Logistic regression, Support vector machine, Random forest, and k-nearest-neighbor are primarily used for classification."
        "\n\n"
        "- Linear regression is for predicting a continuous outcome based on feature data."
        "\n\n"
        "- k-means and Principal component analysis (PCA) are for clustering and dimensionality reduction, respectively."
    )
}


mod55ule_2_q1 = {
    'question': (
        "A financial institution is evaluating loan applications and wants to develop a model to classify applicants as likely to repay or default on a loan."
        "\n\n"
        "a. Given the high cost associated with misclassifying a high-risk applicant as low-risk, which classification model would be most appropriate to maximize the margin between classes and consider the cost of misclassification?"
        "\n"
        "Options:"
        "\n- Logistic Regression"
        "\n- Hard Margin Support Vector Machine (SVM)"
        "\n- Soft Margin Support Vector Machine (SVM)"
        "\n- K-Nearest Neighbor (KNN)\n\n"
        "b. The dataset includes attributes like income, credit score, employment status, and existing debts. Before training the model, why is it important to scale and standardize these attributes, and how might this affect the performance of the SVM model?"
        "\n\n"
        "c. The bank wants to ensure that the model does not overemphasize any single attribute due to differences in scale. Which data preparation technique should be applied, and what is its impact on the coefficients in the classification model?"
        "\n\n"
        "d. After deploying the model, the bank notices that it's still misclassifying some applicants. They decide to adjust the model to allow for some misclassifications in the training data to improve generalization to new applicants. Which SVM approach does this describe, and how does it balance the trade-off between margin size and classification errors?"
    ),
    'correct_answer': (
        "a. Soft Margin Support Vector Machine (SVM)\n\n"
        "b. Scaling and standardizing attributes ensure that all features contribute equally to the model, preventing attributes with larger scales from dominating the decision boundary. This improves the performance of SVM, which is sensitive to the scale of input data.\n\n"
        "c. The bank should apply feature scaling techniques such as standardization (z-score normalization) or min-max scaling. This ensures that all attributes are on a similar scale, which affects the coefficients by giving each attribute an equal opportunity to influence the model.\n\n"
        "d. This describes the Soft Margin SVM approach. It allows some misclassifications to achieve a larger margin, balancing the trade-off by introducing a penalty parameter that controls the margin width and classification errors."
    ),
    'explanation': (
        "a. **Soft Margin SVM** is suitable when data is not perfectly separable, and misclassifications can be tolerated to maximize the margin and improve generalization. It also allows incorporating the cost of misclassification.\n\n"
        "b. **Scaling and standardizing** are crucial because SVM relies on calculating distances between data points. If features are on different scales, attributes with larger scales can disproportionately influence the model.\n\n"
        "c. **Feature scaling techniques** like standardization adjust the data so that each attribute has a mean of zero and a standard deviation of one. This equalizes the influence of all features and stabilizes the coefficients in the model.\n\n"
        "d. **Soft Margin SVM** introduces a penalty parameter (often denoted as C) that allows some data points to be on the wrong side of the margin or even the decision boundary. This helps prevent overfitting and improves the model's ability to generalize to new, unseen data."
    )
}

mo55dule_2_q2 = {
    'question': (
        "An agricultural research institute is developing a system to classify plant species based on leaf measurements. They have collected data on various leaf attributes such as length, width, perimeter, and shape descriptors."
        "\n\n"
        "a. Which classification model is appropriate if the decision boundary between species is non-linear, and why?"
        "\n"
        "Options:"
        "\n- Linear Support Vector Machine (SVM)"
        "\n- SVM with RBF Kernel"
        "\n- Logistic Regression"
        "\n- Decision Tree Classifier\n\n"
        "b. Explain how the use of kernel methods in SVM allows for non-linear classification. Provide an example of a commonly used kernel and its role in transforming the data."
        "\n\n"
        "c. Before training the model, the researchers notice that the scales of the measurements vary significantly. What data preprocessing steps should they take, and how will this impact the SVM model's performance?"
        "\n\n"
        "d. If the researchers decide to use K-Nearest Neighbor (KNN) classification instead, what considerations should they keep in mind regarding the choice of 'k' and distance metrics?"
    ),
    'correct_answer': (
        "a. SVM with RBF Kernel\n\n"
        "b. Kernel methods in SVM allow the algorithm to operate in a transformed feature space where the classes are linearly separable. The Radial Basis Function (RBF) kernel maps the input features into a higher-dimensional space, enabling the SVM to find a non-linear decision boundary in the original feature space.\n\n"
        "c. They should apply feature scaling or normalization to ensure all measurements contribute equally to the model. This is important because SVMs are sensitive to the scale of input features, and unscaled data can lead to poor performance.\n\n"
        "d. When using KNN, they should carefully choose 'k' (the number of neighbors) to balance bias and variance. Additionally, selecting an appropriate distance metric (e.g., Euclidean, Manhattan) is crucial, and they should consider standardizing the data if different features are on different scales."
    ),
    'explanation': (
        "a. **SVM with RBF Kernel** is suitable for non-linear classification tasks because it can model complex decision boundaries by transforming the input space.\n\n"
        "b. **Kernel methods** implicitly map data into higher-dimensional spaces without computing the coordinates in that space explicitly. The **RBF kernel** computes the similarity between two points based on their distance, allowing the SVM to create non-linear boundaries.\n\n"
        "c. **Scaling the data** ensures that no single feature disproportionately influences the model. This preprocessing step is essential for algorithms like SVM that are sensitive to feature scales.\n\n"
        "d. In **KNN classification**, the choice of 'k' affects the model's complexity. A small 'k' can lead to overfitting, while a large 'k' may oversimplify the model. The **distance metric** determines how similarity is measured, impacting the classification results."
    )
}

mod55ule_3_q1 = {
    'question': (
        "A tech company has developed a machine learning model to detect fraudulent transactions. They trained the model on historical transaction data, including both fraudulent and legitimate transactions."
        "\n\n"
        "a. Why is it important for the company to validate their model on a separate dataset rather than relying solely on the training data performance?"
        "\n\n"
        "b. Describe how the company can use a validation set and a test set in their model development process. What is the purpose of each, and how do they differ?"
        "\n\n"
        "c. The dataset is highly imbalanced, with only 1% of transactions being fraudulent. What cross-validation strategy can the company use to ensure that the validation sets adequately represent fraudulent cases?"
        "\n\n"
        "d. Explain the potential risks of not using cross-validation in this scenario, particularly concerning the rare fraudulent transactions."
    ),
    'correct_answer': (
        "a. Validating on a separate dataset is crucial to assess the model's ability to generalize to new, unseen data. Performance on the training data can be overly optimistic due to overfitting, so validation helps estimate true model effectiveness.\n\n"
        "b. The **validation set** is used during model development to tune parameters and select the best model, while the **test set** is used after model selection to assess the final model's performance. The validation set guides model improvement, whereas the test set provides an unbiased evaluation.\n\n"
        "c. They can use **stratified k-fold cross-validation**, which maintains the proportion of fraudulent transactions in each fold. This ensures that each validation set contains a representative sample of both classes.\n\n"
        "d. Without cross-validation, the model might not be exposed to enough fraudulent cases during validation, leading to an unreliable assessment of its ability to detect fraud. This can result in a model that performs poorly in real-world scenarios where detecting fraud is critical."
    ),
    'explanation': (
        "a. **Validation on separate data** prevents misleading conclusions about model performance due to overfitting. It ensures the model's predictive power extends beyond the training data.\n\n"
        "b. The **validation set** is part of the training process, helping to fine-tune the model, while the **test set** evaluates the final model's performance. They serve different purposes in preventing overfitting and assessing generalization.\n\n"
        "c. **Stratified k-fold cross-validation** divides the data so that each fold has the same class proportion as the entire dataset. This is crucial for imbalanced datasets to ensure minority classes are represented.\n\n"
        "d. **Not using cross-validation** risks developing a model that doesn't learn to detect rare events like fraud, as it may not encounter enough examples during training and validation to generalize effectively."
    )
}

modu55le_3_q2 = {
    'question': (
        "An online retailer is testing two different recommendation algorithms to suggest products to customers. They want to determine which algorithm performs better in terms of increasing sales."
        "\n\n"
        "a. Explain how the retailer can use a holdout validation method to compare the two algorithms. What are the steps involved?"
        "\n\n"
        "b. What are some potential drawbacks of using a simple train-test split in this scenario, and how might cross-validation provide a better assessment?"
        "\n\n"
        "c. Suppose the retailer has time-based data where customer behavior changes over seasons. How should they modify their data splitting strategy to account for this, and why?"
        "\n\n"
        "d. After selecting the best algorithm using validation data, why is it important to test its performance on a separate test set before deployment?"
    ),
    'correct_answer': (
        "a. The retailer can split their dataset into a training set and a validation set (holdout set). They train both algorithms on the training set and evaluate their performance on the validation set by measuring sales generated from recommendations.\n\n"
        "b. A simple train-test split may not capture the variability in customer behavior, leading to results that are dependent on the specific split. Cross-validation, especially k-fold cross-validation, provides a more robust assessment by averaging performance over multiple splits.\n\n"
        "c. They should consider **time-based splitting**, such as using earlier data for training and later data for validation and testing. This accounts for temporal changes in customer behavior and ensures the model is evaluated on future data, reflecting real-world deployment.\n\n"
        "d. Testing on a separate test set provides an unbiased evaluation of the algorithm's performance. It ensures that the model's effectiveness is not overestimated due to any potential data leakage or overfitting during the validation phase."
    ),
    'explanation': (
        "a. **Holdout validation** involves setting aside a portion of data for testing after training the model on the rest. Comparing algorithms on the validation set helps select the better-performing one.\n\n"
        "b. A **train-test split** might not be representative of the entire dataset, especially if the data is not randomly distributed. **Cross-validation** reduces this risk by evaluating the model across different subsets of data.\n\n"
        "c. **Time-based splitting** respects the chronological order of data, which is important when past behavior influences future events. It prevents using future data to predict past events, which would not be possible in practice.\n\n"
        "d. **Testing on a separate set** validates that the model generalizes well to new, unseen data, ensuring confidence in its real-world performance."
    )
}

modu55le_4_q1 = {
    'question': (
        "A marketing team wants to segment their customer base to tailor marketing strategies effectively. They have demographic data, purchase history, website interactions, and engagement metrics."
        "\n\n"
        "a. Which clustering algorithm is appropriate for grouping customers based on similarity in their attributes, and why?"
        "\n"
        "Options:"
        "\n- K-Means Clustering"
        "\n- Hierarchical Clustering"
        "\n- DBSCAN"
        "\n- Gaussian Mixture Models\n\n"
        "b. Describe how the choice of distance metric can affect the clustering results in K-Means clustering. Provide examples of different distance metrics that could be used."
        "\n\n"
        "c. The team is unsure about the optimal number of clusters to use. Explain the 'elbow method' and how it can assist in determining the appropriate number of clusters."
        "\n\n"
        "d. After clustering, the team wants to assign new customers to the existing clusters. Explain how K-Means clustering can be used predictively for this purpose."
    ),
    'correct_answer': (
        "a. K-Means Clustering\n\n"
        "b. The distance metric determines how similarity between data points is measured. In K-Means, the default is Euclidean distance, but other metrics like Manhattan or Cosine distance can be used. The choice affects cluster formation; for example, Euclidean distance emphasizes overall magnitude differences, while Cosine distance focuses on the orientation between vectors.\n\n"
        "c. The 'elbow method' involves plotting the within-cluster sum of squares (WCSS) against the number of clusters (k). The point where the rate of decrease sharply changes (the 'elbow') suggests an optimal k, balancing cluster compactness and simplicity.\n\n"
        "d. New customers can be assigned to the nearest cluster center determined during the clustering process. By calculating the distance between a new customer's attributes and each cluster center, the customer is classified into the closest cluster."
    ),
    'explanation': (
        "a. **K-Means Clustering** is suitable for segmenting customers into distinct groups based on attribute similarity due to its simplicity and scalability.\n\n"
        "b. The **distance metric** affects how clusters are defined. Different metrics can capture different aspects of similarity, and choosing the right one depends on the data characteristics and the importance of certain attributes.\n\n"
        "c. The **elbow method** helps identify a point where adding more clusters doesn't significantly improve clustering performance, guiding the selection of an appropriate number of clusters.\n\n"
        "d. **Predictive use of K-Means** involves assigning new data points to the cluster with the nearest centroid, extending the clustering model to classify incoming customers."
    )
}

modu55le_4_q2 = {
    'question': (
        "A healthcare provider wants to identify patterns in patient data to improve treatment plans. They have collected data on symptoms, lab results, diagnoses, and treatment outcomes."
        "\n\n"
        "a. Which unsupervised learning technique can help the provider discover underlying groups of patients with similar characteristics?"
        "\n"
        "Options:"
        "\n- Principal Component Analysis (PCA)"
        "\n- K-Means Clustering"
        "\n- Association Rule Mining"
        "\n- Linear Regression\n\n"
        "b. Explain how the concept of distance norms (e.g., Euclidean, Manhattan) applies to clustering patient data. Why might one norm be chosen over another?"
        "\n\n"
        "c. The data includes both numerical and categorical variables. How can the provider modify the K-Means algorithm or choose a different clustering method to handle mixed data types?"
        "\n\n"
        "d. After clustering, the provider notices some outliers. Discuss how outliers can impact clustering results and what strategies can be employed to address them."
    ),
    'correct_answer': (
        "a. K-Means Clustering\n\n"
        "b. Distance norms define how similarity between patients is measured. **Euclidean distance** considers the straight-line distance between points, while **Manhattan distance** sums absolute differences. The choice depends on the importance of overall magnitude differences versus individual attribute differences. For patient data, Manhattan distance might be preferred if changes in individual symptoms are more significant than overall similarity.\n\n"
        "c. K-Means is not well-suited for categorical data. The provider can use algorithms designed for mixed data types, such as **K-Prototypes** or **Hierarchical Clustering** with appropriate distance measures, or encode categorical variables numerically using techniques like one-hot encoding.\n\n"
        "d. Outliers can distort cluster centroids and lead to misleading clusters. Strategies include removing outliers after investigation, using clustering algorithms robust to outliers (e.g., DBSCAN), or applying data transformation techniques to reduce the impact of extreme values."
    ),
    'explanation': (
        "a. **K-Means Clustering** helps discover groups of patients with similar characteristics without prior labels, making it suitable for this unsupervised learning task.\n\n"
        "b. **Distance norms** determine how differences between patients are quantified. The choice affects clustering results, and selecting a norm depends on the specific aspects of patient similarity that are most clinically relevant.\n\n"
        "c. Since **K-Means** handles only numerical data, the provider can use algorithms like **K-Prototypes** that accommodate mixed data types, or preprocess the data to convert categorical variables into numerical form.\n\n"
        "d. **Outliers** can skew clustering results by affecting centroid positions. Addressing outliers involves careful analysis to decide whether they are data errors or significant but rare cases, followed by appropriate handling."
    )
}

mod55ule_5_q1 = {
    'question': (
        "An energy company is analyzing sensor data from its equipment to predict potential failures. The dataset contains measurements like temperature, pressure, vibration levels, and includes some extreme values."
        "\n\n"
        "a. Identify the types of outliers that might be present in this dataset and explain how they could affect the analysis."
        "\n\n"
        "b. Describe a method for detecting point outliers in the temperature measurements. What statistical techniques can be applied?"
        "\n\n"
        "c. Once outliers are detected, what considerations should the company make before deciding to remove or keep them in the dataset?"
        "\n\n"
        "d. If some outliers are found to be due to sensor malfunctions, how should the company handle these data points during the data preparation phase?"
    ),
    'correct_answer': (
        "a. The dataset may contain **point outliers** (individual extreme values), **contextual outliers** (normal values in general but anomalous in a specific context), and **collective outliers** (a group of data points that are anomalous together). Outliers can skew statistical analyses, affect model training, and lead to incorrect predictions.\n\n"
        "b. They can use **box-and-whisker plots** to visualize the distribution and identify values beyond the interquartile range. **Z-scores** or the **3-sigma rule** can statistically flag values that are several standard deviations away from the mean.\n\n"
        "c. The company should consider whether outliers represent true anomalies (e.g., equipment failures) or data errors (e.g., sensor glitches). Removing outliers without investigation may discard important information about potential failures.\n\n"
        "d. Data points resulting from sensor malfunctions should be removed or corrected if possible. The company can replace them using data imputation methods or exclude them to prevent misleading the analysis."
    ),
    'explanation': (
        "a. Understanding the types of **outliers** helps determine their impact on the analysis and how to address them appropriately.\n\n"
        "b. **Statistical techniques** like box plots and z-scores help detect outliers by highlighting values that deviate significantly from the norm.\n\n"
        "c. Decisions on handling outliers should be informed by their cause and relevance. Outliers may contain valuable information about rare but critical events.\n\n"
        "d. **Sensor malfunctions** produce erroneous data that can mislead models. Cleaning such data ensures the integrity of the analysis."
    )
}

modu55le_6_q1 = {
    'question': (
        "A transportation company wants to monitor the average delivery times of its fleet to detect any significant changes that might indicate issues in the delivery process."
        "\n\n"
        "a. Explain how the company can use the CUSUM method to detect changes in average delivery times. What are the key components of the CUSUM algorithm?"
        "\n\n"
        "b. The company needs to decide on the parameters for the CUSUM method, such as the reference value and threshold. How do these parameters affect the sensitivity of change detection?"
        "\n\n"
        "c. Discuss the trade-offs between detecting changes quickly and the risk of false alarms in the context of setting the threshold value in the CUSUM method."
        "\n\n"
        "d. If the company observes that the CUSUM chart signals a change, what steps should they take to investigate and address the potential issue?"
    ),
    'correct_answer': (
        "a. The company can calculate the cumulative sum of deviations of delivery times from a target value (reference value). The CUSUM method involves computing the cumulative sum (S_t) and signaling a change when S_t exceeds a predefined threshold (T).\n\n"
        "b. The **reference value** represents the target or expected average delivery time. The **threshold (T)** determines when an alarm is raised. A lower threshold makes the method more sensitive to smaller changes but increases false alarms; a higher threshold reduces false alarms but may delay detection.\n\n"
        "c. Lower thresholds detect changes quickly but may result in more false positives, causing unnecessary investigations. Higher thresholds reduce false alarms but may miss or delay detection of actual changes. The company must balance the cost of false alarms against the risk of missing significant issues.\n\n"
        "d. They should analyze the period when the change was detected to identify possible causes (e.g., traffic issues, vehicle problems). Investigations may include reviewing operational data, interviewing drivers, and implementing corrective actions to address identified problems."
    ),
    'explanation': (
        "a. **CUSUM** helps detect shifts in process means by accumulating deviations over time. Key components are the cumulative sum and the threshold for signaling changes.\n\n"
        "b. **Parameters** like the reference value and threshold directly influence how sensitive the method is to changes. Setting these requires understanding the process variability and acceptable levels of risk.\n\n"
        "c. The **trade-off** involves balancing sensitivity and specificity. The company should consider operational costs associated with false alarms and the impact of delayed detection.\n\n"
        "d. Upon detection, the company should conduct a **root cause analysis** to determine the underlying issues and implement solutions to improve the delivery process."
    )
}

module_7_q151 = {
    'question': (
        "An airline wants to forecast passenger demand for the next 12 months to optimize flight schedules and resources. They have historical monthly data on passenger numbers, which exhibit both trend and seasonal patterns."
        "\n\n"
        "a. Which time series forecasting method should the airline use to account for both trend and seasonality in the data?"
        "\n"
        "Options:"
        "\n- Simple Exponential Smoothing"
        "\n- Holt's Linear Trend Method"
        "\n- Holt-Winters Exponential Smoothing"
        "\n- ARIMA without Seasonal Components\n\n"
        "b. Explain how the chosen method incorporates trend and seasonal components into the forecasting model."
        "\n\n"
        "c. The airline needs to determine the optimal values for the smoothing parameters (alpha, beta, gamma). Describe how they can estimate these parameters."
        "\n\n"
        "d. After fitting the model, how can the airline assess the accuracy of their forecasts? Mention at least two evaluation metrics."
    ),
    'correct_answer': (
        "a. Holt-Winters Exponential Smoothing\n\n"
        "b. Holt-Winters method extends exponential smoothing to include both a trend component (beta) and a seasonal component (gamma). It updates estimates for the level, trend, and seasonality at each time point, allowing the model to capture changes in both trend and seasonal patterns.\n\n"
        "c. They can estimate the parameters by minimizing a forecasting error metric, such as the sum of squared errors, over the historical data. This can be done using optimization techniques or built-in functions in statistical software that fit the model to the data.\n\n"
        "d. The airline can assess forecast accuracy using metrics like **Mean Absolute Error (MAE)** and **Mean Absolute Percentage Error (MAPE)**. They can also use **Mean Squared Error (MSE)** or visualize the forecast errors over time."
    ),
    'explanation': (
        "a. **Holt-Winters Exponential Smoothing** is designed to handle time series data with both trend and seasonal components.\n\n"
        "b. The method updates three equations at each time step: one for the level (overall average), one for the trend, and one for the seasonal component. This allows the model to adjust to changes in the data patterns.\n\n"
        "c. **Parameter estimation** involves finding values that minimize the difference between the predicted and actual values. This is often achieved through iterative optimization algorithms.\n\n"
        "d. **Evaluation metrics** like MAE and MAPE provide insights into the average forecast errors, helping the airline understand the model's predictive performance."
    )
}

module_7_q251 = {
    'question': (
        "A retailer is analyzing daily sales data to forecast future sales and manage inventory. The data shows an increasing trend and weekly seasonality."
        "\n\n"
        "a. Explain how the retailer can use the ARIMA model to forecast sales, including how to address the trend and seasonality in the data."
        "\n\n"
        "b. What do the parameters (p, d, q) and (P, D, Q)m represent in a Seasonal ARIMA (SARIMA) model?"
        "\n\n"
        "c. Describe the steps the retailer should take to identify the appropriate order of the ARIMA model, including any diagnostic plots or tests."
        "\n\n"
        "d. How can the retailer assess whether the residuals from the fitted ARIMA model are white noise?"
    ),
    'correct_answer': (
        "a. They can use a **Seasonal ARIMA (SARIMA)** model, which incorporates both non-seasonal and seasonal factors. Differencing can be applied to remove the trend (differencing order 'd') and seasonal differencing (order 'D') to remove seasonality. The model then includes autoregressive (AR) and moving average (MA) terms to capture patterns in the data.\n\n"
        "b. The parameters **p**, **d**, and **q** represent the non-seasonal autoregressive order, differencing order, and moving average order, respectively. **P**, **D**, and **Q** represent the seasonal autoregressive order, seasonal differencing order, and seasonal moving average order. **m** is the number of periods in each season (e.g., 7 for weekly seasonality).\n\n"
        "c. They should examine plots like the **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)** to identify patterns and determine the values of p and q. They can perform the **Augmented Dickey-Fuller test** to check for stationarity and decide on the differencing order d. Seasonal patterns in the ACF/PACF help identify P and Q.\n\n"
        "d. They can analyze the residuals using statistical tests like the **Ljung-Box test** to check for autocorrelation. Plotting the residuals and their ACF/PACF can also reveal whether they resemble white noise."
    ),
    'explanation': (
        "a. **ARIMA models** handle time series data by modeling dependencies in the data. Seasonal differencing and seasonal AR and MA terms in **SARIMA** account for seasonality.\n\n"
        "b. The parameters define the structure of the model, specifying how many past observations and errors are used in the model and how differencing is applied.\n\n"
        "c. **Identifying model order** involves analyzing the data's autocorrelation structure. Diagnostic plots and stationarity tests guide the selection of appropriate parameters.\n\n"
        "d. **Residual analysis** ensures that the model has captured all predictable patterns. If residuals are white noise, it indicates a good fit."
    )
}

module_8_q151 = {
    'question': (
        "A real estate company wants to understand how various factors affect house prices to guide their investment strategies. They have data on house prices and features like size, location, number of bedrooms, age, and amenities."
        "\n\n"
        "a. Which regression model should they use to quantify the relationship between house features and prices, and why?"
        "\n"
        "Options:"
        "\n- Simple Linear Regression"
        "\n- Multiple Linear Regression"
        "\n- Logistic Regression"
        "\n- Poisson Regression\n\n"
        "b. Explain the difference between using R-squared and Adjusted R-squared when evaluating the model. Why might Adjusted R-squared be more appropriate in this case?"
        "\n\n"
        "c. The company wants to test whether the number of bedrooms has a significant impact on price after controlling for other factors. Describe how they can use hypothesis testing within the regression framework to assess this."
        "\n\n"
        "d. They suspect that the relationship between house size and price is non-linear. How can they modify their regression model to account for this?"
    ),
    'correct_answer': (
        "a. Multiple Linear Regression\n\n"
        "b. **R-squared** measures the proportion of variance in the dependent variable explained by the model. **Adjusted R-squared** adjusts for the number of predictors, providing a more accurate measure when multiple variables are included. Adjusted R-squared is more appropriate here because it accounts for the addition of multiple predictors, penalizing unnecessary complexity.\n\n"
        "c. They can perform a **t-test** on the regression coefficient for the number of bedrooms. The null hypothesis is that the coefficient is zero (no effect). A low p-value indicates that the number of bedrooms significantly affects the price after controlling for other variables.\n\n"
        "d. They can include a **quadratic term** or perform a **logarithmic transformation** on house size. Adding **house size squared** as a predictor allows the model to capture curvature in the relationship."
    ),
    'explanation': (
        "a. **Multiple Linear Regression** is appropriate when modeling the relationship between a continuous dependent variable and multiple independent variables.\n\n"
        "b. **Adjusted R-squared** is better for multiple regression because it adjusts for the number of predictors, preventing overestimation of the model's explanatory power.\n\n"
        "c. **Hypothesis testing** on regression coefficients allows the company to assess the significance of individual predictors while controlling for others.\n\n"
        "d. **Non-linear relationships** can be modeled by including polynomial terms or transforming variables, enabling the regression to fit more complex patterns."
    )
}



module_10_q1t1 = {
    'question': (
        "A tech company wants to predict customer churn for its subscription service. The outcome variable is binary: churn (1) or not churn (0). They have data on customer usage patterns, demographic information, and engagement metrics."
        "\n\n"
        "a. Which regression model should they use for this classification problem, and why?"
        "\n"
        "Options:"
        "\n- Linear Regression"
        "\n- Logistic Regression"
        "\n- Poisson Regression"
        "\n- Ridge Regression\n\n"
        "b. Explain how logistic regression estimates the probability of churn and how the coefficients are interpreted."
        "\n\n"
        "c. The company is concerned about overfitting due to the large number of predictors. Describe how Lasso Regression can help in this scenario."
        "\n\n"
        "d. After building the model, they use a confusion matrix to evaluate performance. Define precision and recall in this context and explain their importance."
    ),
    'correct_answer': (
        "a. Logistic Regression\n\n"
        "b. **Logistic regression** models the log-odds of the probability of churn as a linear combination of predictors. The coefficients represent the change in the log-odds of churn for a one-unit increase in the predictor. By applying the logistic function, the model outputs probabilities between 0 and 1.\n\n"
        "c. **Lasso Regression** adds an L1 regularization term to the loss function, penalizing the absolute value of coefficients. This leads to some coefficients being reduced to zero, effectively performing variable selection and reducing overfitting.\n\n"
        "d. **Precision** is the proportion of correctly predicted churn cases out of all predicted churn cases. **Recall** is the proportion of actual churn cases that were correctly predicted. Both are important to understand the model's ability to identify churners accurately and the trade-off between false positives and false negatives."
    ),
    'explanation': (
        "a. **Logistic Regression** is appropriate for binary classification problems where the outcome is categorical.\n\n"
        "b. The model estimates **probabilities** using the logistic function, and coefficients indicate the direction and strength of the association between predictors and the log-odds of the outcome.\n\n"
        "c. **Lasso Regression** helps prevent overfitting by shrinking less important coefficients to zero, simplifying the model.\n\n"
        "d. **Precision and recall** provide insights into the model's predictive performance, especially in imbalanced datasets where accuracy alone may be misleading."
    )
}

module_2_q3 = {
    'question': (
        "George P. Burdell, a renowned data scientist, is developing a model to classify emails as 'Spam' or 'Not Spam' for his startup's new email client. The dataset includes features such as word frequency, email length, presence of certain keywords, and sender reputation."
        "\n\n"
        "a. Considering that the dataset may not be linearly separable, which classification algorithm should George use to maximize the margin between classes while allowing for some misclassifications?"
        "\n"
        "Options:"
        "\n- Hard Margin Support Vector Machine (SVM)"
        "\n- Soft Margin Support Vector Machine (SVM)"
        "\n- Naive Bayes Classifier"
        "\n- Decision Tree Classifier\n\n"
        "b. George notices that emails containing the word 'Congratulations' are often marked as spam, but sometimes they are legitimate. He wants his model to consider the cost of misclassifying legitimate emails as spam. How can he adjust his SVM model to account for different misclassification costs?"
        "\n\n"
        "c. Before training the model, George realizes that the word frequency features have varying scales. Why is it important for him to scale these features, and which scaling method would you recommend?"
        "\n\n"
        "d. After deploying the model, George wants to evaluate its performance. He decides to use a confusion matrix. Explain what a confusion matrix is and how George can interpret precision and recall in the context of his spam classifier."
    ),
    'correct_answer': (
        "a. Soft Margin Support Vector Machine (SVM)\n\n"
        "b. He can assign different penalty parameters (C) to misclassifications in the SVM model, using techniques like cost-sensitive learning to penalize false positives (legitimate emails marked as spam) more heavily.\n\n"
        "c. Scaling is important because SVMs are sensitive to the scale of input features. Features with larger scales can dominate the distance calculations. George should use techniques like standardization (z-score normalization) to scale the features so that they have a mean of zero and a standard deviation of one.\n\n"
        "d. A confusion matrix is a table that summarizes the performance of a classification model by displaying the true positives, false positives, true negatives, and false negatives. Precision is the proportion of correctly identified spam emails out of all emails predicted as spam, and recall is the proportion of correctly identified spam emails out of all actual spam emails. High precision means few legitimate emails are misclassified as spam, and high recall means most spam emails are correctly identified."
    ),
    'explanation': (
        "a. **Soft Margin SVM** allows for some misclassifications and maximizes the margin, which is suitable when data is not perfectly separable.\n\n"
        "b. By using **cost-sensitive learning** in SVM, George can adjust the penalty for different types of misclassification errors, ensuring that false positives (important emails marked as spam) are minimized.\n\n"
        "c. **Scaling features** ensures that no single feature disproportionately influences the model. **Standardization** is recommended as it centers the data around zero with unit variance.\n\n"
        "d. A **confusion matrix** provides insight into how well the classifier is performing. **Precision** and **recall** are critical metrics: precision focuses on the accuracy of positive predictions (spam), and recall measures the model's ability to find all relevant cases (actual spam emails)."
    )
}

module_4_q3 = {
    'question': (
        "George P. Burdell is now working on a project to group his collection of vintage vinyl records. He has data on each record, including genre, year of release, artist popularity, and number of tracks."
        "\n\n"
        "a. George wants to cluster his records to discover natural groupings. Which clustering algorithm would be appropriate for this task, and why?"
        "\n"
        "Options:"
        "\n- K-Means Clustering"
        "\n- Hierarchical Clustering"
        "\n- DBSCAN"
        "\n- Mean Shift Clustering\n\n"
        "b. George is unsure how many clusters to specify in his chosen algorithm. Describe how the elbow method can help him decide on the optimal number of clusters."
        "\n\n"
        "c. Suppose George's dataset includes both numerical features (year of release, number of tracks) and categorical features (genre). How can he modify his clustering approach to handle mixed data types?"
        "\n\n"
        "d. After clustering, George notices that some clusters have only a few records while others have many. What could be the reason for this imbalance, and how might he address it?"
    ),
    'correct_answer': (
        "a. Hierarchical Clustering\n\n"
        "b. The elbow method involves running the clustering algorithm with different numbers of clusters (k) and plotting the within-cluster sum of squares (WCSS) against k. The point where the rate of decrease sharply changes (the 'elbow') suggests the optimal number of clusters.\n\n"
        "c. George can use a clustering algorithm that handles mixed data types, such as Hierarchical Clustering with a suitable distance metric like Gower's distance, which can compute similarities between records with both numerical and categorical features.\n\n"
        "d. The imbalance could be due to natural groupings in the data or outliers. He might address it by analyzing the small clusters to see if they represent meaningful subgroups or if they are the result of noise. Alternatively, he could adjust the clustering parameters or use a different algorithm better suited for imbalanced cluster sizes."
    ),
    'explanation': (
        "a. **Hierarchical Clustering** does not require specifying the number of clusters in advance and can handle different types of data, making it appropriate for exploratory analysis.\n\n"
        "b. The **elbow method** helps determine the optimal number of clusters by identifying the point where adding more clusters doesn't significantly reduce the WCSS.\n\n"
        "c. For **mixed data types**, George can use distance metrics like **Gower's distance** and algorithms that can handle such data, allowing for effective clustering of records with both numerical and categorical features.\n\n"
        "d. **Imbalanced clusters** may result from inherent data distribution or outliers. George should investigate these clusters to understand their composition and consider methods like re-clustering, outlier removal, or using algorithms that can handle clusters of varying sizes."
    )
}

module_7_q3 = {
    'question': (
        "George P. Burdell is analyzing his monthly utility bills over the past five years to forecast future expenses and budget accordingly. His data shows a clear seasonal pattern with higher bills in the summer and winter months due to heating and cooling."
        "\n\n"
        "a. Which time series forecasting method should George use to account for both trend and seasonality in his utility bills?"
        "\n"
        "Options:"
        "\n- Simple Exponential Smoothing"
        "\n- Holt's Linear Trend Method"
        "\n- Holt-Winters Exponential Smoothing"
        "\n- ARIMA without Seasonal Components\n\n"
        "b. George wants to ensure that his forecasting model adapts to changes over time, such as improvements in energy efficiency at his home. Which components of the Holt-Winters method allow for this adaptability?"
        "\n\n"
        "c. Describe how George can use past data to estimate the smoothing parameters (alpha, beta, gamma) in the Holt-Winters model."
        "\n\n"
        "d. After generating forecasts, George wants to evaluate the accuracy of his model. Which error metrics should he consider, and why?"
    ),
    'correct_answer': (
        "a. Holt-Winters Exponential Smoothing\n\n"
        "b. The trend component (beta) and the seasonal component (gamma) of the Holt-Winters method allow the model to adapt to changes over time by updating estimates of the trend and seasonality as new data becomes available.\n\n"
        "c. George can use historical data to fit the model by selecting the smoothing parameters that minimize forecasting errors. This can be done using optimization techniques, such as minimizing the sum of squared errors between the actual and predicted values.\n\n"
        "d. George should consider using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Mean Absolute Percentage Error (MAPE). These metrics measure the average forecast errors and help assess the model's accuracy in predicting future utility bills."
    ),
    'explanation': (
        "a. **Holt-Winters Exponential Smoothing** accounts for both trend and seasonal patterns, making it suitable for George's utility bill forecasting.\n\n"
        "b. The **beta (trend)** and **gamma (seasonality)** parameters allow the model to update the trend and seasonal components over time, enabling it to adapt to changes like energy efficiency improvements.\n\n"
        "c. By using **historical data**, George can apply optimization algorithms to find the values of alpha, beta, and gamma that result in the best fit of the model to the data, minimizing forecasting errors.\n\n"
        "d. **MAE**, **MSE**, and **MAPE** provide insights into the average magnitude of forecast errors, helping George evaluate how well his model predicts future expenses and adjust it if necessary."
    )
}

module_10_q2 = {
    'question': (
        "George P. Burdell is fascinated by the relationship between his coffee consumption and productivity levels. He collects data over several months, recording the number of cups of coffee he drinks per day and his corresponding productivity score on a scale of 1 to 10."
        "\n\n"
        "a. George suspects that there might be a non-linear relationship between coffee consumption and productivity. Which regression technique can he use to model this potential non-linear relationship?"
        "\n"
        "Options:"
        "\n- Linear Regression"
        "\n- Polynomial Regression"
        "\n- Logistic Regression"
        "\n- Ridge Regression\n\n"
        "b. Explain how George can incorporate polynomial terms into his regression model and interpret the coefficients."
        "\n\n"
        "c. After fitting the model, George notices multicollinearity among the predictors. How can he address this issue, and which regression method can help mitigate multicollinearity?"
        "\n\n"
        "d. George wants to ensure that his model doesn't overfit the data. Describe how he can use cross-validation to assess the model's generalization performance."
    ),
    'correct_answer': (
        "a. Polynomial Regression\n\n"
        "b. George can include higher-degree terms of the coffee consumption variable (e.g., squared, cubed terms) in his regression model. For example, the model could be: Productivity = β0 + β1*(Coffee) + β2*(Coffee)^2 + β3*(Coffee)^3 + ε. The coefficients represent the impact of each term on productivity, capturing the curvature in the relationship.\n\n"
        "c. He can address multicollinearity by using Ridge Regression, which adds an L2 penalty term to the loss function. This regularization technique shrinks the coefficients of correlated predictors, reducing their variance and mitigating multicollinearity.\n\n"
        "d. George can perform k-fold cross-validation by splitting his data into k subsets, training the model on k-1 subsets, and validating it on the remaining subset. By repeating this process k times and averaging the results, he can assess the model's ability to generalize to new data."
    ),
    'explanation': (
        "a. **Polynomial Regression** allows modeling of non-linear relationships by including polynomial terms of the predictors.\n\n"
        "b. By adding **polynomial terms**, George can fit a curve to the data. The coefficients of these terms indicate the contribution of each degree of coffee consumption to productivity.\n\n"
        "c. **Ridge Regression** is effective in handling multicollinearity by imposing a penalty on the size of coefficients, which discourages large values and reduces the impact of multicollinearity.\n\n"
        "d. **Cross-validation** provides a robust estimate of the model's performance on unseen data, helping George detect overfitting by evaluating how well the model generalizes beyond the training data."
    )
}


OPEN_QUESTIONS = []
global_items = list(globals().items())

for name, value in global_items:
    if not name.startswith('_'):
        OPEN_QUESTIONS.append(value)


MIDTERM1_OPEN_QUESTIONS = OPEN_QUESTIONS[:-1]
