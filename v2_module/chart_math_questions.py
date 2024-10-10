# questions_1 = {
#     "fp": 'imgs/confusion_matrix_1.png',
#     "questions": [
#         {
#             'question': "What is the accuracy of the model?",
#             'options_list': ['85%', '90%', '95.5%', '92%'],
#             'correct_answer': "95.5%. Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN). Calculation: (960 + 950) / (960 + 950 + 50 + 40) = 0.955. It reflects the overall effectiveness of the model in disease prediction."
#         },
#         {
#             'question': "What is the precision of the model?",
#             'options_list': ['90%', '95.05%', '93%', '96%'],
#             'correct_answer': "95.05%. Formula: Precision = TP / (TP + FP). Calculation: 960 / (960 + 50) = 0.9505. Indicates the model's ability to predict disease presence accurately."
#         },
#         {
#             'question': "What is the recall (sensitivity) of the model?",
#             'options_list': ['96%', '94%', '92%', '98%'],
#             'correct_answer': "96%. Formula: Recall = TP / (TP + FN). Calculation: 960 / (960 + 40) = 0.96. Shows the model's effectiveness at identifying all actual cases of the disease."
#         },
#         {
#             'question': "What is the specificity of the model?",
#             'options_list': ['95%', '93%', '90%', '97%'],
#             'correct_answer': "95%. Formula: Specificity = TN / (TN + FP). Calculation: 950 / (950 + 50) = 0.95. Demonstrates how well the model identifies individuals without the disease."
#         }
#     ]
# }


questions_2 = {
    "fp": 'imgs/confusion_matrix_2.png',
    "questions": [
        {
            'question': "What is the accuracy of the model in predicting product defects?",
            'options_list': ['85%', '90%', '95.5%', '92%'],
            'correct_answer': "95.5%. Formula: Accuracy = (TP + TN) / (Total). Calculation: (940 + 970) / 2000 = 0.955. Reflects the model's overall effectiveness in predicting product defects."
        },
        {
            'question': "How precise is the model in identifying defective products?",
            'options_list': ['94%', '95%', '93%', '94.17%'],
            'correct_answer': "94.17%. Formula: Precision = TP / (TP + FP). Calculation: 970 / (970 + 60) ≈ 0.9417. Indicates the reliability of the model's defect predictions."
        },
        {
            'question': "What is the recall rate of the model for detecting defective products?",
            'options_list': ['96%', '97%', '95%', '98%'],
            'correct_answer': "97%. Formula: Recall = TP / (TP + FN). Calculation: 970 / (970 + 30) = 0.97. Shows the model's ability to identify all truly defective products."
        },
        {
            'question': "What does the model's specificity tell us about its performance?",
            'options_list': ['93%', '94%', '95%', '96%'],
            'correct_answer': "94%. Formula: Specificity = TN / (TN + FP). Calculation: 940 / (940 + 60) = 0.94. Demonstrates the model's effectiveness in correctly identifying non-defective products."
        },
        {
            'question': "What would happen if we increased the threshold for classifying products as defective?",
            'options_list': ['Increase precision, decrease recall', 'Decrease precision, increase recall', 'Increase both precision and recall', 'Decrease both precision and recall'],
            'correct_answer': "Increase precision, decrease recall. Increasing the threshold generally leads to fewer products being classified as defective, which might increase the model's precision (fewer false positives) but decrease its recall (more false negatives), as it becomes more conservative in predicting defects."
        }
    ]
}

questions_3 = {
    "fp": 'imgs/confusion_matrix_3.png',
    "questions": [
        {
            'question': "If it was crucial to identify as many at-risk students as possible, what aspect of the model should we focus on improving?",
            'options_list': ['Accuracy', 'Precision', 'Recall', 'Specificity'],
            'correct_answer': "Recall. Improving recall would mean the model is better at identifying all students who are truly at risk, minimizing the chances of any student slipping through the cracks. This is crucial for providing timely interventions."
        },
        {
            'question': "If we want to ensure that interventions are only provided when truly needed, to avoid unnecessary stress or resource allocation, which metric should we prioritize?",
            'options_list': ['Accuracy', 'Precision', 'Recall', 'Specificity'],
            'correct_answer': "Precision. Enhancing precision ensures that when the model predicts a student is at risk of failing ISYE6501, there is a high probability they truly are. This minimizes interventions based on false alarms."
        },
        {
            'question': "What is the model's precision in identifying students at risk of failing the class (and possibly dropping out), and why is this important?",
            'options_list': ['85%', '86.36%', '90%', '95%'],
            'correct_answer': "86.36%. Formula: Precision = TP / (TP + FP). Calculation: 950 / (950 + 150) = 0.8636. High precision is vital to ensure resources for interventions are allocated efficiently, targeting students who need help the most."
        },
        {
            'question': "Given the model's recall is 95%, what does this indicate about its ability to identify at-risk students, and how could this impact student success?",
            'options_list': ['It indicates a high rate of false positives', 'It suggests a balanced approach between identifying at-risk and not at-risk students', 'It shows the model is effective at identifying nearly all at-risk students, which is crucial for early interventions', 'It means the model is too conservative in predicting at-risk students'],
            'correct_answer': "It shows the model is effective at identifying nearly all at-risk students, which is crucial for early interventions. Formula: Recall = TP / (TP + FN). Calculation: 950 / (950 + 50) = 0.95. By catching nearly all at-risk students, the model helps in implementing interventions that could significantly enhance their chances of success in the class."
        }
    ]
}

questions_4 = {
    "fp": 'imgs/stats_1.png',
    "questions": [
        {
            'question': "What does the R-squared value indicate about the model's effectiveness?",
            'options_list': ['It measures the model’s prediction accuracy.', 'It indicates the percentage of the variance in the dependent variable that is predictable from the independent variable(s).', 'It represents the residual error of the model.', 'It shows the coefficient of the independent variable.'],
            'correct_answer': "It indicates the percentage of the variance in the dependent variable that is predictable from the independent variable(s). R-squared value of 0.214 suggests that approximately 21.4% of the variance in sales can be explained by the model."
        },
        {
            'question': "How does the model interpret the relationship between 'Price' and 'Sales'?",
            'options_list': ['Positive correlation: as Price increases, Sales increase.', 'Negative correlation: as Price increases, Sales decrease.', 'No correlation: Price does not affect Sales.', 'Inversely proportional: as Price decreases, Sales increase exponentially.'],
            'correct_answer': "Negative correlation: as Price increases, Sales decrease. The coefficient for 'Price' is -0.0679, indicating that as price increases, sales are expected to decrease."
        },
        {
            'question': "Why is the F-statistic important, and what does its value convey about the model?",
            'options_list': ['It indicates the overall significance of the model.', 'It measures the individual impact of each predictor.', 'It represents the error term of the model.', 'It adjusts the R-squared value for the number of predictors.'],
            'correct_answer': "It indicates the overall significance of the model. The F-statistic value of 108.2 and its associated p-value (< 2.2e-16) suggest that the model is statistically significant."
        },
        {
            'question': "Based on the summary, how statistically significant are the model's predictors?",
            'options_list': ['Highly significant, as indicated by p-values < 0.05.', 'Not significant, as indicated by p-values > 0.05.', 'Only the intercept is significant.', 'Significance cannot be determined from the provided information.'],
            'correct_answer': "Highly significant, as indicated by p-values < 0.05. Both the intercept and 'Price' have p-values significantly less than 0.05, indicating they are statistically significant predictors of 'Sales'."
        }
    ]
}

questions_5 = {
    "fp": 'imgs/stats_2.png',
    "questions": [
        {
            'question': "How many samples were there in the 2007 dataset?",
            'options_list': ['404', '506', '142', '350'],
            'correct_answer': "142. The dataset for the year 2007 contains 142 observations, each representing a different country."
        },
        {
            'question': "How many independent variables were included in the model?",
            'options_list': ['1', '2', '3', '4'],
            'correct_answer': "1. The model includes one independent variable, 'gdpPercap', to predict life expectancy ('lifeExp')."
        },
        {
            'question': "How do you interpret the coefficient of 'gdpPercap'?",
            'options_list': ['For each unit increase in GDP per capita, life expectancy decreases.', 'GDP per capita has no significant effect on life expectancy.', 'For each unit increase in GDP per capita, life expectancy increases by 0.0006 years.', "The coefficient of 'gdpPercap' is statistically insignificant."],
            'correct_answer': "For each unit increase in GDP per capita, life expectancy increases by 0.0006 years. This indicates a positive relationship between GDP per capita and life expectancy, suggesting that as a country's GDP per capita increases, so does the life expectancy of its people."
        },
        {
            'question': "What does the intercept represent in this model?",
            'options_list': ['The median GDP per capita when life expectancy is zero.', 'The predicted life expectancy when GDP per capita is 0.', 'The average number of years added to life expectancy per unit increase in GDP per capita.', 'The average life expectancy across all countries in 2007.'],
            'correct_answer': "The predicted life expectancy when GDP per capita is 0. The intercept, estimated at 59.5657, represents the model's prediction for life expectancy in the hypothetical scenario where GDP per capita is zero, serving as a baseline against which the impact of GDP per capita on life expectancy is measured."
        },
        {
            'question': "How is the adjusted R-squared calculated, and what is its value for this model?",
            'options_list': ['Adjusted R-squared = 1 - (1 - R-squared) * (n - 1) / (n - k - 1)', 'Adjusted R-squared = R-squared * (n - 1) / (n - k - 1)', 'Adjusted R-squared = 1 + (1 - R-squared) * (n - 1) / (n - k - 1)', 'Adjusted R-squared = 1 - (1 - R-squared) / (n - k)'],
            'correct_answer': "Adjusted R-squared = 1 - (1 - R-squared) * (n - 1) / (n - k - 1). The adjusted R-squared compensates for the number of predictors in the model relative to the number of observations. It provides a more accurate measure of the model's explanatory power when including multiple predictors. For this model, the adjusted R-squared is 0.637, indicating that after adjusting for the number of predictors, approximately 63.7% of the variance in life expectancy can be explained by the model."
        }
    ]
}

questions_6 = {
    "fp": 'imgs/roc_curve.png',
    "questions": [
        {
            'question': "Based on the ROC curve provided, what is the AUC (Area Under the Curve) of the model?",
            'options_list': ['0.25','0.48', '0.61', '0.75', '0.85'],
            'correct_answer': "0.48. The AUC represents the model's ability to distinguish between classes. A value of 0.48 indicates the model is performing slightly worse than random guessing (0.5)."
        },
        {
            'question': "What does the red point marked on the ROC curve signify?",
            'options_list': ['The point where sensitivity equals specificity.', 'The default classification threshold for the model.', 'The threshold with the highest precision.', 'The threshold minimizing false negatives.'],
            'correct_answer': "The default classification threshold for the model. This point shows the model's performance when the probability threshold is set to 0.5."
        },
        {
            'question': "If the goal is to minimize false negatives, which region of the ROC curve should be considered?",
            'options_list': ['Top-left corner.', 'Bottom-left corner.', 'Top-right corner.', 'Along the diagonal line.'],
            'correct_answer': "Top-left corner. This region represents high true positive rates (sensitivity) and low false positive rates."
        }
    ]
}

questions_7 = {
    "fp": 'imgs/svm_decision_boundary.png',
    "questions": [
        {
            'question': "In the SVM plot provided, which points are the support vectors?",
            'options_list': ['Points closest to the decision boundary.', 'Points farthest from the decision boundary.', 'All points in the majority class.', 'Randomly selected points.'],
            'correct_answer': "Points closest to the decision boundary. Support vectors are the data points that lie closest to the decision boundary and influence its position."
        },
        {
            'question': "What effect would increasing the penalty parameter C have on the decision boundary?",
            'options_list': ['It would create a wider margin.', 'It would allow more misclassifications.', 'It would result in a narrower margin with fewer misclassifications.', 'It would have no effect.'],
            'correct_answer': "It would result in a narrower margin with fewer misclassifications. A higher C penalizes misclassifications more, leading to a tighter fit."
        },
        {
            'question': "If the data is not linearly separable, how can the SVM model be adapted?",
            'options_list': ['By using a different classification algorithm.', 'By applying a kernel function.', 'By increasing the number of support vectors.', 'By reducing the dataset size.'],
            'correct_answer': "By applying a kernel function. Kernel functions allow the SVM to create non-linear decision boundaries by transforming the data into a higher-dimensional space."
        }
    ]
}

questions_8 = {
    "fp": 'imgs/time_series_seasonality.png',
    "questions": [
        {
            'question': "What components are visible in the time series plot provided?",
            'options_list': ['Trend only.', 'Seasonality only.', 'Trend and seasonality.', 'Random noise only.'],
            'correct_answer': "Trend and seasonality. The plot shows both a general upward or downward trend and repeating patterns over regular intervals."
        },
        {
            'question': "Which forecasting method is most appropriate for this time series?",
            'options_list': ['Simple Moving Average.', "Holt's Linear Trend Method.", 'Holt-Winters Exponential Smoothing.', 'ARIMA without differencing.'],
            'correct_answer': "Holt-Winters Exponential Smoothing. This method accounts for both trend and seasonality in the data."
        },
        {
            'question': "How can the seasonality component be removed to analyze the underlying trend?",
            'options_list': ['By differencing the data.', 'By applying a moving average.', 'By decomposing the time series.', 'By increasing the sample size.'],
            'correct_answer': "By decomposing the time series. Decomposition separates the time series into trend, seasonal, and residual components."
        }
    ]
}

questions_9 = {
    "fp": 'imgs/kmeans_clusters.png',
    "questions": [
        {
            'question': "How many clusters are shown in the K-Means clustering plot?",
            'options_list': ['2', '3', '4', '5'],
            'correct_answer': "3. The plot displays three distinct clusters, each represented by a different color."
        },
        {
            'question': "What is the primary goal of the K-Means algorithm?",
            'options_list': ['To maximize the distance between clusters.', 'To minimize the distance within clusters.', 'To classify data based on labels.', 'To reduce dimensionality.'],
            'correct_answer': "To minimize the distance within clusters. K-Means aims to group data points so that those within a cluster are as close as possible to each other."
        },
        {
            'question': "Which method can help determine the optimal number of clusters in K-Means?",
            'options_list': ['Silhouette Analysis.', 'Principal Component Analysis.', 'Linear Regression.', 'Decision Trees.'],
            'correct_answer': "Silhouette Analysis. This method measures how similar an object is to its own cluster compared to other clusters, helping to select the optimal number of clusters."
        }
    ]
}

questions_12 = {
    "fp": 'imgs/cusum_chart.png',
    "questions": [
        {
            'question': "At approximately which time point does the CUSUM chart indicate a significant change in the process?",
            'options_list': ['Time point 25', 'Time point 50', 'Time point 75', 'No significant change is indicated'],
            'correct_answer': "Time point 50. The CUSUM chart shows a shift starting at time point 50, where the cumulative sum exceeds the decision interval."
        },
        {
            'question': "What is the purpose of the decision interval (h) in the CUSUM method?",
            'options_list': [
                "To set the target value for the process.",
                "To provide a threshold for detecting significant shifts in the process mean.",
                "To adjust the sensitivity of the process to small fluctuations.",
                "To calculate the slack value (k) used in the CUSUM formula."
            ],
            'correct_answer': "To provide a threshold for detecting significant shifts in the process mean. When the cumulative sum exceeds h, it signals a significant change."
        },
        {
            'question': "In the context of the CUSUM chart, what does a positive cumulative sum (S_positive) indicate?",
            'options_list': [
                "A decrease in the process mean.",
                "An increase in the process mean.",
                "No change in the process mean.",
                "An error in the data collection."
            ],
            'correct_answer': "An increase in the process mean. The positive CUSUM detects upward shifts in the process."
        },
        {
            'question': "Why are both S_positive and S_negative plotted in a CUSUM chart?",
            'options_list': [
                "To monitor for both increases and decreases in the process mean.",
                "To compare two different datasets.",
                "To separate the data into training and testing sets.",
                "To account for seasonal variations in the process."
            ],
            'correct_answer': "To monitor for both increases and decreases in the process mean. S_positive detects upward shifts, while S_negative detects downward shifts."
        },
        {
            'question': "If the slack value (k) is increased, how does it affect the sensitivity of the CUSUM chart?",
            'options_list': [
                "The chart becomes more sensitive to small shifts.",
                "The chart becomes less sensitive to small shifts.",
                "It has no effect on sensitivity.",
                "It reverses the direction of the cumulative sums."
            ],
            'correct_answer': "The chart becomes less sensitive to small shifts. A larger k reduces the impact of small deviations, requiring larger shifts to signal a change."
        }
    ]
}

questions_11_updated = {
    "fp": 'imgs/regression_coefficients.png',
    "questions": [
        {
            'question': "Based on the regression coefficients plot, which predictor has the largest positive effect on the response variable?",
            'options_list': ['Intercept', 'X1', 'X2', 'X3'],
            'correct_answer': "X1. The coefficient for X1 is positive and has the largest magnitude among the predictors, indicating the strongest positive effect."
        },
        {
            'question': "Which predictor has a negative relationship with the response variable?",
            'options_list': ['Intercept', 'X1', 'X2', 'X3'],
            'correct_answer': "X2. The coefficient for X2 is negative, indicating an inverse relationship with the response variable."
        },
        {
            'question': "Which predictor's coefficient is not statistically significant at the 95% confidence level?",
            'options_list': ['Intercept', 'X1', 'X2', 'X3'],
            'correct_answer': "X3. The confidence interval for X3 includes zero, suggesting it is not statistically significant."
        },
        {
            'question': "What does the length of the error bar (confidence interval) for each coefficient represent?",
            'options_list': [
                "The variability of the predictor variable.",
                "The precision of the coefficient estimate; longer bars indicate less precise estimates.",
                "The importance of the predictor in the model.",
                "The correlation between predictors."
            ],
            'correct_answer': "The precision of the coefficient estimate; longer bars indicate less precise estimates. Wider confidence intervals reflect greater uncertainty in the coefficient estimate."
        },
        {
            'question': "If you were to remove one predictor to simplify the model, which one would be the best candidate based on the plot, and why?",
            'options_list': [
                "Intercept, because it doesn't have a confidence interval.",
                "X1, because it has the largest effect.",
                "X2, because it has a negative effect.",
                "X3, because its coefficient is not statistically significant."
            ],
            'correct_answer': "X3, because its coefficient is not statistically significant. Removing non-significant predictors can simplify the model without sacrificing explanatory power."
        }
    ]
}


IMAGE_QUESTIONS = []
global_items = list(globals().items())

for name, value in global_items:
    if not name.startswith('_'):
        IMAGE_QUESTIONS.append(value)

OPEN_QUESTIONS = IMAGE_QUESTIONS[:-1]
