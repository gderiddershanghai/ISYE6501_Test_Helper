questions_1 = {
    "fp": 'imgs/confusion_matrix_1.png',
    "questions": [
        {
            'question': "What is the accuracy of the model?",
            'options_list': ['85%', '90%', '95.5%', '92%'],
            'correct_answer': "95.5%. Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN). Calculation: (960 + 950) / (960 + 950 + 50 + 40) = 0.955. It reflects the overall effectiveness of the model in disease prediction."
        },
        {
            'question': "What is the precision of the model?",
            'options_list': ['90%', '95.05%', '93%', '96%'],
            'correct_answer': "95.05%. Formula: Precision = TP / (TP + FP). Calculation: 960 / (960 + 50) = 0.9505. Indicates the model's ability to predict disease presence accurately."
        },
        {
            'question': "What is the recall (sensitivity) of the model?",
            'options_list': ['96%', '94%', '92%', '98%'],
            'correct_answer': "96%. Formula: Recall = TP / (TP + FN). Calculation: 960 / (960 + 40) = 0.96. Shows the model's effectiveness at identifying all actual cases of the disease."
        },
        {
            'question': "What is the specificity of the model?",
            'options_list': ['95%', '93%', '90%', '97%'],
            'correct_answer': "95%. Formula: Specificity = TN / (TN + FP). Calculation: 950 / (950 + 50) = 0.95. Demonstrates how well the model identifies individuals without the disease."
        }
    ]
}


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



IMAGE_QUESTIONS = []
global_items = list(globals().items())

for name, value in global_items:
    if not name.startswith('_'):
        IMAGE_QUESTIONS.append(value)

OPEN_QUESTIONS = IMAGE_QUESTIONS[:-1]
