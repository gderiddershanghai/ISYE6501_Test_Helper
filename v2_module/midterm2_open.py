q1 = {
    'question': (
        "For each of the scenarios listed, identify the most fitting model or technique from the given options. Note that certain models or techniques might be suitable for more than one scenario.\n\n"
        "**MODELS/TECHNIQUES**\n\n"
        "- i. Support Vector Machines\n"
        "- ii. Decision Trees\n"
        "- iii. Linear Regression\n"
        "- iv. Poisson Models\n"
        "- v. Time Series Analysis\n"
        "- vi. Markov Decision Processes\n\n"
        "**SCENARIOS**\n\n"
        "- **a.** Predicting the number of times a particular software feature will be used in the next month based on historical usage data.\n"
        "- **b.** Developing a plan for the sequential release of product features over time to maximize user adoption and satisfaction.\n"
        "- **c.** Evaluating the impact of advertising spend on sales growth, adjusting for seasonal effects.\n"
        "- **d.** Identifying the primary predictors of energy consumption in commercial buildings to improve efficiency.\n"
        "- **e.** Calculating the optimal strategy for inventory management to minimize holding costs and avoid stockouts."
    ),
    'correct_answer': (
        "The most suitable models or techniques for each scenario are as follows:\n\n"
        "- **a.** Poisson Models\n"
        "- **b.** Markov Decision Processes\n"
        "- **c.** Time Series Analysis\n"
        "- **d.** Decision Trees\n"
        "- **e.** Linear Regression"
    ),
    'explanation': (
        "- **Poisson Models** are ideal for predicting counts or frequencies, making them suitable for scenario a.\n"
        "- **Markov Decision Processes** help in planning under uncertainty, fitting scenario b well.\n"
        "- **Time Series Analysis** is used for forecasting future values based on past data, applicable to scenario c.\n"
        "- **Decision Trees** can classify or predict based on decision rules, making them a good choice for scenario d.\n"
        "- **Linear Regression** models the relationship between a dependent variable and one or more independent variables, which suits the requirements of scenario e."
    )
}

q2 = {
    'question': (
        "An e-commerce platform is exploring ways to optimize its recommendation system to increase sales while maintaining a high level of customer satisfaction. The platform proposes testing two new algorithms:\n\n"
        "1. An algorithm prioritizing products based on individual's browsing history and past purchase behavior.\n"
        "2. An algorithm considering broader trends, like seasonal purchases and top-rated products, alongside individual preferences.\n\n"
        "**Question A1:** Describe a comprehensive approach to evaluating these algorithms before full-scale implementation.\n\n"
        "- **a.** Discuss the experimental design for comparing the two algorithms, considering controlling variables such as time of day, product categories, and customer demographics.\n\n"
        "- **b.** Identify the most appropriate models or approaches for predicting the outcomes of each recommendation algorithm from the following:\n\n"
        "  - Time Series Analysis\n"
        "  - Simulation Modelling\n"
        "  - Queuing Theory\n"
        "  - Bayesian Inference\n"
        "  - Multivariate Regression Analysis\n\n"
        "- **c.** Propose how the platform could use simulation to model the long-term effects of each algorithm on customer purchase patterns and satisfaction, including the key performance indicators (KPIs) to monitor."
    ),
    'correct_answer': (
        "**a.** The platform should utilize a randomized controlled trial (RCT) design, ensuring participants are randomly assigned to each algorithm to control for external variables.\n\n"
        "**b.** The most appropriate models for this scenario are:\n"
        "- Time Series Analysis for seasonal trends.\n"
        "- Simulation Modelling to predict customer behavior under each algorithm.\n"
        "- Bayesian Inference to update predictions as more data becomes available.\n\n"
        "**c.** The platform can use discrete event simulation to model long-term effects, focusing on KPIs such as customer lifetime value (CLV), repeat purchase rate, and customer satisfaction scores."
    ),
    'explanation': (
        "- An **RCT** ensures a fair comparison by minimizing bias.\n"
        "- **Time Series Analysis** is ideal for capturing seasonal trends.\n"
        "- **Simulation Modelling** accurately predicts behavior in complex systems with many interacting variables.\n"
        "- **Bayesian Inference** provides a flexible framework for updating predictions.\n"
        "- Simulating the long-term effects allows the platform to anticipate changes in customer behavior and satisfaction, ensuring the chosen algorithm aligns with business goals and enhances the customer experience."
    )
}

q3 = {
    'question': (
        "A finance research team is analyzing a dataset to predict future stock market trends, facing dataset challenges such as heteroscedasticity, evident trends and seasonal patterns, and high dimensionality with potential correlation among predictors.\n\n"
        "- **a.** To address heteroscedasticity in stock prices, which transformation technique should the team apply?\n\n"
        "  1. Linear Transformation\n"
        "  2. Box-Cox Transformation\n"
        "  3. Logarithmic Transformation\n"
        "  4. Z-Score Normalization\n\n"
        "- **b.** When detrending the data to remove evident trends and seasonal patterns, what is the primary reason for doing so?\n\n"
        "  1. To increase the dataset size\n"
        "  2. To enhance the computational speed of model training\n"
        "  3. To reduce the effect of time-related confounding factors\n"
        "  4. To improve the color scheme of data visualizations\n\n"
        "- **c.** To reduce dimensionality and mitigate multicollinearity among economic indicators, which method should the team use?\n\n"
        "  1. Principal Component Analysis (PCA)\n"
        "  2. Linear Discriminant Analysis (LDA)\n"
        "  3. K-Means Clustering\n"
        "  4. Decision Trees\n\n"
        "- **d.** In Principal Component Analysis (PCA), what role do eigenvalues and eigenvectors play?\n\n"
        "  1. They determine the color palette for data visualization.\n"
        "  2. They are used to calculate the mean and median of the dataset.\n"
        "  3. They identify the principal components and ensure they are orthogonal.\n"
        "  4. They enhance the dataset's security and privacy."
    ),
    'correct_answer': (
        "**a.** Box-Cox Transformation\n\n"
        "**b.** To reduce the effect of time-related confounding factors\n\n"
        "**c.** Principal Component Analysis (PCA)\n\n"
        "**d.** They identify the principal components and ensure they are orthogonal."
    ),
    'explanation': (
        "Box-Cox Transformation is used to stabilize variance across the data.\n\n"
        "Detrending is crucial for removing the influence of temporal trends on the analysis, "
        "making the underlying patterns in data more apparent.\n\n"
        "PCA is a powerful tool for dimensionality reduction and dealing with multicollinearity by "
        "transforming the dataset into a set of orthogonal (uncorrelated) variables, which are the principal components.\n\n"
        "Eigenvalues and eigenvectors are fundamental in determining the direction and magnitude of "
        "these principal components, helping to understand the dataset's variance structure."
    )
}

# q4 = {
#     'question': "A retail company is leveraging its extensive transactional and operational data to enhance various aspects of its business. The company has identified specific areas where analytical models could be applied for insights and optimization.\n\nFor each of the following scenarios, select the most appropriate model from the options: ARIMA, Louvain Algorithm, Integer Programming, SVM (Support Vector Machine), Neural Net, and Poisson.\n\nA. Forecasting monthly customer foot traffic in stores nationwide, based on historical data showing trends and seasonality.\n\nB. Analyzing transaction data to identify clusters of products that are frequently purchased together across different regions.\n\nC. Optimizing the distribution of staff across different departments in a store to minimize customer wait times while considering budget constraints.\n\nD. Predicting the likelihood of a product being out of stock based on sales velocity, supply chain delays, and current inventory levels.\n\nE. Estimating the number of product returns next quarter based on return rates of past quarters, adjusting for recent changes in return policy.",
#     'correct_answer': (
#         "A. ARIMA\n\n"
#         "B. Louvain Algorithm\n\n"
#         "C. Integer Programming\n\n"
#         "D. SVM (Support Vector Machine)\n\n"
#         "E. Poisson\n\n"
#     ),
#     'explanation': (
#         "ARIMA is suited for time series forecasting, making it ideal for predicting foot traffic with trends and seasonality.\n\n"
#         "The Louvain Algorithm is effective for detecting communities within networks, suitable for finding product clusters.\n\n"
#         "Integer Programming is used for resource allocation problems, optimal for staff distribution.\n\n"
#         "SVM can classify data with high accuracy, perfect for predicting stock levels.\n\n"
#         "The Poisson model is good for predicting the number of events happening in a fixed interval, making it suitable for estimating product returns.\n\n"
#     )
# }

# q5 =  {
#     'question': "A marketing research firm is utilizing statistical models to analyze consumer data and market trends. For each of the scenarios below, decide whether linear regression, logistic regression, or ridge regression is the most suitable model based on the type of data and the prediction objective.\n\n1. Predicting the percentage increase in sales volume for a retail store next quarter based on advertising spend, seasonality, and economic indicators.\n\n2. Determining whether a new advertisement campaign will be successful (Yes/No) in increasing market share, based on historical campaign data and competitor activity.\n\n3. Estimating the impact of price changes on demand for a popular product, considering multicollinearity issues among predictors such as price, promotional activities, and competitor pricing strategies.\n\n4. Assessing the probability of a customer making a purchase based on their browsing history, age, income level, and previous purchase history.",
#     'correct_answer': (
#         "1. Linear Regression\n\n"
#         "2. Logistic Regression\n\n"
#         "3. Ridge Regression\n\n"
#         "4. Logistic Regression"
#     ),
#     'explanation': (
#         "Linear Regression is suited for continuous outcomes, making it appropriate for predicting sales volume percentage increases.\n\n"
#         "Logistic Regression is designed for binary outcomes, ideal for predicting the success of an advertisement campaign and assessing customer purchase probability.\n\n"
#         "Ridge Regression addresses multicollinearity among predictors, which is crucial for accurately estimating the impact of price changes on product demand."
#     )
# }

# q6 = {
#     'question': (
#         "A regional airport is evaluating the need to expand its terminal facilities. The airport management team wants to use data analytics to determine the optimal number of additional gates required. They have a comprehensive dataset of flight and passenger numbers over the past twenty years. However, there is a challenge: about 5% of the records are missing key information on the number of passengers per flight.\n\n"

#         "Question A1: The airport's chief strategist proposes the following approach: GIVEN the historical data of flights and passenger numbers, USE a certain statistical method TO impute the missing passenger data. Then, GIVEN the complete dataset, USE a forecasting model TO predict the number of flights and passengers for the next decade. Lastly, GIVEN these forecasts, USE a planning algorithm TO determine the minimum number of additional gates needed to handle the peak passenger traffic with a certain efficiency level. Identify and discuss the potential flaws or considerations in this proposed strategy, focusing on the choice of the statistical method for imputation, the forecasting model, and the planning algorithm.\n\n"
#         "A. The use of multiple regression for imputing missing passenger data may not accurately reflect the complexities of passenger behavior and flight patterns.\n\n"
#         "B. Forecasting the number of flights and passengers for the next decade using ARIMA might not account for unpredictable factors such as economic fluctuations or changes in travel habits.\n\n"
#         "C. Implementing k-nearest neighbors (KNN) for data imputation could lead to biases if the missing data is not randomly distributed.\n\n"
#         "D. Applying Monte Carlo simulation for planning the number of gates might not adequately consider the variability in daily flight schedules and passenger numbers.\n\n"
#     ),
#     'correct_answer': (
#         "A. The use of multiple regression may not accurately capture complex patterns of passenger behavior affecting the imputation accuracy.\n\n"
#         "B. ARIMA forecasts may not fully account for sudden changes in the market or external factors impacting flight and passenger predictions.\n\n"
#         "C. KNN for data imputation could introduce bias if the missing data is not random, potentially skewing the imputation process.\n\n"
#         "D. Monte Carlo simulation for planning gate numbers may not reflect the true variability in daily flight schedules and passenger numbers.\n\n"
#     ),
#     'explanation': (
#         "Multiple Regression may not capture the nonlinear patterns in passenger behavior, making it less ideal for imputing missing data accurately. ARIMA is adept at forecasting based on time series data but may not fully incorporate sudden market changes or external factors. KNN could introduce bias if the missing data correlates with certain unobserved variables, affecting the imputation accuracy. Monte Carlo Simulation provides a robust framework for understanding variability and uncertainty in planning but may require careful modeling of operational details to capture daily variations effectively.\n\n"
#     )
# }


q4 = {
    'question': (
        "A retail company is leveraging its extensive transactional and operational data to enhance various aspects of its business. "
        "The company has identified specific areas where analytical models could be applied for insights and optimization.\n\n"
        "For each of the following scenarios, select the most appropriate model from the options: **ARIMA, Louvain Algorithm, Integer Programming, "
        "SVM (Support Vector Machine), Neural Net, and Poisson.**\n\n"
        "**A.** Forecasting monthly customer foot traffic in stores nationwide, based on historical data showing trends and seasonality.\n\n"
        "**B.** Analyzing transaction data to identify clusters of products that are frequently purchased together across different regions.\n\n"
        "**C.** Optimizing the distribution of staff across different departments in a store to minimize customer wait times while considering budget constraints.\n\n"
        "**D.** Predicting the likelihood of a product being out of stock based on sales velocity, supply chain delays, and current inventory levels.\n\n"
        "**E.** Estimating the number of product returns next quarter based on return rates of past quarters, adjusting for recent changes in return policy."
    ),
    'correct_answer': (
        "**A.** ARIMA\n\n"
        "**B.** Louvain Algorithm\n\n"
        "**C.** Integer Programming\n\n"
        "**D.** SVM (Support Vector Machine)\n\n"
        "**E.** Poisson\n\n"
    ),
    'explanation': (
        "- **ARIMA** is suited for time series forecasting, making it ideal for predicting foot traffic with trends and seasonality.\n\n"
        "- **Louvain Algorithm** is effective for detecting communities within networks, suitable for finding product clusters.\n\n"
        "- **Integer Programming** is used for resource allocation problems, optimal for staff distribution.\n\n"
        "- **SVM** can classify data with high accuracy, perfect for predicting stock levels.\n\n"
        "- **Poisson** model is good for predicting the number of events happening in a fixed interval, making it suitable for estimating product returns."
    )
}

q5 = {
    'question': (
        "A marketing research firm is utilizing statistical models to analyze consumer data and market trends. For each of the scenarios below, "
        "decide whether **linear regression, logistic regression,** or **ridge regression** is the most suitable model based on the type of data and the prediction objective.\n\n"
        "1. Predicting the percentage increase in sales volume for a retail store next quarter based on advertising spend, seasonality, and economic indicators.\n\n"
        "2. Determining whether a new advertisement campaign will be successful (Yes/No) in increasing market share, based on historical campaign data and competitor activity.\n\n"
        "3. Estimating the impact of price changes on demand for a popular product, considering multicollinearity issues among predictors such as price, promotional activities, and competitor pricing strategies.\n\n"
        "4. Assessing the probability of a customer making a purchase based on their browsing history, age, income level, and previous purchase history."
    ),
    'correct_answer': (
        "1. Linear Regression\n\n"
        "2. Logistic Regression\n\n"
        "3. Ridge Regression\n\n"
        "4. Logistic Regression\n\n"
    ),
    'explanation': (
        "- **Linear Regression** is suited for continuous outcomes, making it appropriate for predicting sales volume percentage increases.\n\n"
        "- **Logistic Regression** is designed for binary outcomes, ideal for predicting the success of an advertisement campaign and assessing customer purchase probability.\n\n"
        "- **Ridge Regression** addresses multicollinearity among predictors, which is crucial for accurately estimating the impact of price changes on product demand."
    )
}

q6 = {
    'question': (
        "A regional airport is evaluating the need to expand its terminal facilities. The airport management team wants to use data analytics to determine "
        "the optimal number of additional gates required. They have a comprehensive dataset of flight and passenger numbers over the past twenty years. "
        "However, there is a challenge: about 5% of the records are missing key information on the number of passengers per flight.\n\n"
        "**Question A1**: The airport's chief strategist proposes the following approach:\n\n"
        "- **GIVEN** the historical data of flights and passenger numbers, **USE** a certain statistical method **TO** impute the missing passenger data.\n\n"
        "- **GIVEN** the complete dataset, **USE** a forecasting model **TO** predict the number of flights and passengers for the next decade.\n\n"
        "- **GIVEN** these forecasts, **USE** a planning algorithm **TO** determine the minimum number of additional gates needed to handle the peak passenger traffic with a certain efficiency level.\n\n"
        "Identify and discuss the potential flaws or considerations in this proposed strategy, focusing on the choice of the statistical method for imputation, the forecasting model, and the planning algorithm."
    ),
    'correct_answer': (
        "A. The use of multiple regression may not accurately capture complex patterns of passenger behavior affecting the imputation accuracy.\n\n"
        "B. ARIMA forecasts may not fully account for sudden changes in the market or external factors impacting flight and passenger predictions.\n\n"
        "C. KNN for data imputation could introduce bias if the missing data is not random, potentially skewing the imputation process.\n\n"
        "D. Monte Carlo simulation for planning gate numbers may not reflect the true variability in daily flight schedules and passenger numbers.\n\n"
    ),
    'explanation': (
        "- **Multiple Regression** may not capture the nonlinear patterns in passenger behavior, making it less ideal for imputing missing data accurately.\n\n"
        "- **ARIMA** is adept at forecasting based on time series data but may not fully incorporate sudden market changes or external factors.\n\n"
        "- **KNN** could introduce bias if the missing data correlates with certain unobserved variables, affecting the imputation accuracy.\n\n"
        "- **Monte Carlo Simulation** provides a robust framework for understanding variability and uncertainty in planning but may require careful modeling of operational details to capture daily variations effectively."
    )
}



# q7 = {
#     'question': (
#         "An urban transportation department is planning to enhance its public transit systems to improve reliability and manage peak-hour passenger flow more efficiently. The department has collected extensive data on route usage, passenger counts, and service delays.\n\n"
#         "a. To design an optimized transit schedule that balances route frequency with vehicle availability and expected passenger demand, which models/approaches should the department consider? Consider factors like vehicle capacity, route length, and peak travel times:\n"
#         "1. Linear Programming\n"
#         "2. Clustering\n"
#         "3. Dynamic Programming\n"
#         "4. Neural Networks\n"
#         "5. Integer Programming\n\n"
#         "b. To develop a flexible transit operation plan that can adjust to unexpected events such as road closures, severe weather, and varying passenger demand, which models/approaches would be most effective? This plan needs to anticipate and react to changes swiftly to minimize disruptions.\n"
#         "1. Stochastic Optimization\n"
#         "2. Support Vector Machines\n"
#         "3. Scenario Modeling\n"
#         "4. Convex Optimization\n"
#         "5. Decision Trees\n\n"
#     ),
#     'correct_answer': (
#         "a. Linear Programming, Dynamic Programming, Integer Programming\n\n"
#         "b. Stochastic Optimization, Scenario Modeling, Convex Optimization\n\n"
#     ),
#     'explanation': (
#         "For optimizing transit schedules, Linear Programming, Dynamic Programming, and Integer Programming are suitable as they can handle the complexities of resource allocation, scheduling, and optimization within predefined constraints effectively.\n\n"
#         "Stochastic Optimization and Convex Optimization, along with Scenario Modeling, are ideal for creating adaptable operation plans. They can model uncertainties and provide solutions that can adapt to changing conditions, which is crucial for managing unexpected events in urban transit systems.\n\n"
#     )
# }


# q8 = {
#     'question': "A multinational corporation is exploring innovative analytical methods to solve complex business problems across its operations. The corporation's research division is tasked with applying advanced models to address issues ranging from market analysis to operational efficiency and strategic planning.\n\nGiven the scenarios described below, select the most appropriate analytical approach from the options provided: Non-Parametric Methods, Bayesian Modeling, Community Detection in Graphs, and Deep Learning.\n\na. The corporation wants to analyze customer feedback from various sources to understand overall sentiment towards a new product line. The feedback is unstructured and varies widely in format and content.\n\nb. To improve supply chain resilience, the corporation seeks to identify closely knit clusters of suppliers and manufacturers based on transaction history, communication frequency, and geographical proximity.\n\nc. The marketing team is developing a campaign targeting specific customer segments. They have historical data on customer responses but face uncertainty about the effectiveness of new marketing channels.\n\nd. The HR department wants to predict employee turnover by analyzing factors such as job satisfaction, performance ratings, and team interactions. However, they are concerned about the potential non-linear relationships and interactions between these factors.\n\ne. The strategic planning team needs to forecast the demand for a new product in a market with little prior data. They wish to incorporate expert opinions and market analysis reports into their model.",
#     'correct_answer': (
#         "a. Deep Learning\n"
#         "b. Community Detection in Graphs\n"
#         "c. Bayesian Modeling\n"
#         "d. Non-Parametric Methods\n"
#         "e. Bayesian Modeling\n"
#     ),

#     'explanation': (
#         "a. Deep Learning is highly effective for processing and analyzing unstructured data, making it suitable for sentiment analysis of varied customer feedback.\n\n"
#         "b. Community Detection in Graphs, such as the Louvain Algorithm, can identify clusters within large networks, applicable for optimizing supply chain networks by highlighting tightly-knit groups.\n\n"
#         "c. Bayesian Modeling allows for incorporating prior knowledge and handling uncertainty, ideal for assessing new marketing strategies based on historical data and assumptions.\n\n"
#         "d. Non-Parametric Methods, like the Wilcoxon Signed-Rank Test or the Mann-Whitney Test, are useful for analyzing data without assuming a specific distribution, suitable for exploring complex relationships in HR data.\n\n"
#         "e. Bayesian Modeling's strength in integrating expert opinions and sparse data makes it a robust choice for demand forecasting in uncertain or new markets.\n"
#     )
# }



# q9 = {
#     'question': "Given the optimization scenarios below, classify each problem based on its most fitting optimization category. Consider the variables (x, y, etc.), known data (a, b, c, etc.), and note that all values of c are positive.\n\nChoices:\n- Linear Program\n- Convex Program\n- Convex Quadratic Program\n- General Nonconvex Program\n- Integer Program\n\nQuestion a:\nMinimize the sum of (c_i * x_i * y_i), subject to the sum of (a_ij * x_i) being equal to b_j for all j, x_i greater than or equal to 0, and y_i being either 0 or 1 for all i.\n\nQuestion b:\nMinimize the sum of (x_i^2 + y_i^2), subject to (a_i * x_i + b_i * y_i) less than or equal to c for all i, and x_i, y_i greater than or equal to 0.\n\nQuestion c:\nMaximize the product of (x_i^c_i), subject to the sum of (a_ij * x_i) less than or equal to b_j for all j, and x_i greater than or equal to 1.\n\nQuestion d:\nMaximize the sum of (sqrt(c_i) * x_i), subject to the sum of (a_ij * x_i) being equal to b_j for all j, x_i greater than or equal to 0.\n\nQuestion e:\nMinimize c^T x + d^T y, subject to A x + B y = b, x_i belonging to {0,1} for all i, and y_i greater than or equal to 0.",
#     'correct_answer': (
#         "a. Integer Program\n"
#         "b. Convex Quadratic Program\n"
#         "c. General Nonconvex Program\n"
#         "d. Convex Program\n"
#         "e. Integer Program"
#     ),

#     'explanation': (
#         "a. The presence of binary variables (y_i being either 0 or 1) along with linear constraints and a linear objective function where the variables are multiplied by binary variables indicates an Integer Program.\n\n"
#         "b. The objective function is the sum of squares, which is a classic form of a convex quadratic function, and the constraints are linear, making it a Convex Quadratic Program.\n\n"
#         "c. Maximizing the product of variables each raised to a power (where c_i > 0) creates a non-linear, non-convex objective function, indicating a General Nonconvex Program.\n\n"
#         "d. The objective function involves the square root of constants multiplied by variables, which is a convex function. Along with linear constraints, this forms a Convex Program.\n\n"
#         "e. The mixture of binary (x_i) and continuous variables (y_i), with linear constraints and a linear objective function, indicates an Integer Program, particularly a Mixed Integer Linear Program (MILP) due to the presence of both integer and continuous variables."
#     )
# }


q7 = {
    'question': (
        "An urban transportation department is planning to enhance its public transit systems to improve reliability and manage peak-hour passenger flow more efficiently. "
        "The department has collected extensive data on route usage, passenger counts, and service delays.\n\n"
        "a. To design an optimized transit schedule that balances route frequency with vehicle availability and expected passenger demand, which models/approaches should the department consider? "
        "Consider factors like vehicle capacity, route length, and peak travel times:\n"
        "1. Linear Programming\n"
        "2. Clustering\n"
        "3. Dynamic Programming\n"
        "4. Neural Networks\n"
        "5. Integer Programming\n\n"
        "b. To develop a flexible transit operation plan that can adjust to unexpected events such as road closures, severe weather, and varying passenger demand, which models/approaches would be most effective? "
        "This plan needs to anticipate and react to changes swiftly to minimize disruptions.\n"
        "1. Stochastic Optimization\n"
        "2. Support Vector Machines\n"
        "3. Scenario Modeling\n"
        "4. Convex Optimization\n"
        "5. Decision Trees"
    ),
    'correct_answer': (
        "a. Linear Programming, Dynamic Programming, Integer Programming\n\n"
        "b. Stochastic Optimization, Scenario Modeling, Convex Optimization"
    ),
    'explanation': (
        "For optimizing transit schedules, Linear Programming, Dynamic Programming, and Integer Programming are suitable as they can handle the complexities of resource allocation, scheduling, and optimization within predefined constraints effectively.\n\n"
        "Stochastic Optimization and Convex Optimization, along with Scenario Modeling, are ideal for creating adaptable operation plans. They can model uncertainties and provide solutions that can adapt to changing conditions, which is crucial for managing unexpected events in urban transit systems."
    )
}

q8 = {
    'question': (
        "A multinational corporation is exploring innovative analytical methods to solve complex business problems across its operations. "
        "The corporation's research division is tasked with applying advanced models to address issues ranging from market analysis to operational efficiency and strategic planning.\n\n"
        "Given the scenarios described below, select the most appropriate analytical approach from the options provided: Non-Parametric Methods, Bayesian Modeling, Community Detection in Graphs, and Deep Learning.\n\n"
        "a. The corporation wants to analyze customer feedback from various sources to understand overall sentiment towards a new product line. The feedback is unstructured and varies widely in format and content.\n\n"
        "b. To improve supply chain resilience, the corporation seeks to identify closely knit clusters of suppliers and manufacturers based on transaction history, communication frequency, and geographical proximity.\n\n"
        "c. The marketing team is developing a campaign targeting specific customer segments. They have historical data on customer responses but face uncertainty about the effectiveness of new marketing channels.\n\n"
        "d. The HR department wants to predict employee turnover by analyzing factors such as job satisfaction, performance ratings, and team interactions. However, they are concerned about the potential non-linear relationships and interactions between these factors.\n\n"
        "e. The strategic planning team needs to forecast the demand for a new product in a market with little prior data. They wish to incorporate expert opinions and market analysis reports into their model."
    ),
    'correct_answer': (
        "a. Deep Learning\n"
        "b. Community Detection in Graphs\n"
        "c. Bayesian Modeling\n"
        "d. Non-Parametric Methods\n"
        "e. Bayesian Modeling"
    ),
    'explanation': (
        "a. Deep Learning is highly effective for processing and analyzing unstructured data, making it suitable for sentiment analysis of varied customer feedback.\n\n"
        "b. Community Detection in Graphs, such as the Louvain Algorithm, can identify clusters within large networks, applicable for optimizing supply chain networks by highlighting tightly-knit groups.\n\n"
        "c. Bayesian Modeling allows for incorporating prior knowledge and handling uncertainty, ideal for assessing new marketing strategies based on historical data and assumptions.\n\n"
        "d. Non-Parametric Methods, like the Wilcoxon Signed-Rank Test or the Mann-Whitney Test, are useful for analyzing data without assuming a specific distribution, suitable for exploring complex relationships in HR data.\n\n"
        "e. Bayesian Modeling's strength in integrating expert opinions and sparse data makes it a robust choice for demand forecasting in uncertain or new markets."
    )
}

q9 = {
    'question': (
        "Given the optimization scenarios below, classify each problem based on its most fitting optimization category. Consider the variables $x$, $y$, etc., known data $a$, $b$, $c$, etc., and note that all values of $c$ are positive.\n\n"
        "Choices:\n- Linear Program\n- Convex Program\n- Convex Quadratic Program\n- General Nonconvex Program\n- Integer Program\n\n"
        "Question a:\nMinimize the sum of $(c_i \\cdot x_i \\cdot y_i)$, subject to the sum of $(a_{ij} \\cdot x_i)$ being equal to $b_j$ for all $j$, $x_i \\geq 0$, and $y_i$ being either 0 or 1 for all $i$.\n\n"
        "Question b:\nMinimize the sum of $(x_i^2 + y_i^2)$, subject to $(a_i \\cdot x_i + b_i \\cdot y_i) \\leq c$ for all $i$, and $x_i, y_i \\geq 0$.\n\n"
        "Question c:\nMaximize the product of $(x_i^{c_i})$, subject to the sum of $(a_{ij} \\cdot x_i) \\leq b_j$ for all $j$, and $x_i \\geq 1$.\n\n"
        "Question d:\nMaximize the sum of $(\\sqrt{c_i} \\cdot x_i)$, subject to the sum of $(a_{ij} \\cdot x_i)$ being equal to $b_j$ for all $j$, $x_i \\geq 0$.\n\n"
        "Question e:\nMinimize $c^T x + d^T y$, subject to $A x + B y = b$, $x_i \\in \\{0,1\\}$ for all $i$, and $y_i \\geq 0$."
    ),
    'correct_answer': (
        "a. Integer Program\n"
        "b. Convex Quadratic Program\n"
        "c. General Nonconvex Program\n"
        "d. Convex Program\n"
        "e. Integer Program"
    ),
    'explanation': (
        "a. The presence of binary variables $(y_i$ being either 0 or 1) along with linear constraints and a linear objective function where the variables are multiplied by binary variables indicates an Integer Program.\n\n"
        "b. The objective function is the sum of squares, which is a classic form of a convex quadratic function, and the constraints are linear, making it a Convex Quadratic Program.\n\n"
        "c. Maximizing the product of variables each raised to a power (where $c_i > 0$) creates a non-linear, non-convex objective function, indicating a General Nonconvex Program.\n\n"
        "d. The objective function involves the square root of constants multiplied by variables, which is a convex function. Along with linear constraints, this forms a Convex Program.\n\n"
        "e. The mixture of binary $(x_i)$ and continuous variables $(y_i)$, with linear constraints and a linear objective function, indicates an Integer Program, particularly a Mixed Integer Linear Program (MILP) due to the presence of both integer and continuous variables."
    )
}

# q10 = {
#     'question': (
#         "A tech startup is facing multiple challenges as it scales its operations, from analyzing customer feedback to optimizing its internal team dynamics and predicting market trends. "
#         "The data science team at the startup is considering employing various advanced analytical models to address these challenges. "
#         "Given the following scenarios, identify the most appropriate model or approach from the options: Non-Parametric Methods, Bayesian Modeling, Community Detection in Graphs, Neural Networks and Deep Learning, and Game Theory.\n\n"
#         "a. The startup has collected a vast amount of unstructured customer feedback through various channels. They want to understand the overall sentiment towards their product to inform future development. The data varies widely in format and content.\n"
#         "b. To enhance collaboration within the company, the team wants to analyze the communication patterns among employees. They have data on email interactions, meeting attendances, and project collaborations.\n"
#         "c. The product team is considering several new features based on market trends and internal data. However, there is significant uncertainty regarding the adoption of these features by their user base.\n"
#         "d. The company wants to create a predictive model for identifying potential churn customers based on their interaction with the product's online platform, including page views, feature usage, and support ticket submissions.\n"
#         "e. In response to emerging competitive threats, the startup needs to strategize its pricing model. This requires understanding how competitors might react to their pricing changes and the potential impact on market share."
#     ),
#     'correct_answer': (
#         "a. Neural Networks and Deep Learning\n"
#         "b. Community Detection in Graphs\n"
#         "c. Bayesian Modeling\n"
#         "d. Neural Networks and Deep Learning\n"
#         "e. Game Theory"
#     ),
#     'explanation': (
#         "a. Neural Networks and Deep Learning are highly effective for processing and analyzing unstructured data, making them suitable for sentiment analysis of varied customer feedback.\n"
#         "b. Community Detection in Graphs can identify clusters within large networks, such as communication patterns among employees, highlighting tightly-knit groups or isolated individuals.\n"
#         "c. Bayesian Modeling allows for incorporating prior knowledge and handling uncertainty, ideal for assessing the potential adoption of new features based on limited or uncertain data.\n"
#         "d. Neural Networks and Deep Learning excel at identifying complex patterns in high-dimensional data, suitable for predicting customer churn from varied interactions with an online platform.\n"
#         "e. Game Theory provides a framework for competitive decision-making, enabling the startup to anticipate competitors' reactions to pricing changes and strategize accordingly."
#     )
# }

# q11 = {
#     'question': (
#         "A financial analytics firm is analyzing historical stock market data to enhance its prediction models for future stock prices. The dataset includes daily closing prices, trading volume, and various economic indicators over the last 20 years. "
#         "However, the data exhibits heteroscedasticity, trends, and high dimensionality, making standard modeling approaches inadequate. "
#         "Given the scenarios described, identify the most appropriate model or approach from the options: Box-Cox Transformation, Detrending via linear regression, Principal Component Analysis (PCA), and Eigenvalues and Eigenvectors.\n\n"
#         "a. The firm notices that the variance in trading volumes increases significantly with higher volumes. To stabilize the variance across the dataset, which transformation should be applied, and what is the primary reason for its application?\n"
#         "b. Upon further analysis, a long-term upward trend in stock prices is evident, attributed to overall market growth. Before modeling, the firm wants to remove this trend to focus on underlying patterns. Which method should they use, and why?\n"
#         "c. The firm aims to reduce the number of economic indicators in the dataset without losing essential information due to the high correlation among indicators. Which technique is most suitable, and what is its primary benefit?\n"
#         "d. After applying PCA, the firm wants to understand the influence of the original economic indicators on the principal components. Which concept is crucial for this interpretation, and why?"
#     ),
#     'correct_answer': (
#         "a. Box-Cox Transformation\n"
#         "b. Detrending via linear regression\n"
#         "c. Principal Component Analysis (PCA)\n"
#         "d. Eigenvalues and Eigenvectors"
#     ),
#     'explanation': (
#         "a. The Box-Cox Transformation is applied to correct heteroscedasticity, a common issue in financial data, by making the variance constant across the range of data.\n"
#         "b. Detrending is essential for removing long-term trends, such as general market growth, to analyze the more nuanced fluctuations in stock prices that are of interest.\n"
#         "c. PCA is used for dimensionality reduction, particularly useful in datasets with many correlated variables, by transforming them into a smaller number of uncorrelated variables while retaining most of the original variance.\n"
#         "d. Eigenvectors from PCA provide insight into the contribution of each original variable to the principal components, helping interpret the reduced-dimensional space in terms of the original variables."
#     )
# }

# q12 = {
#     'question': (
#         "An e-commerce company is revamping its analytics framework to address several critical areas: sales forecasting, customer segmentation, marketing effectiveness, and operational efficiency. "
#         "The data science team plans to employ a range of optimization models to tackle these challenges effectively. "
#         "Given the scenarios below, identify the most suitable optimization model or approach for each, based on the descriptions provided in the document.\n\n"
#         "a. The company intends to forecast quarterly sales using historical sales data, which includes seasonal patterns and economic indicators. The model must accommodate fluctuating variance in sales volume.\n"
#         "b. To better understand its customer base, the company seeks to segment customers into distinct groups based on purchasing behavior, frequency, and preferences.\n"
#         "c. The marketing department wants to evaluate the impact of various advertising campaigns on sales. This requires a model that can handle both the campaigns' direct effects and their interactions.\n"
#         "d. With an aim to reduce shipping costs and delivery times, the company is analyzing its logistics and supply chain operations. The challenge includes balancing multiple factors, such as warehouse stock levels, transportation costs, and delivery routes.\n"
#         "e. Given the competitive nature of the e-commerce industry, the company also wants to optimize its pricing strategy to maximize profits while remaining attractive to customers, taking into account competitors' pricing and market demand elasticity."
#     ),
#     'correct_answer': (
#         "a. Time Series Model with Box-Cox Transformation\n"
#         "b. Clustering\n"
#         "c. Elastic Net\n"
#         "d. Linear/Quadratic Integer Program\n"
#         "e. Game Theory"
#     ),
#     'explanation': (
#         "a. The Box-Cox Transformation corrects heteroscedasticity, stabilizing variance in sales volume, making Time Series Models more effective.\n"
#         "b. Clustering optimizes the sum of distances from each data point to its cluster center, effectively segmenting customers based on various behaviors.\n"
#         "c. Elastic Net is ideal for models requiring variable selection and regularization to handle collinearity, perfect for marketing data with many predictors.\n"
#         "d. Logistics optimization involves binary decisions (e.g., whether to use a specific route), making it a case for Linear/Quadratic Integer Programming.\n"
#         "e. Game Theory models the strategic interactions between the company and its competitors, optimizing pricing strategies in a competitive environment."
#     )
# }

q10 = {
    'question': (
        "A tech startup is facing multiple challenges as it scales its operations, from analyzing customer feedback to optimizing its internal team dynamics and predicting market trends. "
        "The data science team at the startup is considering employing various advanced analytical models to address these challenges. "
        "Given the following scenarios, identify the most appropriate model or approach from the options: Non-Parametric Methods, Bayesian Modeling, Community Detection in Graphs, Neural Networks and Deep Learning, and Game Theory.\n\n"
        "a. The startup has collected a vast amount of unstructured customer feedback through various channels. They want to understand the overall sentiment towards their product to inform future development. The data varies widely in format and content.\n"
        "b. To enhance collaboration within the company, the team wants to analyze the communication patterns among employees. They have data on email interactions, meeting attendances, and project collaborations.\n"
        "c. The product team is considering several new features based on market trends and internal data. However, there is significant uncertainty regarding the adoption of these features by their user base.\n"
        "d. The company wants to create a predictive model for identifying potential churn customers based on their interaction with the product's online platform, including page views, feature usage, and support ticket submissions.\n"
        "e. In response to emerging competitive threats, the startup needs to strategize its pricing model. This requires understanding how competitors might react to their pricing changes and the potential impact on market share."
    ),
    'correct_answer': (
        "a. Neural Networks and Deep Learning\n"
        "b. Community Detection in Graphs\n"
        "c. Bayesian Modeling\n"
        "d. Neural Networks and Deep Learning\n"
        "e. Game Theory"
    ),
    'explanation': (
        "a. Neural Networks and Deep Learning are highly effective for processing and analyzing unstructured data, making them suitable for sentiment analysis of varied customer feedback.\n"
        "b. Community Detection in Graphs can identify clusters within large networks, such as communication patterns among employees, highlighting tightly-knit groups or isolated individuals.\n"
        "c. Bayesian Modeling allows for incorporating prior knowledge and handling uncertainty, ideal for assessing the potential adoption of new features based on limited or uncertain data.\n"
        "d. Neural Networks and Deep Learning excel at identifying complex patterns in high-dimensional data, suitable for predicting customer churn from varied interactions with an online platform.\n"
        "e. Game Theory provides a framework for competitive decision-making, enabling the startup to anticipate competitors' reactions to pricing changes and strategize accordingly."
    )
}

q11 = {
    'question': (
        "A financial analytics firm is analyzing historical stock market data to enhance its prediction models for future stock prices. The dataset includes daily closing prices, trading volume, and various economic indicators over the last 20 years. "
        "However, the data exhibits heteroscedasticity, trends, and high dimensionality, making standard modeling approaches inadequate. "
        "Given the scenarios described, identify the most appropriate model or approach from the options: Box-Cox Transformation, Detrending via linear regression, Principal Component Analysis (PCA), and Eigenvalues and Eigenvectors.\n\n"
        "a. The firm notices that the variance in trading volumes increases significantly with higher volumes. To stabilize the variance across the dataset, which transformation should be applied, and what is the primary reason for its application?\n"
        "b. Upon further analysis, a long-term upward trend in stock prices is evident, attributed to overall market growth. Before modeling, the firm wants to remove this trend to focus on underlying patterns. Which method should they use, and why?\n"
        "c. The firm aims to reduce the number of economic indicators in the dataset without losing essential information due to the high correlation among indicators. Which technique is most suitable, and what is its primary benefit?\n"
        "d. After applying PCA, the firm wants to understand the influence of the original economic indicators on the principal components. Which concept is crucial for this interpretation, and why?"
    ),
    'correct_answer': (
        "a. Box-Cox Transformation\n"
        "b. Detrending via linear regression\n"
        "c. Principal Component Analysis (PCA)\n"
        "d. Eigenvalues and Eigenvectors"
    ),
    'explanation': (
        "a. The Box-Cox Transformation is applied to correct heteroscedasticity, a common issue in financial data, by making the variance constant across the range of data.\n"
        "b. Detrending is essential for removing long-term trends, such as general market growth, to analyze the more nuanced fluctuations in stock prices that are of interest.\n"
        "c. PCA is used for dimensionality reduction, particularly useful in datasets with many correlated variables, by transforming them into a smaller number of uncorrelated variables while retaining most of the original variance.\n"
        "d. Eigenvectors from PCA provide insight into the contribution of each original variable to the principal components, helping interpret the reduced-dimensional space in terms of the original variables."
    )
}

q12 = {
    'question': (
        "An e-commerce company is revamping its analytics framework to address several critical areas: sales forecasting, customer segmentation, marketing effectiveness, and operational efficiency. "
        "The data science team plans to employ a range of optimization models to tackle these challenges effectively. "
        "Given the scenarios below, identify the most suitable optimization model or approach for each, based on the descriptions provided in the document.\n\n"
        "a. The company intends to forecast quarterly sales using historical sales data, which includes seasonal patterns and economic indicators. The model must accommodate fluctuating variance in sales volume.\n"
        "b. To better understand its customer base, the company seeks to segment customers into distinct groups based on purchasing behavior, frequency, and preferences.\n"
        "c. The marketing department wants to evaluate the impact of various advertising campaigns on sales. This requires a model that can handle both the campaigns' direct effects and their interactions.\n"
        "d. With an aim to reduce shipping costs and delivery times, the company is analyzing its logistics and supply chain operations. The challenge includes balancing multiple factors, such as warehouse stock levels, transportation costs, and delivery routes.\n"
        "e. Given the competitive nature of the e-commerce industry, the company also wants to optimize its pricing strategy to maximize profits while remaining attractive to customers, taking into account competitors' pricing and market demand elasticity."
    ),
    'correct_answer': (
        "a. Time Series Model with Box-Cox Transformation\n"
        "b. Clustering\n"
        "c. Elastic Net\n"
        "d. Linear/Quadratic Integer Program\n"
        "e. Game Theory"
    ),
    'explanation': (
        "a. The Box-Cox Transformation corrects heteroscedasticity, stabilizing variance in sales volume, making Time Series Models more effective.\n"
        "b. Clustering optimizes the sum of distances from each data point to its cluster center, effectively segmenting customers based on various behaviors.\n"
        "c. Elastic Net is ideal for models requiring variable selection and regularization to handle collinearity, perfect for marketing data with many predictors.\n"
        "d. Logistics optimization involves binary decisions (e.g., whether to use a specific route), making it a case for Linear/Quadratic Integer Programming.\n"
        "e. Game Theory models the strategic interactions between the company and its competitors, optimizing pricing strategies in a competitive environment."
    )
}



q13 = {
    'question': (
        "A logistics company is analyzing its package sorting system to improve efficiency and reduce bottlenecks during peak operation hours. "
        "The system involves sorting packages based on destination, size, and priority, with a complex network of conveyor belts and automated sorting machines. "
        "The company has collected data on package arrival times, sorting times, and system throughput. The data science team intends to use probabilistic models and simulation to address several challenges.\n\n"
        
        "**a. To model the arrival of packages to the sorting facility, which probabilistic distribution should be applied, considering that arrivals are continuous and independent?**\n"
        
        "**b. The company observes that the time taken by a sorting machine to process a package varies, but it generally follows a pattern. Which model best describes the processing time of packages, given the requirement for a memoryless property?**\n"
        
        "**c. Given the complexity of the sorting system and the variability in package arrivals and processing times, which simulation approach would best allow the company to analyze system performance and identify potential improvements?**\n"
        
        "**d. The management is considering implementing a new queueing strategy to prioritize urgent packages. They want to estimate the impact of this change on average waiting times and system throughput. How can simulation be utilized to assess the effectiveness of this new strategy?**\n"
        
        "**e. Finally, the company wants to explore the long-term effects of adding an additional sorting machine to the system. Which method would allow them to model the system's states and transitions over time, considering the probabilistic nature of package arrivals and processing times?**"
    ),
    'correct_answer': (
        "a. Poisson Distribution\n"
        "b. Exponential Distribution\n"
        "c. Discrete Event Stochastic Simulation\n"
        "d. Prescriptive Simulation\n"
        "e. Markov Chain Model"
    ),
    'explanation': (
        "a. The Poisson Distribution is appropriate for modeling the arrival of packages because it deals with the number of events (package arrivals) that occur in a fixed interval of time and space, assuming these events occur with a known constant mean rate and independently of the time since the last event.\n"
        "b. The Exponential Distribution, known for its memoryless property, is suitable for modeling the time between events in a Poisson process, making it ideal for describing the sorting time of packages.\n"
        "c. Discrete Event Stochastic Simulation allows for modeling the complex interactions within the sorting system, including the randomness of package arrivals and processing times, to analyze performance and identify bottlenecks.\n"
        "d. Prescriptive Simulation can be used to simulate the effect of different queueing strategies on the system, allowing the company to assess changes before implementation based on various metrics such as average waiting times and system throughput.\n"
        "e. The Markov Chain Model is effective for modeling the system's states and transitions over time, providing insights into the long-term effects of operational changes such as adding an additional sorting machine."
    )
}

q14 = {
    'question': (
        "An agricultural technology company is developing a predictive model to optimize crop yields across various regions. "
        "The project involves integrating diverse datasets, including soil properties, weather patterns, crop genetics, and historical yield data. "
        "The analytics team is exploring several modeling techniques to address the challenges presented by the variability and complexity of the data.\n\n"
        
        "**a. Given the variability in weather patterns and their significant impact on crop yields, which modeling approach would best accommodate the uncertainty inherent in weather forecasts?**\n"
        
        "**b. Considering the need to classify regions based on soil properties that directly influence crop choice and management practices, which technique is most suitable for segmenting the regions into distinct categories?**\n"
        
        "**c. To account for the non-linear relationship between soil moisture levels and crop yield, which method should be applied to capture the complexity of this relationship?**\n"
        
        "**d. In evaluating the effectiveness of different fertilizers on crop yield, how should the company approach the analysis to determine the most impactful factors?**\n"
        
        "**e. The company aims to develop a model that predicts the likelihood of a pest outbreak based on historical data and current observations. Which model would allow for a dynamic assessment of risk over time?**"
    ),
    'correct_answer': (
        "a. Bayesian Modeling\n"
        "b. Clustering\n"
        "c. Neural Networks\n"
        "d. Elastic Net\n"
        "e. Markov Chains"
    ),
    'explanation': (
        "a. Bayesian Modeling is well-suited for incorporating uncertainty and making probabilistic predictions, ideal for weather-related forecasts.\n"
        "b. Clustering is an effective unsupervised learning technique for categorizing regions with similar soil properties without predefined labels.\n"
        "c. Neural Networks can model complex, non-linear relationships, making them suitable for understanding the nuanced effects of soil moisture on crop yield.\n"
        "d. Elastic Net combines Lasso and Ridge regression techniques, offering a balanced approach to variable selection and regularization, crucial for identifying the key factors influencing crop yield.\n"
        "e. Markov Chains are effective for modeling state transitions over time, providing a robust framework for predicting pest outbreaks based on changing conditions."
    )
}

q15 = {
    'question': (
        "A healthcare research institute is conducting a longitudinal study on the impact of lifestyle factors on chronic diseases. "
        "The dataset encompasses a wide range of variables, including dietary habits, physical activity levels, genetic predispositions, and health outcomes over ten years. "
        "However, the institute faces challenges with missing data across various time points and variables.\n\n"
        
        "**a. Dietary habit data is sporadically missing for some participants over the ten-year period, with no apparent pattern to the missingness. Which imputation method would best handle this scenario without introducing significant bias?**\n"
        
        "**b. Physical activity levels are consistently missing for a subgroup of participants who dropped out after the first year. The researchers suspect this dropout is not random but related to the participants' health conditions. How should the missing data be treated to avoid bias in the study's conclusions?**\n"
        
        "**c. Genetic predisposition data is missing for 2% of the participants due to technical issues with genetic testing. The missingness is considered completely random. What is the most straightforward imputation method that could be applied in this case?**\n"
        
        "**d. The study plans to analyze the correlation between lifestyle factors and the onset of chronic diseases. However, health outcome data is missing for 5% of the observations. The missing health outcomes are mostly from the earlier part of the study. Which approach should the institute use to impute this missing data, considering the importance of temporal patterns?**\n"
        
        "**e. Finally, the researchers want to ensure that their imputation strategy for missing dietary and physical activity data does not distort the study's findings. What validation technique could they employ to assess the effectiveness of their imputation methods?**"
    ),
    'correct_answer': (
        "a. Multiple Imputation\n"
        "b. Use a model-based approach, such as a mixed-effects model, to account for non-random dropout\n"
        "c. Mean/Median/Mode Imputation\n"
        "d. Time Series Imputation\n"
        "e. Cross-validation using a subset of complete cases to simulate missingness and then comparing imputed values against actual values"
    ),
    'explanation': (
        "a. Multiple Imputation is suitable for handling sporadic missing data across time, as it accounts for the uncertainty around the missing data and generates several imputed datasets for analysis.\n"
        "b. Model-based approaches like mixed-effects models can handle missing data due to dropout by incorporating random effects that account for the variability among participants and fixed effects for observed variables.\n"
        "c. Mean/Median/Mode Imputation is a straightforward method for handling completely random missing data, using the most central tendency measure that best fits the distribution of the data.\n"
        "d. Time Series Imputation is appropriate for data with temporal patterns, as it can use trends and seasonal components to estimate missing values more accurately.\n"
        "e. Cross-validation is a robust technique for validating imputation effectiveness, allowing researchers to simulate missingness in known cases and assess how well the imputation method recovers the original data."
    )
}





midterm_1_q25 = {
    'question': (
        "A marketing agency is segmenting its audience for targeted advertising campaigns.\n\n"
        
        "**a. For creating customer segments based on shopping behavior and preferences, which clustering method would be most suitable?**\n"
        "- K-means Clustering\n"
        "- KNN Clustering\n"
        "- PCA\n"
        "- Poisson Variance Classification\n\n"
        
        "A retail chain is analyzing factors affecting its sales performance.\n\n"
        
        "**a. To predict future sales based on factors like store location, advertising spend, and local demographics, which regression method should be employed?**\n"
        "- Linear Regression\n"
        "- Poisson Regression\n"
        "- Bayesian Regression\n"
        "- Lasso Regression\n\n"
        
        "**b. The retailer needs to understand the relationship between temperature and outdoor sales. If the relationship is non-linear, what should they consider in their regression model?**\n"
        "- Transformation and Interaction Terms\n"
        "- Logistic Regression\n"
        "- Polynomial Regression\n"
        "- Ridge Regression."
    ),
    'correct_answer': (
        "a. K-means Clustering\n"
        "a. Linear Regression\n"
        "b. Polynomial Regression"
    ),
    'explanation': (
        "a. K-means Clustering is commonly used in market segmentation due to its simplicity and efficiency in clustering similar items based on predefined characteristics.\n"
        "a. Linear Regression is suitable for predicting future sales based on several independent variables, offering a straightforward approach to regression analysis.\n"
        "b. Polynomial Regression helps capture non-linear relationships, providing flexibility when modeling sales trends influenced by varying temperatures."
    )
}

midterm_1_q24 = {
    'question': (
        "A financial institution is implementing a new system to classify loan applicants based on risk.\n\n"
        
        "**a. Which classifier would be more effective for categorizing applicants into 'high risk' and 'low risk', considering the cost of misclassification?**\n"
        "- Linear Regression\n"
        "- K-Nearest Neighbor (KNN)\n"
        "- Support Vector Machine (SVM)\n"
        "- Random Forest\n\n"
        
        "**b. In a scenario where the bank needs to identify potential fraudulent transactions, which approach should they use, given the transactions data is highly imbalanced?**\n"
        "- Hard Classifiers\n"
        "- Soft Classifiers\n"
        "- Decision Trees\n"
        "- Bayesian Classifiers\n\n"
        
        "An e-commerce company is evaluating different models for predicting customer purchase behavior.\n\n"
        
        "**a. To ensure the chosen model is not overfitting, which method should be used for validating the model's effectiveness?**\n"
        "- Cross-Validation\n"
        "- Training on Entire Dataset\n"
        "- AIC/BIC Comparison\n"
        "- Holdout Method\n\n"
        
        "**b. If the model performs well on the training data but poorly on the validation data, what might this indicate?**\n"
        "- The model is underfitting\n"
        "- The model is overfitting\n"
        "- The model is perfectly fitted\n"
        "- The model is not complex enough."
    ),
    'correct_answer': (
        "a. Support Vector Machine (SVM)\n"
        "b. Soft Classifiers\n"
        "a. Cross-Validation\n"
        "b. The model is overfitting"
    ),
    'explanation': (
        "a. Support Vector Machine (SVM) is effective for binary classification, particularly when the cost of misclassification is high. It creates a hyperplane that best separates the two categories, minimizing classification errors.\n"
        "b. Soft Classifiers, which include techniques like probability thresholds, are suitable for highly imbalanced datasets, allowing for a more flexible classification approach.\n"
        "a. Cross-Validation provides a reliable method for validating a model's effectiveness by testing it across different subsets of the data to ensure robustness.\n"
        "b. Overfitting occurs when the model is too closely fitted to the training data, resulting in poor performance on validation data."
    )
}



midterm_1_q23 = {
    'question': (
        "A retailer wants to segment their customer base for targeted marketing."
        "\n\n"
        "a. Which clustering method would be best for a dataset with well-defined, separate customer groups?"
        "\n"
        "K-means Clustering\n"
        "PCA\n\n"
        "b. In analyzing customer purchase data, the team identifies several extreme values. What is the most appropriate initial step in handling these outliers?"
        "\n"
        "Removing them from the dataset\n"
        "Investigating their source and context\n\n"
        "A utility company is analyzing electricity usage patterns over time."
        "\n\n"
        "a. To forecast future usage that exhibits both trend and seasonality, which method would be most appropriate?"
        "\n"
        "ARIMA\n"
        "Exponential Smoothing with trend and seasonality\n\n"
        "b. If the company wants to smooth out short-term fluctuations in daily usage data while giving more weight to recent observations, what should be the approach to setting the alpha value in exponential smoothing?"
        "\n"
        "A high alpha value\n"
        "A low alpha value"
    ),
    'correct_answer': (
        "a. K-means Clustering\n\n"
        "b. Investigating their source and context\n\n"
        "a. Exponential Smoothing with trend and seasonality\n\n"
        "b. A high alpha value"
    ),
    'explanation': (
        "a. K-means Clustering is ideal for segmenting datasets with distinct groups, due to its ability to create clear clusters.\n\n"
        "b. Investigating the source and context of outliers is crucial before deciding whether to remove them, as they may indicate important data trends or errors.\n\n"
        "a. Exponential Smoothing with trend and seasonality effectively captures both trend and seasonality, making it suitable for forecasting electricity usage patterns over time.\n\n"
        "b. A high alpha value in exponential smoothing gives more weight to recent observations, making it ideal for smoothing out short-term fluctuations in daily data."
    )
}

midterm_1_q21 = {
    'question': (
        "A healthcare analytics team is working on various models to analyze patient data for improving treatment outcomes. They have collected extensive patient data over the years, including demographics, treatment details, and health outcomes."
        "\n\n"
        "a. For classifying patients into high-risk and low-risk categories based on their treatment outcomes, which model would be best suited?"
        "\n"
        "Cusum\n"
        "K-Nearest Neighbors\n"
        "Support Vector Machines (SVM)\n\n"
        "b. To cluster patients based on similarities in their diagnosis and treatment types, which algorithm would be most effective?"
        "\n"
        "K-Means Clustering\n"
        "PCA\n"
        "GARCH Variance Clustering\n\n"
        "The healthcare analytics team is also interested in predicting the efficacy of treatments over time."
        "\n\n"
        "a. If the team wants to forecast treatment efficacy based on past trends and seasonal variations, which model should they use?"
        "\n"
        "ARIMA\n"
        "Exponential Smoothing\n"
        "Random Forests\n\n"
        "b. To detect significant changes in treatment efficacy over time, which method would be most suitable?"
        "\n"
        "CUSUM\n"
        "Principal Component Analysis\n"
        "Box-Cox Transformation"
    ),
    'correct_answer': (
        "a. Support Vector Machines (SVM)\n\n"
        "b. K-Means Clustering\n\n"
        "a. Exponential Smoothing\n\n"
        "b. CUSUM"
    ),
    'explanation': (
        "a. Support Vector Machines (SVM) are effective for classifying patients into distinct categories based on a clear decision boundary, ideal for high-risk and low-risk segmentation.\n\n"
        "b. K-Means Clustering is widely used for creating clusters with well-defined groupings, making it suitable for segmenting patients based on diagnosis and treatment types.\n\n"
        "a. Exponential Smoothing is ideal for forecasting trends with seasonal components, providing a smooth approach to predicting treatment efficacy.\n\n"
        "b. CUSUM (Cumulative Sum) is a robust method for detecting significant changes or shifts in data over time, well-suited for monitoring treatment efficacy."
    )
}

midterm_1_q22 = {
    'question': (
        "A bank is developing a model to classify loan applicants as high-risk or low-risk, with the goal of minimizing misclassification."
        "\n\n"
        "a. Which model would be more suitable for this task, considering the importance of minimizing the misclassification of high-risk applicants?"
        "\n"
        "Support Vector Machines (SVM)\n"
        "K-Nearest Neighbors (KNN)\n\n"
        "b. In a medical diagnosis system, which model would be preferable for classifying patients based on a dataset with many overlapping characteristics?"
        "\n"
        "Support Vector Machines (SVM)\n"
        "K-Nearest Neighbors (KNN)\n\n"
        "A marketing team has developed several predictive models for customer behavior."
        "\n\n"
        "a. To avoid overfitting, which approach should they use for model assessment?"
        "\n"
        "Cross-validation\n"
        "Training on the entire dataset\n\n"
        "b. When choosing between two different models for predicting sales, one with a lower AIC and one with a higher BIC, which model should be preferred considering both simplicity and likelihood?"
        "\n"
        "Model with lower AIC\n"
        "Model with higher BIC"
    ),
    'correct_answer': (
        "a. Support Vector Machines (SVM)\n\n"
        "b. Support Vector Machines (SVM)\n\n"
        "a. Cross-validation\n\n"
        "b. Model with lower AIC"
    ),
    'explanation': (
        "a. Support Vector Machines (SVM) offer a robust decision boundary that is effective for binary classification tasks where the cost of misclassification is high, making it suitable for loan risk assessment.\n\n"
        "b. SVMs are also preferred in a medical diagnosis system with overlapping characteristics due to their ability to handle complex decision boundaries.\n\n"
        "a. Cross-validation helps avoid overfitting by testing the model on different subsets of data, providing a comprehensive validation approach.\n\n"
        "b. A model with a lower AIC (Akaike Information Criterion) is generally preferred because it balances simplicity and likelihood, reducing the risk of overfitting."
    )
}

q2217 = {
    'question': (
        "A research team is conducting a study on the effects of dietary habits on long-term health outcomes. They've collected a dataset over 10 years, tracking individuals' consumption of fruits, vegetables, processed foods, and their health outcomes, including cholesterol levels, blood pressure, and incidence of heart disease.\n\n"
        
        "Given the dataset and research goals, answer the following questions:\n\n"
        
        "**a. Given the objective to analyze the impact of each dietary habit on cholesterol levels, which regression model should the team use, and why?**\n"
        
        "**b. The team notices that blood pressure readings have a non-linear relationship with processed food consumption. How should they adjust their regression model to account for this?**\n"
        
        "**c. To predict the likelihood of heart disease based on dietary habits, what type of regression model is most appropriate, and what is the primary reason for its selection?**\n"
        
        "**d. The team wants to control for the variable 'age' while analyzing the effects of dietary habits on health outcomes. How should they incorporate 'age' into their regression analysis?**\n"
        
        "**e. After initial analysis, the team realizes that multicollinearity between fruit and vegetable consumption might be skewing their results. What strategy should they consider to address this issue?**"
    ),
    'correct_answer': (
        "a. Multiple linear regression, because it allows for the assessment of the impact of multiple predictors (dietary habits) on a continuous outcome (cholesterol levels).\n"
        "b. Introduce polynomial terms for processed food consumption into the model to capture the non-linear relationship.\n"
        "c. Logistic regression, as it is designed for binary outcomes (e.g., the presence or absence of heart disease) and can handle predictors like dietary habits.\n"
        "d. Include 'age' as a covariate in the regression model to adjust for its potential confounding effect on the relationship between dietary habits and health outcomes.\n"
        "e. Apply Ridge or Lasso regression techniques to penalize the regression coefficients, thereby reducing the effect of multicollinearity and selecting the most relevant predictors."
    ),
    'explanation': (
        "a. Multiple linear regression is suitable for modeling the relationship between multiple independent variables and a single continuous dependent variable, making it ideal for assessing the impacts of various dietary habits on cholesterol levels.\n"
        "b. Polynomial regression allows for the modeling of non-linear relationships by adding polynomial terms, which can better fit the observed data when linear terms are insufficient.\n"
        "c. Logistic regression is used when the dependent variable is categorical, such as predicting a binary outcome. It models the probability that the outcome is present given the predictors.\n"
        "d. Including 'age' as a covariate helps to adjust for its effects, providing a clearer picture of how dietary habits alone influence health outcomes, without the confounding impact of age.\n"
        "e. Ridge and Lasso regression are regularization methods that address multicollinearity by penalizing the size of coefficients, which can help in selecting the most significant variables when predictors are correlated."
    )
}

q2218 = {
    'question': (
        "A city's public transportation system is evaluating its bus service efficiency. The system operates with varying arrival rates throughout the day due to peak and off-peak hours. During peak hours (30% of the time), buses arrive at a rate of 10 buses/hour. During off-peak hours, buses arrive at a rate of 4 buses/hour. The average boarding time per passenger is consistently 1 minute.\n\n"
        
        "Given this information, answer the following questions regarding the application of queuing theory and stochastic modeling to improve service efficiency:\n\n"
        
        "**a. Initially, the transportation system models bus arrivals with 15 buses running during peak hours and 6 buses during off-peak hours. What would you expect the model to show regarding passenger wait times?**\n"
        "- Wait times are minimal at both peak and off-peak hours.\n"
        "- Wait times are minimal at peak hours and longer at off-peak hours.\n"
        "- Wait times are longer at peak hours and minimal at off-peak hours.\n"
        "- Wait times are long at both peak and off-peak hours.\n\n"
        
        "**b. After analyzing passenger feedback on wait times, the system experiments with dynamic scheduling, adjusting the number of buses to 12 during peak hours and 8 during off-peak hours. What would you expect the model to show under this new scheduling approach?**\n"
        "- Wait times are reduced at both peak and off-peak hours compared to the initial model.\n"
        "- Wait times are reduced at peak hours but increased at off-peak hours compared to the initial model.\n"
        "- Wait times are increased at peak hours but reduced at off-peak hours compared to the initial model.\n"
        "- Wait times remain unchanged at both peak and off-peak hours compared to the initial model.\n\n"
        
        "**c. Considering the introduction of a real-time tracking and notification system, which statement best describes its expected impact on the modeling of the bus service?**\n"
        "- It introduces variability that makes the queuing model less predictable.\n"
        "- It reduces the need for a queuing model by directly managing passenger expectations.\n"
        "- It enhances the accuracy of the queuing model by providing real-time data for better decision-making.\n"
        "- It has no significant impact on the queuing model as it does not affect bus arrival rates or boarding times."
    ),
    'correct_answer': (
        "a. Wait times are longer at peak hours and minimal at off-peak hours.\n"
        "b. Wait times are reduced at both peak and off-peak hours compared to the initial model.\n"
        "c. It enhances the accuracy of the queuing model by providing real-time data for better decision-making."
    ),
    'explanation': (
        "a. Given the higher arrival rate of buses during peak hours in the initial model, passenger wait times are expected to be longer due to the increased demand. The model's setup during off-peak hours should minimize wait times due to a better balance between bus arrivals and passenger demand.\n"
        "b. The adjusted number of buses during both peak and off-peak hours aims to more closely match the arrival rate with passenger demand, potentially reducing wait times across the board by optimizing resources.\n"
        "c. A real-time tracking and notification system provides passengers with current wait times and bus arrival information, allowing for more informed decision-making. For the transportation system, this real-time data can inform adjustments to bus frequencies, enhancing the queuing model's relevance and accuracy by incorporating current demand and traffic conditions."
    )
}

q2219 = {
    'question': (
        "A data science team at a tech company is analyzing user interaction data with their software to improve user experience and engagement. The dataset includes daily active users (DAU), average session time, number of sessions per user, feature usage frequency, and user retention rate. The team wants to use regression analysis to predict user retention based on these variables.\n\n"
        
        "**a. Considering the team's objective to predict user retention rate, a continuous outcome variable, which regression model is most appropriate for initial analysis?**\n"
        "- Simple linear regression\n"
        "- Multiple linear regression\n"
        "- Logistic regression\n\n"
        
        "**b. The team observes a potential non-linear relationship between average session time and user retention rate. Which method could effectively capture this non-linearity in the regression model?**\n"
        "- Adding polynomial terms for average session time\n"
        "- Transforming the retention rate using a log function\n"
        "- Using a logistic regression model\n\n"
        
        "**c. To identify which features are most predictive of user retention while avoiding overfitting with too many variables, which technique should the team employ?**\n"
        "- Stepwise regression\n"
        "- Ridge regression\n"
        "- Principal Component Analysis (PCA)\n\n"
        
        "**d. After developing the regression model, the team wants to evaluate its performance in predicting user retention. Which metric is most suitable for assessing the model's accuracy?**\n"
        "- R-squared\n"
        "- Mean Squared Error (MSE)\n"
        "- Area Under the ROC Curve (AUC)\n\n"
        
        "**e. The team plans to segment users based on their likelihood of retention, as predicted by the regression model. For users classified as at risk of low retention, targeted engagement strategies will be implemented. Which approach allows the team to classify users based on predicted retention rates?**\n"
        "- K-means clustering on predicted retention rates\n"
        "- Setting a threshold on the predicted retention rate to classify users\n"
        "- Using a logistic regression model for classification"
    ),
    'correct_answer': (
        "a. Multiple linear regression\n"
        "b. Adding polynomial terms for average session time\n"
        "c. Stepwise regression\n"
        "d. Mean Squared Error (MSE)\n"
        "e. Setting a threshold on the predicted retention rate to classify users"
    ),
    'explanation': (
        "a. Multiple linear regression is appropriate for modeling the relationship between multiple predictors and a continuous outcome variable.\n"
        "b. Adding polynomial terms can help model non-linear relationships between predictors and the outcome variable within a linear regression framework.\n"
        "c. Stepwise regression is a method of adding or removing variables from the model based on their statistical significance, which helps in selecting the most predictive features while avoiding overfitting.\n"
        "d. Mean Squared Error (MSE) measures the average of the squares of the errors between the predicted and actual values, making it a suitable metric for evaluating the accuracy of regression models.\n"
        "e. Setting a threshold on the predicted retention rate allows for the classification of users into different categories (e.g., high risk vs. low risk of churn) based on their predicted likelihood of retention."
    )
}


q2220 = {
    'question': (
        "A pharmaceutical company is developing a new drug and aims to identify the key factors affecting its efficacy. The dataset includes numerous potential predictor variables such as dosage levels, patient age, weight, genetic markers, previous treatment history, and several biochemical indicators. The response variable is the measured effectiveness of the drug in reducing symptoms on a continuous scale.\n"
        "a. The research team wants to build a predictive model while selecting the most relevant variables and reducing the impact of multicollinearity among predictors. Which method should they use?\n"
        "- Forward Selection\n"
        "- Backward Elimination\n"
        "- Ridge Regression\n"
        "- Lasso Regression\n"
        "b. After applying the method chosen in (a), the team notices that some of the coefficients are exactly zero. What does this indicate?\n"
        "- The variables associated with these coefficients are not significant and have been effectively removed from the model.\n"
        "- There is perfect multicollinearity among the predictors.\n"
        "- The model has overfitted the data.\n"
        "- The response variable does not vary with changes in these predictors.\n"
        "c. To balance between selecting variables and reducing the magnitude of coefficients, the team decides to use a method that combines both Lasso and Ridge Regression penalties. Which method is appropriate?\n"
        "- Forward Selection\n"
        "- Backward Elimination\n"
        "- Elastic Net\n"
        "- Stepwise Regression\n"
        "d. The team needs to determine the optimal values for the penalty parameters in the chosen method from (c). What is the best approach to select these parameters?\n"
        "- Use cross-validation to test different parameter values and choose those that minimize the prediction error.\n"
        "- Choose arbitrary values based on prior studies.\n"
        "- Set the penalty parameters to zero to eliminate their effect.\n"
        "- Use the method's default parameter values without modification.\n"
        "e. After building the final model, the team wants to assess its predictive performance on new data. Which metric would be most appropriate for evaluating the model's accuracy in predicting a continuous outcome?\n"
        "- Mean Absolute Error (MAE)\n"
        "- R-squared\n"
        "- Area Under the ROC Curve (AUC)\n"
        "- Confusion Matrix"
    ),
    'correct_answer': (
        "a. Lasso Regression\n"
        "b. The variables associated with these coefficients are not significant and have been effectively removed from the model.\n"
        "c. Elastic Net\n"
        "d. Use cross-validation to test different parameter values and choose those that minimize the prediction error.\n"
        "e. Mean Absolute Error (MAE)"
    ),
    'explanation': (
        "a. **Lasso Regression** adds a penalty proportional to the absolute value of the coefficients, which can shrink some coefficients to exactly zero, effectively performing variable selection while handling multicollinearity.\n"
        "b. When using **Lasso Regression**, coefficients of less important variables can become exactly zero, indicating these variables do not contribute significantly and have been effectively removed from the model.\n"
        "c. **Elastic Net** combines both Lasso and Ridge penalties, balancing variable selection and coefficient shrinkage, making it suitable for handling correlated predictors while selecting variables.\n"
        "d. **Using cross-validation** allows the team to test different penalty parameters and select those that minimize prediction error, ensuring the model generalizes well to new data.\n"
        "e. **Mean Absolute Error (MAE)** measures the average magnitude of errors in predictions of continuous outcomes, making it appropriate for evaluating the model's predictive accuracy."
    )
}

q2133 = {
    'question': (
        "An e-commerce company wants to optimize its website to increase sales conversion rates. They plan to test different combinations of website designs, including variations in color schemes, layouts, and call-to-action (CTA) button placements. Due to resource constraints, they cannot test every possible combination.\n"
        "a. Which experimental design method should they use to efficiently test multiple factors and their interactions?\n"
        "- A/B Testing\n"
        "- Full Factorial Design\n"
        "- Fractional Factorial Design\n"
        "- Multi-Armed Bandit Algorithm\n"
        "b. During the experiment, they want to ensure that external variables like time of day and day of the week do not confound the results. What technique should they employ?\n"
        "- Randomization\n"
        "- Blocking\n"
        "- Replication\n"
        "- Stratification\n"
        "c. After collecting data, they need to analyze the effects of each factor and their interactions on the conversion rate. Which statistical method is most appropriate?\n"
        "- ANOVA (Analysis of Variance)\n"
        "- Regression Analysis\n"
        "- t-test\n"
        "- Chi-Squared Test\n"
        "d. Suppose they have three factors (color scheme, layout, CTA placement), each at two levels. How many experimental conditions are there in a full factorial design?\n"
        "- 6\n"
        "- 8\n"
        "- 9\n"
        "- 12\n"
        "e. To maximize conversions while continuously learning from user interactions, which method should the company implement that balances exploration and exploitation?\n"
        "- A/B Testing\n"
        "- Full Factorial Design\n"
        "- Multi-Armed Bandit Algorithm\n"
        "- Stepwise Regression"
    ),
    'correct_answer': (
        "a. Fractional Factorial Design\n"
        "b. Blocking\n"
        "c. ANOVA (Analysis of Variance)\n"
        "d. 8\n"
        "e. Multi-Armed Bandit Algorithm"
    ),
    'explanation': (
        "a. **Fractional Factorial Design** allows the company to test a subset of all possible combinations efficiently, studying multiple factors and their interactions without testing every combination.\n"
        "b. **Blocking** controls for external variables by grouping experimental units with similar characteristics, reducing variability due to these factors.\n"
        "c. **ANOVA (Analysis of Variance)** is appropriate for analyzing the effects of categorical factors and their interactions on a continuous outcome like conversion rate.\n"
        "d. In a full factorial design with three factors at two levels each, the number of experimental conditions is **2^3 = 8**.\n"
        "e. The **Multi-Armed Bandit Algorithm** balances exploration of new options and exploitation of known ones, optimizing conversions while continuously learning from user interactions."
    )
}

ques23tion_1 = {
    'question': (
        "A healthcare analytics team is building a predictive model to identify patients at high risk for certain diseases. They have collected numerous potential predictor variables, including patient demographics, lifestyle factors, medical history, and lab results. However, they need to carefully select variables to improve the model’s performance and interpretability.\n\n"
        "a. The team decides to begin with a model with no predictors and add variables one at a time based on their statistical significance. Which variable selection method is best suited for this approach?\n"
        "- Forward Selection\n"
        "- Backward Elimination\n"
        "- Stepwise Regression\n"
        "- Ridge Regression\n\n"
        "b. After initial testing, they want a method that combines adding and removing variables based on significance to ensure that only meaningful predictors remain in the model. Which method would be the most effective?\n"
        "- Forward Selection\n"
        "- Backward Elimination\n"
        "- Stepwise Regression\n"
        "- Elastic Net\n\n"
        "c. The team is concerned about multicollinearity in the data, especially among highly correlated lab results. They choose a method that penalizes the coefficients but retains all variables. Which method is appropriate for this purpose?\n"
        "- Lasso Regression\n"
        "- Elastic Net\n"
        "- Ridge Regression\n"
        "- Stepwise Regression\n\n"
        "d. For final model selection, the team decides to use a technique that sets some coefficients exactly to zero, which can improve interpretability by selecting only the most relevant predictors. Which method aligns with this goal?\n"
        "- Forward Selection\n"
        "- Ridge Regression\n"
        "- Lasso Regression\n"
        "- Elastic Net"
    ),
    'correct_answer': "a. Forward Selection, b. Stepwise Regression, c. Ridge Regression, d. Lasso Regression",
    'explanation': (
        "a. Forward Selection starts with no variables and adds them one at a time based on their significance.\n\n"
        "b. Stepwise Regression combines Forward Selection and Backward Elimination, allowing for more dynamic variable selection.\n\n"
        "c. Ridge Regression imposes a penalty on coefficient size, which reduces multicollinearity without eliminating variables.\n\n"
        "d. Lasso Regression can set some coefficients exactly to zero, effectively performing variable selection."
    )
}

questi2on_2 = {
    'question': (
        "A product development team at a tech company is testing new features for a mobile app. They want to determine the most impactful features to improve user engagement while minimizing development time and costs.\n\n"
        "a. The team has two variations of a new feature they want to test to see which version results in higher engagement. Which experimental approach should they use?\n"
        "- Multi-Armed Bandit\n"
        "- Factorial Design\n"
        "- A/B Testing\n"
        "- Blocking\n\n"
        "b. They want to test multiple features at once and understand the interaction effects between them, such as how a notification feature might work differently depending on the presence of a recommendation feature. Which experimental design is most suitable?\n"
        "- A/B Testing\n"
        "- Multi-Armed Bandit\n"
        "- Factorial Design\n"
        "- Forward Selection\n\n"
        "c. The team decides to implement an algorithm that allocates resources dynamically among different feature versions to maximize user engagement over time, rather than conducting a static test. Which approach should they choose?\n"
        "- Multi-Armed Bandit\n"
        "- Blocking\n"
        "- Ridge Regression\n"
        "- Stepwise Regression\n\n"
        "d. During analysis, they need to control for external factors like time of day and user demographics that could affect engagement. Which DOE technique is useful for accounting for these sources of variability?\n"
        "- Blocking\n"
        "- Forward Selection\n"
        "- A/B Testing\n"
        "- Lasso Regression"
    ),
    'correct_answer': "a. A/B Testing, b. Factorial Design, c. Multi-Armed Bandit, d. Blocking",
    'explanation': (
        "a. A/B Testing is ideal for comparing two versions of a single feature to determine which performs better.\n\n"
        "b. Factorial Design allows for testing multiple features and interactions simultaneously, which is useful for assessing complex feature interactions.\n\n"
        "c. Multi-Armed Bandit algorithms adaptively allocate resources based on observed performance, optimizing engagement dynamically.\n\n"
        "d. Blocking controls for variability from external factors by isolating their influence, making the results more reliable."
    )
}

q1uestion_3 = {
    'question': (
        "A retail company is developing a model to predict customer purchase behavior based on various factors, such as browsing history, demographics, and purchasing frequency. They are exploring different variable selection techniques to balance model complexity with predictive accuracy.\n\n"
        "a. The team begins with a model that includes all factors and systematically removes the least significant ones until only significant predictors remain. Which method is appropriate for this process?\n"
        "- Forward Selection\n"
        "- Backward Elimination\n"
        "- Stepwise Regression\n"
        "- Elastic Net\n\n"
        "b. They are interested in using a technique that not only reduces the number of predictors but also controls for potential multicollinearity among them. Which technique should they use?\n"
        "- Lasso Regression\n"
        "- Forward Selection\n"
        "- Backward Elimination\n"
        "- Ridge Regression\n\n"
        "c. For a model that requires both variable selection and control over large coefficient values, which combined approach would best fit the company’s needs?\n"
        "- Forward Selection\n"
        "- Backward Elimination\n"
        "- Elastic Net\n"
        "- Ridge Regression\n\n"
        "d. If the team wants a method that selects only the most relevant variables by setting some coefficients exactly to zero, which method would they choose?\n"
        "- Ridge Regression\n"
        "- Lasso Regression\n"
        "- Elastic Net\n"
        "- Backward Elimination"
    ),
    'correct_answer': "a. Backward Elimination, b. Ridge Regression, c. Elastic Net, d. Lasso Regression",
    'explanation': (
        "a. Backward Elimination starts with all variables and removes those that are least significant until all remaining variables are significant.\n\n"
        "b. Ridge Regression controls multicollinearity by shrinking coefficients without eliminating variables.\n\n"
        "c. Elastic Net combines Lasso and Ridge penalties, providing both variable selection and coefficient shrinkage.\n\n"
        "d. Lasso Regression sets some coefficients to zero, performing effective variable selection."
    )
}

q1uestion_4 = {
    'question': (
        "A marketing team is planning to launch a series of digital ads and wants to measure which ad variations are most effective at driving conversions, as well as the impact of ad timing on customer engagement.\n\n"
        "a. The team wants to test multiple ad elements, such as visuals and messaging, and understand how they interact to affect conversions. Which DOE method would allow them to analyze these interactions?\n"
        "- A/B Testing\n"
        "- Factorial Design\n"
        "- Blocking\n"
        "- Multi-Armed Bandit\n\n"
        "b. To improve conversions over time, they want an adaptive approach that favors the best-performing ad elements as more data is collected. Which approach is best suited for this dynamic testing?\n"
        "- Factorial Design\n"
        "- A/B Testing\n"
        "- Blocking\n"
        "- Multi-Armed Bandit\n\n"
        "c. The team needs to control for potential confounding factors, such as user demographics and the time of day the ad is shown, to ensure that results reflect the ad elements’ effectiveness rather than external factors. Which method is appropriate?\n"
        "- Blocking\n"
        "- A/B Testing\n"
        "- Factorial Design\n"
        "- Elastic Net"
    ),
    'correct_answer': "a. Factorial Design, b. Multi-Armed Bandit, c. Blocking",
    'explanation': (
        "a. Factorial Design allows testing multiple factors and their interactions, making it ideal for assessing ad element combinations.\n\n"
        "b. Multi-Armed Bandit optimizes resource allocation dynamically based on ongoing performance, favoring high-performing ads.\n\n"
        "c. Blocking controls for confounding variables by grouping observations based on these factors, leading to more reliable results."
    )
}


ques22tion_5 = {
    'question': (
        "An e-commerce platform is analyzing data on customer purchases to improve recommendations. They are using probability-based models to better understand purchasing patterns. "
        "a. The team wants to model the likelihood that a customer makes a purchase after browsing a product page. Which distribution is most appropriate if they want to capture events with only two possible outcomes (purchase or no purchase)? "
        "- Binomial Distribution "
        "- Poisson Distribution "
        "- Geometric Distribution "
        "- Bernoulli Distribution "
        
        "b. To predict the number of purchases in a fixed time interval, such as per hour, which distribution should they choose? "
        "- Binomial Distribution "
        "- Poisson Distribution "
        "- Exponential Distribution "
        "- Weibull Distribution "
        
        "c. If they wish to model the time between consecutive purchases on their website, which distribution best suits this purpose? "
        "- Binomial Distribution "
        "- Geometric Distribution "
        "- Exponential Distribution "
        "- Weibull Distribution "
        
        "d. The team suspects that the number of clicks needed before a customer makes a purchase follows a distribution where they are counting trials until the first success. Which distribution applies here? "
        "- Binomial Distribution "
        "- Poisson Distribution "
        "- Geometric Distribution "
        "- Exponential Distribution"
    ),
    'correct_answer': "a. Bernoulli Distribution, b. Poisson Distribution, c. Exponential Distribution, d. Geometric Distribution",
    'explanation': (
        "a. The Bernoulli Distribution models binary outcomes, such as purchase or no purchase. "
        "b. The Poisson Distribution models the count of events in a fixed interval, suitable for predicting purchase frequency. "
        "c. The Exponential Distribution is used for modeling time intervals between consecutive events. "
        "d. The Geometric Distribution is ideal for modeling the number of trials until the first success."
    )
}

quest22ion_6 = {
    'question': (
        "A government health survey collects data from thousands of participants each year to analyze health trends, but they encounter missing data issues. "
        "a. The team notices missing data for some questions that participants tended to skip. Which approach might be suitable if they aim to retain as much data as possible without introducing significant bias? "
        "- Discarding Data Points "
        "- Using Categorical Variables "
        "- Imputation Using Midrange Values "
        "- Perturbation "
        
        "b. For missing income data, they want to impute a representative value. Which imputation method is most appropriate if they wish to avoid skewing the dataset with outliers? "
        "- Mean Imputation "
        "- Median Imputation "
        "- Mode Imputation "
        "- Regression Imputation "
        
        "c. The team wants to avoid overfitting in their imputation model, especially when predicting values based on other variables. Which imputation approach should they consider? "
        "- Perturbation "
        "- Mean Imputation "
        "- Regression Imputation "
        "- Discarding Data Points "
        
        "d. For analysis reliability, they aim to minimize imputation errors. What general recommendation should they follow regarding the extent of data imputation? "
        "- Impute no more than 5% of missing data for each factor "
        "- Impute all missing values to ensure completeness "
        "- Only impute missing values if they are below 10% of the dataset "
        "- Use mean imputation across all factors"
    ),
    'correct_answer': "a. Using Categorical Variables, b. Median Imputation, c. Perturbation, d. Impute no more than 5% of missing data for each factor",
    'explanation': (
        "a. Using categorical variables for missing data helps indicate where data is missing while preserving as much information as possible. "
        "b. Median Imputation is less affected by outliers than mean imputation, making it suitable for missing income data. "
        "c. Perturbation adds variability to imputed values, reducing the risk of overfitting by ensuring that the imputations reflect the data's spread. "
        "d. Limiting imputation to no more than 5% per factor minimizes the risk of introducing imputation errors."
    )
}

questi22on_7 = {
    'question': (
        "A logistics company is optimizing its delivery routes to reduce costs and improve efficiency. They are using optimization models to address complex decision-making. "
        "a. The team wants a model that can handle yes/no decisions for route selection. Which type of optimization model should they use? "
        "- Linear Program "
        "- Integer Program "
        "- Convex Program "
        "- Logistic Regression "
        
        "b. To minimize fuel costs, which part of the optimization model should represent this objective? "
        "- Constraints "
        "- Decision Variables "
        "- Objective Function "
        "- Solution "
        
        "c. If the company needs to consider uncertain traffic conditions in its routing decisions, which approach would be most appropriate? "
        "- Conservative Modeling "
        "- Binary Variables "
        "- Integer Program "
        "- Linear Program "
        
        "d. The company decides to analyze scenarios where they optimize routes across different potential traffic scenarios. What strategy should they use to account for multiple possible outcomes? "
        "- Scenario Modeling "
        "- Dynamic Programming "
        "- Beliman’s Equation "
        "- Blocking"
    ),
    'correct_answer': "a. Integer Program, b. Objective Function, c. Conservative Modeling, d. Scenario Modeling",
    'explanation': (
        "a. Integer Programs are used for decisions that are binary, such as selecting or not selecting a route. "
        "b. The Objective Function represents the goal of the optimization, which is to minimize fuel costs in this case. "
        "c. Conservative Modeling allows for uncertainty by adding safety margins to constraints. "
        "d. Scenario Modeling optimizes across multiple scenarios, which is useful when conditions like traffic vary."
    )
}

qu2estion_8 = {
    'question': (
        "A manufacturing plant is implementing queuing theory to optimize its production line, where delays are affecting efficiency. "
        "a. The team wants to model the queue of items waiting for assembly and identify factors affecting wait times. Which key element of queuing theory should they focus on? "
        "- Arrival Rate "
        "- Transition Matrix "
        "- Agent-Based Models "
        "- Memorylessness "
        
        "b. To determine if the time between item arrivals follows a Poisson process, which distribution should they use to model this? "
        "- Bernoulli Distribution "
        "- Exponential Distribution "
        "- Weibull Distribution "
        "- Binomial Distribution "
        
        "c. If the team notices a pattern in the data suggesting that wait times may not fit a Poisson process, what type of plot would help verify this assumption? "
        "- Q-Q Plot "
        "- Scatter Plot "
        "- Line Graph "
        "- Histogram "
        
        "d. The plant considers simulating different production scenarios to make more informed decisions about queue management. Which simulation type is best for analyzing discrete events like item arrivals and assembly? "
        "- Discrete Event Simulation "
        "- Continuous Simulation "
        "- Agent-Based Simulation "
        "- Time Series Analysis"
    ),
    'correct_answer': "a. Arrival Rate, b. Exponential Distribution, c. Q-Q Plot, d. Discrete Event Simulation",
    'explanation': (
        "a. Arrival Rate is a fundamental element in queuing theory, affecting wait times and queue length. "
        "b. The Exponential Distribution models the time between events in a Poisson process, which is appropriate for item arrivals. "
        "c. Q-Q Plots are used to assess whether data follows a specified distribution, helpful in verifying Poisson assumptions. "
        "d. Discrete Event Simulation models distinct events, making it suitable for analyzing production processes with item arrivals and assembly."
    )
}

ques1tion_9 = {
    'question': (
        "A tech company is creating a recommendation system and wants to model the probability of a user interacting with content using Markov Chains. "
        "a. If they assume each content interaction depends only on the last viewed content, which characteristic of Markov Chains does this assumption illustrate? "
        "- Memorylessness "
        "- Arrival Rate "
        "- Objective Function "
        "- Perturbation "
        
        "b. They aim to represent different content types (e.g., videos, articles, ads) as states in the Markov Chain. What tool should they use to capture the probabilities of moving from one content type to another? "
        "- Transition Matrix "
        "- Q-Q Plot "
        "- Exponential Distribution "
        "- Blocking "
        
        "c. If they notice that users frequently revisit the same content type, which property of Markov Chains could help model this repeated behavior? "
        "- Steady-State Probabilities "
        "- Discrete Event Simulation "
        "- Binary Variables "
        "- Constraints "
        
        "d. To analyze long-term behavior, the team wants to calculate the probabilities of users staying on certain content types over time. Which Markov Chain concept should they use? "
        "- Steady-State Probabilities "
        "- Memorylessness "
        "- Arrival Rate "
        "- Poisson Distribution"
    ),
    'correct_answer': "a. Memorylessness, b. Transition Matrix, c. Steady-State Probabilities, d. Steady-State Probabilities",
    'explanation': (
        "a. Memorylessness indicates that each state depends only on the current state, aligning with the assumption of content interaction based on the last viewed content. "
        "b. A Transition Matrix represents the probabilities of moving from one state to another in a Markov Chain. "
        "c. Steady-State Probabilities help model the likelihood of users staying or returning to the same content type over time. "
        "d. Steady-State Probabilities are used to understand long-term behaviors in Markov Chains, such as user interactions with specific content types."
    )
}

q22r = {
    'question': (
        "A call center wants to model the number of incoming calls they receive per hour to optimize staffing levels. They have observed that on average, they receive 15 calls per hour. They assume that the arrival of calls follows a Poisson process. "
        "a. What is the probability that exactly 10 calls will arrive in a given hour? "
        "- Use the Binomial distribution with $n=15$ and $p=0.5$ "
        "- Use the Poisson distribution with $\\lambda=15$ to calculate $P(X=10)$ "
        "- Use the Exponential distribution with $\\lambda=15$ to calculate $P(X=10)$ "
        "- Use the Normal distribution with $\\mu=15$ and $\\sigma^2=15$ to calculate $P(X=10)$ "
        
        "b. To model the time between incoming calls, which probability distribution should the call center use? "
        "- Uniform distribution "
        "- Exponential distribution "
        "- Poisson distribution "
        "- Weibull distribution "
        
        "c. If the call center wants to check if the arrival of calls follows the assumed distribution, which graphical method should they employ? "
        "- Histogram "
        "- Scatter plot "
        "- Q-Q plot "
        "- Box plot "
        
        "d. The call center manager wants to simulate the call arrivals over a typical 8-hour shift. Which simulation method is most appropriate? "
        "- Discrete event simulation "
        "- Agent-based simulation "
        "- Continuous simulation "
        "- Monte Carlo simulation "
        
        "e. To model the future state of the system where the next state depends only on the current state (number of calls in the current hour), which model should be used? "
        "- Queuing theory model "
        "- Markov chain model "
        "- Regression model "
        "- Time series ARIMA model"
    ),
    'correct_answer': (
        "a. Use the Poisson distribution with $\\lambda=15$ to calculate $P(X=10)$ "
        "b. Exponential distribution "
        "c. Q-Q plot "
        "d. Discrete event simulation "
        "e. Markov chain model"
    ),
    'explanation': (
        "a. The number of calls per hour can be modeled using the Poisson distribution with $\\lambda=15$. To find the probability of exactly 10 calls, use $P(X=10)$ with the Poisson distribution. "
        "b. The time between events in a Poisson process follows an Exponential distribution. So, to model the time between incoming calls, the Exponential distribution is appropriate. "
        "c. A Q-Q plot can be used to check if the observed data follows a particular theoretical distribution by plotting the quantiles of the data against the quantiles of the theoretical distribution. "
        "d. Discrete event simulation is suitable for simulating events that occur at discrete points in time, such as call arrivals in a call center. "
        "e. A Markov chain model is appropriate when the next state depends only on the current state (memorylessness), fitting scenarios where future call volumes depend only on the current state."
    )
}

q23r = {
    'question': (
        "A national health survey collects data on various health indicators from participants across the country. Due to the sensitive nature of some questions, a significant portion of participants did not report their income levels. The dataset includes variables such as age, gender, height, weight, blood pressure, cholesterol levels, smoking status, and income. "
        "a. The analysts decide not to use any imputation method and exclude records with missing income data from the analysis. What is a potential drawback of this approach? "
        "- It simplifies the analysis without any significant impact. "
        "- It can lead to biased results due to loss of data and non-random missingness. "
        "- It ensures that the remaining data is of higher quality. "
        "- It violates the privacy of participants. "
        
        "b. Instead of excluding records, the analysts consider using the mean income to impute the missing values. Which limitation is associated with this imputation method? "
        "- It is computationally intensive. "
        "- It introduces significant bias by underestimating the variability of income. "
        "- It is not applicable for continuous variables. "
        "- It requires advanced statistical software. "
        
        "c. To better estimate the missing income values, the analysts decide to use other variables in the dataset to predict income. Which method are they employing? "
        "- Midrange value imputation "
        "- Predictive model imputation (e.g., regression) "
        "- Deleting the missing data "
        "- Using categorical variables "
        
        "d. The analysts are concerned about imputing more than 5% of the data for the income variable. What is a reason for this concern? "
        "- Imputing over 5% of data is against statistical regulations. "
        "- High imputation rates can introduce significant errors and bias into the analysis. "
        "- Imputing data is unethical. "
        "- The software cannot handle imputation beyond 5%. "
        
        "e. Which of the following is a method that does not require imputation and can utilize all available data? "
        "- Mean substitution "
        "- Multiple imputation "
        "- Using a categorical variable to indicate missingness "
        "- Regression imputation"
    ),
    'correct_answer': (
        "a. It can lead to biased results due to loss of data and non-random missingness. "
        "b. It introduces significant bias by underestimating the variability of income. "
        "c. Predictive model imputation (e.g., regression) "
        "d. High imputation rates can introduce significant errors and bias into the analysis. "
        "e. Using a categorical variable to indicate missingness"
    ),
    'explanation': (
        "a. Excluding records with missing data can lead to biased results, especially if the missingness is not completely random. This approach results in loss of valuable data and may not represent the population accurately. "
        "b. Using the mean to impute missing values can underestimate the variability and spread of the income data, leading to biased parameter estimates and weakening statistical inferences. "
        "c. Using other variables to predict income for imputation is called predictive model imputation or regression imputation. This method can reduce bias compared to simpler imputation methods. "
        "d. Imputing more than 5% of data can introduce significant errors and bias, making the results less reliable. It is generally advised to limit imputation to a small proportion of the dataset. "
        "e. Using a categorical variable to indicate missingness allows analysts to include all available data without imputing missing values, helping to identify patterns associated with missing data."
    )
}

q24r = {
    'question': (
        "A manufacturing company needs to determine the optimal production levels for its two products, A and B, to maximize profits. Each unit of product A yields a profit of $50, and each unit of product B yields a profit of $40. The production of each product requires time on two machines: Machine 1 and Machine 2. Product A requires 2 hours on Machine 1 and 1 hour on Machine 2. Product B requires 1 hour on Machine 1 and 2 hours on Machine 2. Machine 1 is available for 100 hours, and Machine 2 is available for 80 hours.\n\n"
        "**a. Formulate the objective function for maximizing profit:**\n"
        "- Maximize Profit = $50A + 40B$\n"
        "- Maximize Profit = $40A + 50B$\n"
        "- Maximize Profit = $2A + B$\n"
        "- Maximize Profit = $A + B$\n\n"
        
        "**b. Which of the following constraints correctly represents the availability of Machine 1?**\n"
        "- $2A + 1B \\leq 100$\n"
        "- $2A + 1B \\geq 100$\n"
        "- $1A + 2B \\leq 100$\n"
        "- $1A + 2B \\leq 80$\n\n"
        
        "**c. What type of optimization problem is this?**\n"
        "- Linear programming problem\n"
        "- Non-linear programming problem\n"
        "- Integer programming problem\n"
        "- Dynamic programming problem\n\n"
        
        "**d. If the company wants to produce whole units of products (no fractions), which modification should be made to the model?**\n"
        "- Add integer constraints to A and B\n"
        "- Use a non-linear objective function\n"
        "- Increase the profit coefficients\n"
        "- Remove the time constraints\n\n"
        
        "**e. If the company is uncertain about the availability of Machine 2 and wants to ensure the solution is feasible under different scenarios, which approach should they take?**\n"
        "- Use conservative modeling by adding a margin of error to the Machine 2 constraint\n"
        "- Ignore the Machine 2 constraint\n"
        "- Double the production of product B\n"
        "- Use a Markov chain model"
    ),
    'correct_answer': (
        "a. Maximize Profit = $50A + 40B$\n"
        "b. $2A + 1B \\leq 100$\n"
        "c. Linear programming problem\n"
        "d. Add integer constraints to A and B\n"
        "e. Use conservative modeling by adding a margin of error to the Machine 2 constraint"
    ),
    'explanation': (
        "a. The objective is to maximize profit, which is calculated as Profit = $50A + 40B$, where A and B are the quantities of products A and B.\n\n"
        "b. The constraint for Machine 1 is $2A + 1B \\leq 100$, representing that the total hours used by products A and B cannot exceed the available 100 hours.\n\n"
        "c. This is a linear programming problem because both the objective function and the constraints are linear.\n\n"
        "d. To ensure that only whole units are produced, integer constraints need to be added to the variables A and B, making it an integer linear programming problem.\n\n"
        "e. To account for uncertainty in Machine 2's availability, conservative modeling can be used by adding a margin of error to the constraint, ensuring the solution remains feasible under different scenarios."
    )
}

q2r5 = {
    'question': (
        "A data scientist is developing a predictive model using a large dataset with numerous features. She is concerned about overfitting and wants to perform variable selection while handling multicollinearity among predictors.\n\n"
        
        "**a. Which regression method should she use to achieve variable selection by shrinking some coefficients exactly to zero?**\n"
        "- Ridge Regression\n"
        "- Lasso Regression\n"
        "- Elastic Net\n"
        "- Principal Component Regression\n\n"
        
        "**b. She decides to use Elastic Net Regression to balance between variable selection and coefficient shrinkage. Which penalties does Elastic Net combine?**\n"
        "- L1 penalty only\n"
        "- L2 penalty only\n"
        "- Both L1 and L2 penalties\n"
        "- Neither L1 nor L2 penalties\n\n"
        
        "**c. To find the optimal hyperparameters for her Elastic Net model, such as the mixing parameter and regularization strength, which method should she employ?**\n"
        "- Use default parameters provided by software\n"
        "- Cross-validation\n"
        "- Grid search without validation\n"
        "- Randomly select parameters\n\n"
        
        "**d. She is considering using a Support Vector Machine (SVM) for classification tasks. What is the main objective of an SVM in classification problems?**\n"
        "- Minimize the number of misclassifications\n"
        "- Maximize the margin between classes\n"
        "- Reduce the dimensionality of the data\n"
        "- Cluster the data into groups\n\n"
        
        "**e. To enable the SVM to handle non-linear relationships in the data, which technique should she apply?**\n"
        "- Use a linear kernel\n"
        "- Increase the regularization parameter\n"
        "- Use a kernel trick with a non-linear kernel (e.g., RBF kernel)\n"
        "- Standardize the features"
    ),
    'correct_answer': (
        "a. Lasso Regression\n"
        "b. Both L1 and L2 penalties\n"
        "c. Cross-validation\n"
        "d. Maximize the margin between classes\n"
        "e. Use a kernel trick with a non-linear kernel (e.g., RBF kernel)"
    ),
    'explanation': (
        "a. **Lasso Regression** uses an L1 penalty that can shrink some coefficients to exactly zero, effectively performing variable selection while handling multicollinearity.\n\n"
        "b. **Elastic Net** combines both **L1 (Lasso)** and **L2 (Ridge)** penalties, balancing variable selection and coefficient shrinkage.\n\n"
        "c. **Cross-validation** is used to determine optimal hyperparameters by evaluating model performance on different subsets of the data, ensuring the model generalizes well.\n\n"
        "d. The main objective of an **SVM** in classification is to **maximize the margin** between different classes, finding the optimal hyperplane that best separates the classes.\n\n"
        "e. Applying the **kernel trick with a non-linear kernel** (such as the RBF kernel) allows the SVM to handle non-linear relationships by mapping data into a higher-dimensional space."
    )
}

q26r = {
    'question': (
        "A researcher is analyzing paired sample data where the assumption of normality is violated. She wants to test if there is a significant difference between the two related samples.\n\n"
        
        "**a. Which non-parametric test should she use?**\n"
        "- McNemar's Test\n"
        "- Wilcoxon Signed-Rank Test\n"
        "- Mann-Whitney U Test\n"
        "- Paired t-test\n\n"
        
        "**b. Another researcher wants to compare two independent samples to see if they come from the same distribution, without assuming normality. Which test is appropriate?**\n"
        "- McNemar's Test\n"
        "- Wilcoxon Signed-Rank Test\n"
        "- Mann-Whitney U Test\n"
        "- Independent t-test\n\n"
        
        "**c. In a study involving categorical data with paired observations, which test is suitable for detecting changes or differences in proportions?**\n"
        "- Chi-Squared Test\n"
        "- McNemar's Test\n"
        "- Fisher's Exact Test\n"
        "- ANOVA\n\n"
        
        "**d. What is a key advantage of using non-parametric tests over parametric tests?**\n"
        "- They are more powerful when data is normally distributed\n"
        "- They make fewer assumptions about the data distribution\n"
        "- They are easier to interpret\n"
        "- They require larger sample sizes\n\n"
        
        "**e. Non-parametric tests are especially useful when dealing with which type of data?**\n"
        "- Normally distributed data\n"
        "- Ordinal data or data with outliers\n"
        "- Interval data without outliers\n"
        "- Nominal data only"
    ),
    'correct_answer': (
        "a. Wilcoxon Signed-Rank Test\n"
        "b. Mann-Whitney U Test\n"
        "c. McNemar's Test\n"
        "d. They make fewer assumptions about the data distribution\n"
        "e. Ordinal data or data with outliers"
    ),
    'explanation': (
        "a. The **Wilcoxon Signed-Rank Test** is a non-parametric test used for comparing two related samples when the data cannot be assumed to be normally distributed.\n\n"
        "b. The **Mann-Whitney U Test** is appropriate for comparing two independent samples without assuming normality, testing whether they come from the same distribution.\n\n"
        "c. **McNemar's Test** is used for paired nominal data to detect differences in proportions on a dichotomous trait.\n\n"
        "d. A key advantage of non-parametric tests is that **they make fewer assumptions** about the data, particularly regarding the data's distribution.\n\n"
        "e. Non-parametric tests are useful for **ordinal data or data with outliers**, where parametric test assumptions (like normality) are violated."
    )
}

qr27 = {
    'question': (
        "A company is analyzing its network of customers to identify communities for targeted marketing. They have data representing customers as nodes and their interactions as edges in a graph.\n\n"
        
        "**a. Which algorithm is suitable for detecting communities within this large network?**\n"
        "- K-means clustering\n"
        "- Louvain Algorithm\n"
        "- Apriori Algorithm\n"
        "- Support Vector Machine\n\n"
        
        "**b. What is the primary goal of the Louvain Algorithm in network analysis?**\n"
        "- To minimize the distance between nodes\n"
        "- To maximize the modularity of partitions\n"
        "- To sort nodes based on centrality\n"
        "- To predict future connections\n\n"
        
        "**c. In the context of network graphs, what does a high modularity score indicate?**\n"
        "- Strong community structure with dense connections within communities and sparse connections between them\n"
        "- Weak community structure with random connections\n"
        "- That the network is fully connected\n"
        "- That there are no communities present\n\n"
        
        "**d. Which of the following is a key advantage of using the Louvain Algorithm?**\n"
        "- It requires labeled data for training\n"
        "- It is efficient and scalable for large networks\n"
        "- It guarantees finding the global maximum modularity\n"
        "- It only works on small datasets\n\n"
        
        "**e. After detecting communities, the company wants to visualize the network. Which tool or method can help in visualizing complex networks?**\n"
        "- Histogram\n"
        "- Scatter plot\n"
        "- Network graph visualization software (e.g., Gephi)\n"
        "- Box plot"
    ),
    'correct_answer': (
        "a. Louvain Algorithm\n"
        "b. To maximize the modularity of partitions\n"
        "c. Strong community structure with dense connections within communities and sparse connections between them\n"
        "d. It is efficient and scalable for large networks\n"
        "e. Network graph visualization software (e.g., Gephi)"
    ),
    'explanation': (
        "a. The **Louvain Algorithm** is designed for community detection in large networks, making it suitable for this task.\n\n"
        "b. The primary goal of the Louvain Algorithm is **to maximize the modularity** of the network partitions, effectively detecting communities.\n\n"
        "c. A high modularity score indicates a **strong community structure**, where nodes are densely connected within communities but sparsely connected between them.\n\n"
        "d. A key advantage of the Louvain Algorithm is that **it is efficient and scalable**, allowing it to handle large networks effectively.\n\n"
        "e. **Network graph visualization software**, such as Gephi, helps in visualizing complex networks, making it easier to interpret community structures."
    )
}

rq28 = {
    'question': (
        "An economist is studying market competition between two firms in an oligopoly. Each firm must decide whether to set high or low prices without knowing the other's decision. The payoffs for each combination of decisions are known.\n\n"
        
        "**a. Which field of study provides the framework for analyzing this strategic interaction?**\n"
        "- Game Theory\n"
        "- Regression Analysis\n"
        "- Time Series Analysis\n"
        "- Cluster Analysis\n\n"
        
        "**b. What is the term for a situation where each player's strategy is optimal, given the other player's strategy, and no player has anything to gain by changing only their own strategy?**\n"
        "- Dominant Strategy\n"
        "- Nash Equilibrium\n"
        "- Pareto Efficiency\n"
        "- Zero-Sum Game\n\n"
        
        "**c. If both firms choose to set low prices leading to lower profits for both, even though higher prices would benefit them more, what kind of dilemma is this?**\n"
        "- Coordination Game\n"
        "- Prisoner's Dilemma\n"
        "- Cooperative Game\n"
        "- Bargaining Problem\n\n"
        
        "**d. In game theory, what is a dominant strategy?**\n"
        "- A strategy that results in the highest payoff regardless of the opponent's action\n"
        "- A strategy that depends on the opponent's move\n"
        "- A randomized strategy to confuse the opponent\n"
        "- A strategy that results in a tie\n\n"
        
        "**e. How can repeated interactions between the firms affect their strategic choices?**\n"
        "- They will always choose low prices\n"
        "- Repeated interactions can lead to cooperation and better outcomes over time\n"
        "- Repeated interactions have no effect on strategy\n"
        "- They will exit the market"
    ),
    'correct_answer': (
        "a. Game Theory\n"
        "b. Nash Equilibrium\n"
        "c. Prisoner's Dilemma\n"
        "d. A strategy that results in the highest payoff regardless of the opponent's action\n"
        "e. Repeated interactions can lead to cooperation and better outcomes over time"
    ),
    'explanation': (
        "a. **Game Theory** is the study of mathematical models of strategic interaction among rational decision-makers.\n\n"
        "b. A **Nash Equilibrium** occurs when each player's strategy is optimal given the other player's strategy, and no unilateral deviation is beneficial.\n\n"
        "c. This scenario is an example of the **Prisoner's Dilemma**, where individual rationality leads to a worse outcome than cooperation.\n\n"
        "d. A **dominant strategy** is one that results in the best payoff for a player, no matter what the opponent does.\n\n"
        "e. **Repeated interactions** can encourage firms to cooperate (e.g., both setting high prices) to achieve better long-term outcomes, as they can punish non-cooperative behavior in future rounds."
    )
}

q249 = {
    'question': (
        "A machine learning engineer is building a deep neural network for image classification.\n\n"
        
        "**a. Which activation function is commonly used in hidden layers of deep neural networks to introduce non-linearity?**\n"
        "- Sigmoid function\n"
        "- ReLU (Rectified Linear Unit)\n"
        "- Linear function\n"
        "- Softmax function\n\n"
        
        "**b. To prevent overfitting in the neural network, which technique can be applied during training?**\n"
        "- Decrease the amount of training data\n"
        "- Use dropout regularization\n"
        "- Remove hidden layers\n"
        "- Use a higher learning rate\n\n"
        
        "**c. What is the main purpose of using an optimizer like stochastic gradient descent (SGD) in training neural networks?**\n"
        "- To adjust the weights to minimize the loss function\n"
        "- To initialize the weights and biases\n"
        "- To evaluate the model's performance on test data\n"
        "- To increase the complexity of the model\n\n"
        
        "**d. In a classification problem with multiple classes, which activation function is typically used in the output layer?**\n"
        "- ReLU (Rectified Linear Unit)\n"
        "- Sigmoid function\n"
        "- Tanh function\n"
        "- Softmax function\n\n"
        
        "**e. What is a key challenge when training very deep neural networks?**\n"
        "- Vanishing or exploding gradients\n"
        "- Lack of computational resources\n"
        "- Overabundance of training data\n"
        "- Simplicity of the model"
    ),
    'correct_answer': (
        "a. ReLU (Rectified Linear Unit)\n"
        "b. Use dropout regularization\n"
        "c. To adjust the weights to minimize the loss function\n"
        "d. Softmax function\n"
        "e. Vanishing or exploding gradients"
    ),
    'explanation': (
        "a. **ReLU** is commonly used in hidden layers to introduce non-linearity while being computationally efficient.\n\n"
        "b. **Dropout regularization** helps prevent overfitting by randomly dropping neurons during training, which prevents co-adaptation of neurons.\n\n"
        "c. An optimizer like **stochastic gradient descent** adjusts the weights and biases to minimize the loss function during training.\n\n"
        "d. The **Softmax function** is used in the output layer for multi-class classification to convert raw scores into probabilities summing to one.\n\n"
        "e. **Vanishing or exploding gradients** are challenges in training very deep networks, where gradients become too small or too large, hindering effective learning."
    )
}



OPEN_QUESTIONS = []
global_items = list(globals().items())

for name, value in global_items:
    if not name.startswith('_'):
        OPEN_QUESTIONS.append(value)

MIDTERM2_OPEN_QUESTIONS = OPEN_QUESTIONS[:-1]

