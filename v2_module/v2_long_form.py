# q1 = {
#     'question': "For each of the scenarios listed, identify the most fitting model or technique from the given options. Note that certain models or techniques might be suitable for more than one scenario. \n\n**MODELS/TECHNIQUES**\n\ni. Support Vector Machines\n\nii. Decision Trees\n\niii. Linear Regression\n\niv. Poisson Models\n\nv. Time Series Analysis\n\nvi. Markov Decision Processes\n\n**SCENARIOS**\n\na. Predicting the number of times a particular software feature will be used in the next month based on historical usage data.\n\nb. Developing a plan for the sequential release of product features over time to maximize user adoption and satisfaction.\n\nc. Evaluating the impact of advertising spend on sales growth, adjusting for seasonal effects.\n\nd. Identifying the primary predictors of energy consumption in commercial buildings to improve efficiency.\n\ne. Calculating the optimal strategy for inventory management to minimize holding costs and avoid stockouts.",
#     'correct_answer': """
#     The most suitable models or techniques for each scenario are as follows:

#     - a. Poisson Models
#     - b. Markov Decision Processes
#     - c. Time Series Analysis
#     - d. Decision Trees
#     - e. Linear Regression
#     """,
#         'explanation': """
#     - Poisson Models are ideal for predicting counts or frequencies, making them suitable for scenario a.
#     - Markov Decision Processes help in planning under uncertainty, fitting scenario b well.
#     - Time Series Analysis is used for forecasting future values based on past data, applicable to scenario c.
#     - Decision Trees can classify or predict based on decision rules, making them a good choice for scenario d.
#     - Linear Regression models the relationship between a dependent variable and one or more independent variables, which suits the requirements of scenario e.
#     """
# }

# q2 = {
#     'question': "An e-commerce platform is exploring ways to optimize its recommendation system to increase sales while maintaining a high level of customer satisfaction. The platform proposes testing two new algorithms: \n\n1. An algorithm prioritizing products based on individual's browsing history and past purchase behavior.\n\n2. An algorithm considering broader trends, like seasonal purchases and top-rated products, alongside individual preferences.\n\n**Question A1:** Describe a comprehensive approach to evaluating these algorithms before full-scale implementation.\n\na. Discuss the experimental design for comparing the two algorithms, considering controlling variables such as time of day, product categories, and customer demographics.\n\nb. Identify the most appropriate models or approaches for predicting the outcomes of each recommendation algorithm from the following:\n- Time Series Analysis\n- Simulation Modelling\n- Queuing Theory\n- Bayesian Inference\n- Multivariate Regression Analysis\n\nc. Propose how the platform could use simulation to model the long-term effects of each algorithm on customer purchase patterns and satisfaction, including the key performance indicators (KPIs) to monitor.",
#     'correct_answer': """
#     a. The platform should utilize a randomized controlled trial (RCT) design, ensuring participants are randomly assigned to each algorithm to control for external variables.

#     b. The most appropriate models for this scenario are:
#     - Time Series Analysis for seasonal trends.
#     - Simulation Modelling to predict customer behavior under each algorithm.
#     - Bayesian Inference to update predictions as more data becomes available.

#     c. The platform can use discrete event simulation to model long-term effects, focusing on KPIs such as customer lifetime value (CLV), repeat purchase rate, and customer satisfaction scores.
#     """,
#         'explanation': """
#     - An RCT ensures a fair comparison by minimizing bias.
#     - Time Series Analysis is ideal for capturing seasonal trends.
#     - Simulation Modelling accurately predicts behavior in complex systems with many interacting variables.
#     - Bayesian Inference provides a flexible framework for updating predictions.
#     - Simulating the long-term effects allows the platform to anticipate changes in customer behavior and satisfaction, ensuring the chosen algorithm aligns with business goals and enhances the customer experience.
#     """
# }

# q3 = {
#     'question': "A finance research team is analyzing a dataset to predict future stock market trends, facing dataset challenges such as heteroscedasticity, evident trends and seasonal patterns, and high dimensionality with potential correlation among predictors.\n\n**a.** To address heteroscedasticity in stock prices, which transformation technique should the team apply?\n1. Linear Transformation\n2. Box-Cox Transformation\n3. Logarithmic Transformation\n4. Z-Score Normalization\n\n**b.** When detrending the data to remove evident trends and seasonal patterns, what is the primary reason for doing so?\n1. To increase the dataset size\n2. To enhance the computational speed of model training\n3. To reduce the effect of time-related confounding factors\n4. To improve the color scheme of data visualizations\n\n**c.** To reduce dimensionality and mitigate multicollinearity among economic indicators, which method should the team use?\n1. Principal Component Analysis (PCA)\n2. Linear Discriminant Analysis (LDA)\n3. K-Means Clustering\n4. Decision Trees\n\n**d.** In Principal Component Analysis (PCA), what role do eigenvalues and eigenvectors play?\n1. They determine the color palette for data visualization.\n2. They are used to calculate the mean and median of the dataset.\n3. They identify the principal components and ensure they are orthogonal.\n4. They enhance the dataset's security and privacy.",
#     'correct_answer': (
#         "a. Box-Cox Transformation\n\n"
#         "b. To reduce the effect of time-related confounding factors\n\n"
#         "c. Principal Component Analysis (PCA)\n\n"
#         "d. They identify the principal components and ensure they are orthogonal.\n\n"
#     ),
#     'explanation': (
#         "Box-Cox Transformation is used to stabilize variance across the data.\n\n"
#         "Detrending is crucial for removing the influence of temporal trends on the analysis, "
#         "making the underlying patterns in data more apparent.\n\n"
#         "PCA is a powerful tool for dimensionality reduction and dealing with multicollinearity by "
#         "transforming the dataset into a set of orthogonal (uncorrelated) variables, which are the principal components.\n\n"
#         "Eigenvalues and eigenvectors are fundamental in determining the direction and magnitude of "
#         "these principal components, helping to understand the dataset's variance structure.\n\n"
#     )
# }

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





q9 = {
    'question': "Given the optimization scenarios below, classify each problem based on its most fitting optimization category. Consider the variables (x, y, etc.), known data (a, b, c, etc.), and note that all values of c are positive.\n\nChoices:\n- Linear Program\n- Convex Program\n- Convex Quadratic Program\n- General Nonconvex Program\n- Integer Program\n\nQuestion a:\nMinimize the sum of (c_i * x_i * y_i), subject to the sum of (a_ij * x_i) being equal to b_j for all j, x_i greater than or equal to 0, and y_i being either 0 or 1 for all i.\n\nQuestion b:\nMinimize the sum of (x_i^2 + y_i^2), subject to (a_i * x_i + b_i * y_i) less than or equal to c for all i, and x_i, y_i greater than or equal to 0.\n\nQuestion c:\nMaximize the product of (x_i^c_i), subject to the sum of (a_ij * x_i) less than or equal to b_j for all j, and x_i greater than or equal to 1.\n\nQuestion d:\nMaximize the sum of (sqrt(c_i) * x_i), subject to the sum of (a_ij * x_i) being equal to b_j for all j, x_i greater than or equal to 0.\n\nQuestion e:\nMinimize c^T x + d^T y, subject to A x + B y = b, x_i belonging to {0,1} for all i, and y_i greater than or equal to 0.",
    'correct_answer': (
        "a. Integer Program\n"
        "b. Convex Quadratic Program\n"
        "c. General Nonconvex Program\n"
        "d. Convex Program\n"
        "e. Integer Program"
    ),

    'explanation': (
        "a. The presence of binary variables (y_i being either 0 or 1) along with linear constraints and a linear objective function where the variables are multiplied by binary variables indicates an Integer Program.\n\n"
        "b. The objective function is the sum of squares, which is a classic form of a convex quadratic function, and the constraints are linear, making it a Convex Quadratic Program.\n\n"
        "c. Maximizing the product of variables each raised to a power (where c_i > 0) creates a non-linear, non-convex objective function, indicating a General Nonconvex Program.\n\n"
        "d. The objective function involves the square root of constants multiplied by variables, which is a convex function. Along with linear constraints, this forms a Convex Program.\n\n"
        "e. The mixture of binary (x_i) and continuous variables (y_i), with linear constraints and a linear objective function, indicates an Integer Program, particularly a Mixed Integer Linear Program (MILP) due to the presence of both integer and continuous variables."
    )
}

q10 = {
    'question': "A tech startup is facing multiple challenges as it scales its operations, from analyzing customer feedback to optimizing its internal team dynamics and predicting market trends. The data science team at the startup is considering employing various advanced analytical models to address these challenges.\n\nGiven the following scenarios, identify the most appropriate model or approach from the options: Non-Parametric Methods, Bayesian Modeling, Community Detection in Graphs, Neural Networks and Deep Learning, and Game Theory.\n\na. The startup has collected a vast amount of unstructured customer feedback through various channels. They want to understand the overall sentiment towards their product to inform future development. The data varies widely in format and content.\n\nb. To enhance collaboration within the company, the team wants to analyze the communication patterns among employees. They have data on email interactions, meeting attendances, and project collaborations.\n\nc. The product team is considering several new features based on market trends and internal data. However, there is significant uncertainty regarding the adoption of these features by their user base.\n\nd. The company wants to create a predictive model for identifying potential churn customers based on their interaction with the product's online platform, including page views, feature usage, and support ticket submissions.\n\ne. In response to emerging competitive threats, the startup needs to strategize its pricing model. This requires understanding how competitors might react to their pricing changes and the potential impact on market share.",
    'correct_answer': (
        "a. Neural Networks and Deep Learning\n"
        "b. Community Detection in Graphs\n"
        "c. Bayesian Modeling\n"
        "d. Neural Networks and Deep Learning\n"
        "e. Game Theory"
    ),

    'explanation': (
        "a. Neural Networks and Deep Learning are highly effective for processing and analyzing unstructured data, making them suitable for sentiment analysis of varied customer feedback.\n\n"
        "b. Community Detection in Graphs can identify clusters within large networks, such as communication patterns among employees, highlighting tightly-knit groups or isolated individuals.\n\n"
        "c. Bayesian Modeling allows for incorporating prior knowledge and handling uncertainty, ideal for assessing the potential adoption of new features based on limited or uncertain data.\n\n"
        "d. Neural Networks and Deep Learning excel at identifying complex patterns in high-dimensional data, suitable for predicting customer churn from varied interactions with an online platform.\n\n"
        "e. Game Theory provides a framework for competitive decision-making, enabling the startup to anticipate competitors' reactions to pricing changes and strategize accordingly."
    )
}

q11 = {
    'question': "A financial analytics firm is analyzing historical stock market data to enhance its prediction models for future stock prices. The dataset includes daily closing prices, trading volume, and various economic indicators over the last 20 years. However, the data exhibits heteroscedasticity, trends, and high dimensionality, making standard modeling approaches inadequate.\n\nGiven the scenarios described, identify the most appropriate model or approach from the options: Box-Cox Transformation, Detrending via linear regression, Principal Component Analysis (PCA), and Eigenvalues and Eigenvectors.\n\na. The firm notices that the variance in trading volumes increases significantly with higher volumes. To stabilize the variance across the dataset, which transformation should be applied, and what is the primary reason for its application?\n\nb. Upon further analysis, a long-term upward trend in stock prices is evident, attributed to overall market growth. Before modeling, the firm wants to remove this trend to focus on underlying patterns. Which method should they use, and why?\n\nc. The firm aims to reduce the number of economic indicators in the dataset without losing essential information due to the high correlation among indicators. Which technique is most suitable, and what is its primary benefit?\n\nd. After applying PCA, the firm wants to understand the influence of the original economic indicators on the principal components. Which concept is crucial for this interpretation, and why?",
    'correct_answer': (
        "a. Box-Cox Transformation\n"
        "b. Detrending via linear regression\n"
        "c. Principal Component Analysis (PCA)\n"
        "d. Eigenvalues and Eigenvectors"
    ),
    'explanation': (
        "a. The Box-Cox Transformation is applied to correct heteroscedasticity, a common issue in financial data, by making the variance constant across the range of data.\n\n"
        "b. Detrending is essential for removing long-term trends, such as general market growth, to analyze the more nuanced fluctuations in stock prices that are of interest.\n\n"
        "c. PCA is used for dimensionality reduction, particularly useful in datasets with many correlated variables, by transforming them into a smaller number of uncorrelated variables while retaining most of the original variance.\n\n"
        "d. Eigenvectors from PCA provide insight into the contribution of each original variable to the principal components, helping interpret the reduced-dimensional space in terms of the original variables."
    )
}



q12 = {
    'question': ("An e-commerce company is revamping its analytics framework to address several critical areas: sales forecasting, customer segmentation, marketing effectiveness, and operational efficiency. The data science team plans to employ a range of optimization models to tackle these challenges effectively.\n\n"
                 "Given the scenarios below, identify the most suitable optimization model or approach for each, based on the descriptions provided in the document.\n\n"
                 "a. The company intends to forecast quarterly sales using historical sales data, which includes seasonal patterns and economic indicators. The model must accommodate fluctuating variance in sales volume.\n\n"
                 "b. To better understand its customer base, the company seeks to segment customers into distinct groups based on purchasing behavior, frequency, and preferences.\n\n"
                 "c. The marketing department wants to evaluate the impact of various advertising campaigns on sales. This requires a model that can handle both the campaigns' direct effects and their interactions.\n\n"
                 "d. With an aim to reduce shipping costs and delivery times, the company is analyzing its logistics and supply chain operations. The challenge includes balancing multiple factors, such as warehouse stock levels, transportation costs, and delivery routes.\n\n"
                 "e. Given the competitive nature of the e-commerce industry, the company also wants to optimize its pricing strategy to maximize profits while remaining attractive to customers, taking into account competitors' pricing and market demand elasticity.\n"
                ),
    'correct_answer': (
        "a. Time Series Model with Box-Cox Transformation\n\n"
        "b. Clustering\n\n"
        "c. Elastic Net\n\n"
        "d. Linear/Quadratic Integer Program\n\n"
        "e. Game Theory"
    ),
    'explanation': (
        "a. The Box-Cox Transformation corrects heteroscedasticity, stabilizing variance in sales volume, making Time Series Models more effective.\n\n"
        "b. Clustering optimizes the sum of distances from each data point to its cluster center, effectively segmenting customers based on various behaviors.\n\n"
        "c. Elastic Net is ideal for models requiring variable selection and regularization to handle collinearity, perfect for marketing data with many predictors.\n\n"
        "d. Logistics optimization involves binary decisions (e.g., whether to use a specific route), making it a case for Linear/Quadratic Integer Programming.\n\n"
        "e. Game Theory models the strategic interactions between the company and its competitors, optimizing pricing strategies in a competitive environment."
    )
}



q13 = {
    'question': ("A logistics company is analyzing its package sorting system to improve efficiency and reduce bottlenecks during peak operation hours. The system involves sorting packages based on destination, size, and priority, with a complex network of conveyor belts and automated sorting machines. The company has collected data on package arrival times, sorting times, and system throughput. The data science team intends to use probabilistic models and simulation to address several challenges.\n\n"
                 "a. To model the arrival of packages to the sorting facility, which probabilistic distribution should be applied, considering that arrivals are continuous and independent?\n\n"
                 "b. The company observes that the time taken by a sorting machine to process a package varies, but it generally follows a pattern. Which model best describes the processing time of packages, given the requirement for a memoryless property?\n\n"
                 "c. Given the complexity of the sorting system and the variability in package arrivals and processing times, which simulation approach would best allow the company to analyze system performance and identify potential improvements?\n\n"
                 "d. The management is considering implementing a new queueing strategy to prioritize urgent packages. They want to estimate the impact of this change on average waiting times and system throughput. How can simulation be utilized to assess the effectiveness of this new strategy?\n\n"
                 "e. Finally, the company wants to explore the long-term effects of adding an additional sorting machine to the system. Which method would allow them to model the system's states and transitions over time, considering the probabilistic nature of package arrivals and processing times?"
                ),
    'correct_answer': (
        "a. Poisson Distribution\n\n"
        "b. Exponential Distribution\n\n"
        "c. Discrete Event Stochastic Simulation\n\n"
        "d. Prescriptive Simulation\n\n"
        "e. Markov Chain Model"
    ),
    'explanation': (
        "a. The Poisson Distribution is appropriate for modeling the arrival of packages because it deals with the number of events (package arrivals) that occur in a fixed interval of time and space, assuming these events occur with a known constant mean rate and independently of the time since the last event.\n\n"
        "b. The Exponential Distribution, known for its memoryless property, is suitable for modeling the time between events in a Poisson process, making it ideal for describing the sorting time of packages.\n\n"
        "c. Discrete Event Stochastic Simulation allows for modeling the complex interactions within the sorting system, including the randomness of package arrivals and processing times, to analyze performance and identify bottlenecks.\n\n"
        "d. Prescriptive Simulation can be used to simulate the effect of different queueing strategies on the system, allowing the company to assess changes before implementation based on various metrics such as average waiting times and system throughput.\n\n"
        "e. The Markov Chain Model is effective for modeling the system's states and transitions over time, providing insights into the long-term effects of operational changes such as adding an additional sorting machine."
    )
}


q14 = {
    'question': ("An agricultural technology company is developing a predictive model to optimize crop yields across various regions. The project involves integrating diverse datasets, including soil properties, weather patterns, crop genetics, and historical yield data. The analytics team is exploring several modeling techniques to address the challenges presented by the variability and complexity of the data.\n\n"
                 "a. Given the variability in weather patterns and their significant impact on crop yields, which modeling approach would best accommodate the uncertainty inherent in weather forecasts?\n\n"
                 "b. Considering the need to classify regions based on soil properties that directly influence crop choice and management practices, which technique is most suitable for segmenting the regions into distinct categories?\n\n"
                 "c. To account for the non-linear relationship between soil moisture levels and crop yield, which method should be applied to capture the complexity of this relationship?\n\n"
                 "d. In evaluating the effectiveness of different fertilizers on crop yield, how should the company approach the analysis to determine the most impactful factors?\n\n"
                 "e. The company aims to develop a model that predicts the likelihood of a pest outbreak based on historical data and current observations. Which model would allow for a dynamic assessment of risk over time?"
                ),
    'correct_answer': (
        "a. Bayesian Modeling\n\n"
        "b. Clustering\n\n"
        "c. Neural Networks\n\n"
        "d. Elastic Net\n\n"
        "e. Markov Chains"
    ),
    'explanation': (
        "a. Bayesian Modeling is well-suited for incorporating uncertainty and making probabilistic predictions, ideal for weather-related forecasts.\n\n"
        "b. Clustering is an effective unsupervised learning technique for categorizing regions with similar soil properties without predefined labels.\n\n"
        "c. Neural Networks can model complex, non-linear relationships, making them suitable for understanding the nuanced effects of soil moisture on crop yield.\n\n"
        "d. Elastic Net combines Lasso and Ridge regression techniques, offering a balanced approach to variable selection and regularization, crucial for identifying the key factors influencing crop yield.\n\n"
        "e. Markov Chains are effective for modeling state transitions over time, providing a robust framework for predicting pest outbreaks based on changing conditions."
    )
}



q15 = {
    'question': ("A healthcare research institute is conducting a longitudinal study on the impact of lifestyle factors on chronic diseases. The dataset encompasses a wide range of variables, including dietary habits, physical activity levels, genetic predispositions, and health outcomes over ten years. However, the institute faces challenges with missing data across various time points and variables.\n\n"
                 "a. Dietary habit data is sporadically missing for some participants over the ten-year period, with no apparent pattern to the missingness. Which imputation method would best handle this scenario without introducing significant bias?\n\n"
                 "b. Physical activity levels are consistently missing for a subgroup of participants who dropped out after the first year. The researchers suspect this dropout is not random but related to the participants' health conditions. How should the missing data be treated to avoid bias in the study's conclusions?\n\n"
                 "c. Genetic predisposition data is missing for 2% of the participants due to technical issues with genetic testing. The missingness is considered completely random. What is the most straightforward imputation method that could be applied in this case?\n\n"
                 "d. The study plans to analyze the correlation between lifestyle factors and the onset of chronic diseases. However, health outcome data is missing for 5% of the observations. The missing health outcomes are mostly from the earlier part of the study. Which approach should the institute use to impute this missing data, considering the importance of temporal patterns?\n\n"
                 "e. Finally, the researchers want to ensure that their imputation strategy for missing dietary and physical activity data does not distort the study's findings. What validation technique could they employ to assess the effectiveness of their imputation methods?"
                ),
    'correct_answer': (
        "a. Multiple Imputation\n\n"
        "b. Use a model-based approach, such as a mixed-effects model, to account for non-random dropout\n\n"
        "c. Mean/Median/Mode Imputation\n\n"
        "d. Time Series Imputation\n\n"
        "e. Cross-validation using a subset of complete cases to simulate missingness and then comparing imputed values against actual values"
    ),
    'explanation': (
        "a. Multiple Imputation is suitable for handling sporadic missing data across time, as it accounts for the uncertainty around the missing data and generates several imputed datasets for analysis.\n\n"
        "b. Model-based approaches like mixed-effects models can handle missing data due to dropout by incorporating random effects that account for the variability among participants and fixed effects for observed variables.\n\n"
        "c. Mean/Median/Mode Imputation is a straightforward method for handling completely random missing data, using the most central tendency measure that best fits the distribution of the data.\n\n"
        "d. Time Series Imputation is appropriate for data with temporal patterns, as it can use trends and seasonal components to estimate missing values more accurately.\n\n"
        "e. Cross-validation is a robust technique for validating imputation effectiveness, allowing researchers to simulate missingness in known cases and assess how well the imputation method recovers the original data."
    )
}

midterm_1_q25 = {
    'question': (
        "A marketing agency is segmenting its audience for targeted advertising campaigns."
        "\n\n"
        "a. For creating customer segments based on shopping behavior and preferences, which clustering method would be most suitable?"
        "\n"
        "K-means Clustering\n"
        "KNN Clustering\n"
        "PCA\n"
        "Poisson Variance Classification\n\n"
        "A retail chain is analyzing factors affecting its sales performance."
        "\n\n"
        "a. To predict future sales based on factors like store location, advertising spend, and local demographics, which regression method should be employed?"
        "\n"
        "Linear Regression\n"
        "Poisson Regression\n"
        "Bayesian Regression\n"
        "Lasso Regression\n\n"
        "b. The retailer needs to understand the relationship between temperature and outdoor sales. If the relationship is non-linear, what should they consider in their regression model?"
        "\n"
        "Transformation and Interaction Terms\n"
        "Logistic Regression\n"
        "Polynomial Regression\n"
        "Ridge Regression"
    ),
    'correct_answer': (
        "a. K-means Clustering\n\n"
        "a. Linear Regression\n\n"
        "b. Polynomial Regression"
    ),
    'explanation': (
        "a. K-means Clustering is commonly used in market segmentation due to its simplicity and efficiency in clustering similar items based on predefined characteristics.\n\n"
        "a. Linear Regression is suitable for predicting future sales based on several independent variables, offering a straightforward approach to regression analysis.\n\n"
        "b. Polynomial Regression helps capture non-linear relationships, providing flexibility when modeling sales trends influenced by varying temperatures."
    )
}

midterm_1_q24 = {
    'question': (
        "A financial institution is implementing a new system to classify loan applicants based on risk."
        "\n\n"
        "a. Which classifier would be more effective for categorizing applicants into 'high risk' and 'low risk', considering the cost of misclassification?"
        "\n"
        "Linear Regression\n"
        "K-Nearest Neighbor (KNN)\n"
        "Support Vector Machine (SVM)\n"
        "Random Forest\n\n"
        "b. In a scenario where the bank needs to identify potential fraudulent transactions, which approach should they use, given the transactions data is highly imbalanced?"
        "\n"
        "Hard Classifiers\n"
        "Soft Classifiers\n"
        "Decision Trees\n"
        "Bayesian Classifiers\n\n"
        "An e-commerce company is evaluating different models for predicting customer purchase behavior."
        "\n\n"
        "a. To ensure the chosen model is not overfitting, which method should be used for validating the model's effectiveness?"
        "\n"
        "Cross-Validation\n"
        "Training on Entire Dataset\n"
        "AIC/BIC Comparison\n"
        "Holdout Method\n\n"
        "b. If the model performs well on the training data but poorly on the validation data, what might this indicate?"
        "\n"
        "The model is underfitting\n"
        "The model is overfitting\n"
        "The model is perfectly fitted\n"
        "The model is not complex enough"
    ),
    'correct_answer': (
        "a. Support Vector Machine (SVM)\n\n"
        "b. Soft Classifiers\n\n"
        "a. Cross-Validation\n\n"
        "b. The model is overfitting"
    ),
    'explanation': (
        "a. Support Vector Machine (SVM) is effective for binary classification, particularly when the cost of misclassification is high. It creates a hyperplane that best separates the two categories, minimizing classification errors.\n\n"
        "b. Soft Classifiers, which include techniques like probability thresholds, are suitable for highly imbalanced datasets, allowing for a more flexible classification approach.\n\n"
        "a. Cross-Validation provides a reliable method for validating a model's effectiveness by testing it across different subsets of the data to ensure robustness.\n\n"
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




# q16 = {
#     'question': "A national health survey collects data on various health metrics, lifestyle choices, and demographic information from participants across the country. However, the dataset contains missing values in several variables, such as Body Mass Index (BMI), smoking status, and income level, which are crucial for the study's objectives. The data science team at the health department needs to decide on the best strategies for handling this missing data.\n\na. BMI values are missing for 5% of the participants. These missing values are believed to be Missing Completely at Random (MCAR). What imputation method should be applied to handle the missing BMI values?\n\nb. Smoking status is missing for a significant number of participants, especially among older demographics. The missingness is suspected to be Not Missing at Random (NMAR), possibly because older participants chose not to disclose their smoking habits. Which approach is most suitable for imputing missing data in this case?\n\nc. Income level data is missing for 10% of the participants, predominantly in regions with lower socio-economic status. This pattern suggests the missing data is Missing at Random (MAR). Given the importance of accurately estimating missing income levels, which imputation method would be most effective?\n\nd. After handling missing data for BMI, smoking status, and income level, what technique can the team use to assess the impact of imputation on the analysis results?\n\ne. Considering the need to maintain the integrity of the survey results and the potential bias introduced by missing data, what overarching strategy should guide the team's approach to imputation?",
#     'correct_answer': {
#         'a': "Mean/Median/Mode Imputation",
#         'b': "Model-Based Imputation, incorporating variables correlated with smoking status",
#         'c': "Multiple Imputation, using variables related to socio-economic status",
#         'd': "Sensitivity Analysis, comparing results with and without imputed data",
#         'e': "A comprehensive approach, prioritizing methods that minimize bias and reflect the underlying data distribution"
#     },
#     'explanation': {
#         'a': "Mean/Median/Mode Imputation is straightforward and appropriate for MCAR data, as it does not introduce significant bias.",
#         'b': "Model-Based Imputation allows for the inclusion of auxiliary variables that may explain the mechanism behind the missingness of smoking status.",
#         'c': "Multiple Imputation is suitable for MAR data, as it accounts for the uncertainty of missing values and uses the relationships between variables to generate plausible estimates.",
#         'd': "Sensitivity Analysis helps in evaluating the robustness of the study's findings by analyzing the effects of imputation on the results.",
#         'e': "Considering the varied nature of missing data, a comprehensive approach ensures that the imputation strategy is aligned with the data's characteristics and the study's objectives, thereby maintaining the validity and reliability of the results."
#     }
# }

q17 = {
    'question': ("A research team is conducting a study on the effects of dietary habits on long-term health outcomes. They've collected a dataset over 10 years, tracking individuals' consumption of fruits, vegetables, processed foods, and their health outcomes, including cholesterol levels, blood pressure, and incidence of heart disease.\n\n"
                 "Given the dataset and research goals, answer the following questions:\n\n"
                 "a. Given the objective to analyze the impact of each dietary habit on cholesterol levels, which regression model should the team use, and why?\n\n"
                 "b. The team notices that blood pressure readings have a non-linear relationship with processed food consumption. How should they adjust their regression model to account for this?\n\n"
                 "c. To predict the likelihood of heart disease based on dietary habits, what type of regression model is most appropriate, and what is the primary reason for its selection?\n\n"
                 "d. The team wants to control for the variable 'age' while analyzing the effects of dietary habits on health outcomes. How should they incorporate 'age' into their regression analysis?\n\n"
                 "e. After initial analysis, the team realizes that multicollinearity between fruit and vegetable consumption might be skewing their results. What strategy should they consider to address this issue?"
                ),
    'correct_answer': (
        "a. Multiple linear regression, because it allows for the assessment of the impact of multiple predictors (dietary habits) on a continuous outcome (cholesterol levels).\n\n"
        "b. Introduce polynomial terms for processed food consumption into the model to capture the non-linear relationship.\n\n"
        "c. Logistic regression, as it is designed for binary outcomes (e.g., the presence or absence of heart disease) and can handle predictors like dietary habits.\n\n"
        "d. Include 'age' as a covariate in the regression model to adjust for its potential confounding effect on the relationship between dietary habits and health outcomes.\n\n"
        "e. Apply Ridge or Lasso regression techniques to penalize the regression coefficients, thereby reducing the effect of multicollinearity and selecting the most relevant predictors."
    ),
    'explanation': (
        "a. Multiple linear regression is suitable for modeling the relationship between multiple independent variables and a single continuous dependent variable, making it ideal for assessing the impacts of various dietary habits on cholesterol levels.\n\n"
        "b. Polynomial regression allows for the modeling of non-linear relationships by adding polynomial terms, which can better fit the observed data when linear terms are insufficient.\n\n"
        "c. Logistic regression is used when the dependent variable is categorical, such as predicting a binary outcome. It models the probability that the outcome is present given the predictors.\n\n"
        "d. Including 'age' as a covariate helps to adjust for its effects, providing a clearer picture of how dietary habits alone influence health outcomes, without the confounding impact of age.\n\n"
        "e. Ridge and Lasso regression are regularization methods that address multicollinearity by penalizing the size of coefficients, which can help in selecting the most significant variables when predictors are correlated."
    )
}


q18 = {
    'question': ("A city's public transportation system is evaluating its bus service efficiency. The system operates with varying arrival rates throughout the day due to peak and off-peak hours. During peak hours (30% of the time), buses arrive at a rate of 10 buses/hour. During off-peak hours, buses arrive at a rate of 4 buses/hour. The average boarding time per passenger is consistently 1 minute.\n\n"
                 "Given this information, answer the following questions regarding the application of queuing theory and stochastic modeling to improve service efficiency:\n\n"
                 "a. Initially, the transportation system models bus arrivals with 15 buses running during peak hours and 6 buses during off-peak hours. What would you expect the model to show regarding passenger wait times?\n- Wait times are minimal at both peak and off-peak hours.\n- Wait times are minimal at peak hours and longer at off-peak hours.\n- Wait times are longer at peak hours and minimal at off-peak hours.\n- Wait times are long at both peak and off-peak hours.\n\n"
                 "b. After analyzing passenger feedback on wait times, the system experiments with dynamic scheduling, adjusting the number of buses to 12 during peak hours and 8 during off-peak hours. What would you expect the model to show under this new scheduling approach?\n- Wait times are reduced at both peak and off-peak hours compared to the initial model.\n- Wait times are reduced at peak hours but increased at off-peak hours compared to the initial model.\n- Wait times are increased at peak hours but reduced at off-peak hours compared to the initial model.\n- Wait times remain unchanged at both peak and off-peak hours compared to the initial model.\n\n"
                 "To optimize passenger wait times further, the transportation system considers implementing a real-time tracking and notification system for bus arrivals. This system could inform potential adjustments to bus frequency based on current demand and traffic conditions.\n\n"
                 "c. Considering the introduction of a real-time tracking and notification system, which statement best describes its expected impact on the modeling of the bus service?\n- It introduces variability that makes the queuing model less predictable.\n- It reduces the need for a queuing model by directly managing passenger expectations.\n- It enhances the accuracy of the queuing model by providing real-time data for better decision-making.\n- It has no significant impact on the queuing model as it does not affect bus arrival rates or boarding times.\n"
                ),
    'correct_answer': (
        "a. Wait times are longer at peak hours and minimal at off-peak hours.\n\n"
        "b. Wait times are reduced at both peak and off-peak hours compared to the initial model.\n\n"
        "c. It enhances the accuracy of the queuing model by providing real-time data for better decision-making."
    ),
    'explanation': (
        "a. Given the higher arrival rate of buses during peak hours in the initial model, passenger wait times are expected to be longer due to the increased demand. The model's setup during off-peak hours should minimize wait times due to a better balance between bus arrivals and passenger demand.\n\n"
        "b. The adjusted number of buses during both peak and off-peak hours aims to more closely match the arrival rate with passenger demand, potentially reducing wait times across the board by optimizing resources.\n\n"
        "c. A real-time tracking and notification system provides passengers with current wait times and bus arrival information, allowing for more informed decision-making. For the transportation system, this real-time data can inform adjustments to bus frequencies, enhancing the queuing model's relevance and accuracy by incorporating current demand and traffic conditions."
    )
}



q19 = {
    'question': ("A data science team at a tech company is analyzing user interaction data with their software to improve user experience and engagement. The dataset includes daily active users (DAU), average session time, number of sessions per user, feature usage frequency, and user retention rate. The team wants to use regression analysis to predict user retention based on these variables.\n\n"
                 "a. Considering the team's objective to predict user retention rate, a continuous outcome variable, which regression model is most appropriate for initial analysis?\n- Simple linear regression\n- Multiple linear regression\n- Logistic regression\n\n"
                 "b. The team observes a potential non-linear relationship between average session time and user retention rate. Which method could effectively capture this non-linearity in the regression model?\n- Adding polynomial terms for average session time\n- Transforming the retention rate using a log function\n- Using a logistic regression model\n\n"
                 "c. To identify which features are most predictive of user retention while avoiding overfitting with too many variables, which technique should the team employ?\n- Stepwise regression\n- Ridge regression\n- Principal Component Analysis (PCA)\n\n"
                 "d. After developing the regression model, the team wants to evaluate its performance in predicting user retention. Which metric is most suitable for assessing the model's accuracy?\n- R-squared\n- Mean Squared Error (MSE)\n- Area Under the ROC Curve (AUC)\n\n"
                 "e. The team plans to segment users based on their likelihood of retention, as predicted by the regression model. For users classified as at risk of low retention, targeted engagement strategies will be implemented. Which approach allows the team to classify users based on predicted retention rates?\n- K-means clustering on predicted retention rates\n- Setting a threshold on the predicted retention rate to classify users\n- Using a logistic regression model for classification"
                ),
    'correct_answer': (
        "a. Multiple linear regression\n\n"
        "b. Adding polynomial terms for average session time\n\n"
        "c. Stepwise regression\n\n"
        "d. Mean Squared Error (MSE)\n\n"
        "e. Setting a threshold on the predicted retention rate to classify users"
    ),
    'explanation': (
        "a. Multiple linear regression is appropriate for modeling the relationship between multiple predictors and a continuous outcome variable.\n\n"
        "b. Adding polynomial terms can help model non-linear relationships between predictors and the outcome variable within a linear regression framework.\n\n"
        "c. Stepwise regression is a method of adding or removing variables from the model based on their statistical significance, which helps in selecting the most predictive features while avoiding overfitting.\n\n"
        "d. Mean Squared Error (MSE) measures the average of the squares of the errors between the predicted and actual values, making it a suitable metric for evaluating the accuracy of regression models.\n\n"
        "e. Setting a threshold on the predicted retention rate allows for the classification of users into different categories (e.g., high risk vs. low risk of churn) based on their predicted likelihood of retention."
    ),
}


q20 ={
    'question': ("A marketing team is interested in understanding the factors that influence a customer's decision to subscribe to a new online service. The dataset includes customer demographics (age, income, education), previous subscription history, marketing engagement metrics (email opens, click-through rates), and whether the customer subscribed to the new service (yes or no).\n\n"
                 "a. Given the binary nature of the outcome variable (subscribed or not), which regression model should the marketing team use to analyze the data?\n- Simple linear regression\n- Multiple linear regression\n- Logistic regression\n\n"
                 "b. To understand the effect of age on the likelihood of subscribing, while controlling for income and education, which component should be included in the logistic regression model?\n- Interaction terms between age, income, and education\n- Age as a main effect, without interaction terms\n- Polynomial terms for age\n\n"
                 "c. The marketing team suspects that the relationship between click-through rates and subscription likelihood is not linear. How can they adjust their logistic regression model to account for this?\n- Include click-through rate as a categorical variable\n- Add polynomial terms for click-through rate\n- Transform click-through rate using a logistic function\n\n"
                 "d. After fitting the logistic regression model, which metric would be most appropriate for evaluating its performance in classifying customers as subscribers or non-subscribers?\n- R-squared\n- Mean Squared Error (MSE)\n- Area Under the ROC Curve (AUC)\n\n"
                 "e. If the marketing team wants to use the model to target customers who are most likely to subscribe, which strategy would effectively leverage the model's predictions?\n- Send marketing materials to customers with a predicted probability of subscribing above a certain threshold\n- Cluster customers into groups based on predicted probabilities and target the largest cluster\n- Perform A/B testing on a random sample of customers regardless of their predicted probabilities"
                ),
 'correct_answer': (
        "a. Logistic regression\n\n"
        "b. Age as a main effect, without interaction terms\n\n"
        "c. Add polynomial terms for click-through rate\n\n"
        "d. Area Under the ROC Curve (AUC)\n\n"
        "e. Send marketing materials to customers with a predicted probability of subscribing above a certain threshold"
    ),
    'explanation': (
        "a. Logistic regression is suitable for binary outcome variables, making it the appropriate choice for analyzing subscription decisions.\n\n"
        "b. Including age as a main effect allows the model to assess its direct impact on subscription likelihood, controlling for other factors.\n\n"
        "c. Adding polynomial terms for click-through rate can capture non-linear effects on the likelihood of subscribing.\n\n"
        "d. The Area Under the ROC Curve (AUC) is a performance metric for classification models, indicating the model's ability to distinguish between classes.\n\n"
        "e. Targeting customers based on a probability threshold allows for more efficient allocation of marketing resources to those most likely to subscribe."
    ),

}
# midterm_1_q21 = {
#     'question': (
#         "A healthcare analytics team is working on various models to analyze patient data for improving treatment outcomes. They have collected extensive patient data over the years, including demographics, treatment details, and health outcomes."
#         "\n\n"
#         "a. For classifying patients into high-risk and low-risk categories based on their treatment outcomes, which model would be best suited?"
#         "\n"
#         "Cusum\n"
#         "K-Nearest Neighbors\n"
#         "Support Vector Machines (SVM)\n\n"
#         "b. To cluster patients based on similarities in their diagnosis and treatment types, which algorithm would be most effective?"
#         "\n"
#         "K-Means Clustering\n"
#         "PCA\n"
#         "GARCH Variance Clustering\n\n"
#         "The healthcare analytics team is also interested in predicting the efficacy of treatments over time."
#         "\n\n"
#         "a. If the team wants to forecast treatment efficacy based on past trends and seasonal variations, which model should they use?"
#         "\n"
#         "ARIMA\n"
#         "Exponential Smoothing\n"
#         "Random Forests\n\n"
#         "b. To detect significant changes in treatment efficacy over time, which method would be most suitable?"
#         "\n"
#         "CUSUM\n"
#         "Principal Component Analysis\n"
#         "Box-Cox Transformation"
#     ),
#     'correct_answer': (
#         "a. Support Vector Machines (SVM)\n\n"
#         "b. K-Means Clustering\n\n"
#         "a. Exponential Smoothing\n\n"
#         "b. CUSUM"
#     ),
#     'explanation': (
#         "a. Support Vector Machines (SVM) are effective for classifying patients into distinct categories based on a clear decision boundary, ideal for high-risk and low-risk segmentation.\n\n"
#         "b. K-Means Clustering is widely used for creating clusters with well-defined groupings, making it suitable for segmenting patients based on diagnosis and treatment types.\n\n"
#         "a. Exponential Smoothing is ideal for forecasting trends with seasonal components, providing a smooth approach to predicting treatment efficacy.\n\n"
#         "b. CUSUM (Cumulative Sum) is a robust method for detecting significant changes or shifts in data over time, well-suited for monitoring treatment efficacy."
#     )
# }

# midterm_1_q22 = {
#     'question': (
#         "A bank is developing a model to classify loan applicants as high-risk or low-risk, with the goal of minimizing misclassification."
#         "\n\n"
#         "a. Which model would be more suitable for this task, considering the importance of minimizing the misclassification of high-risk applicants?"
#         "\n"
#         "Support Vector Machines (SVM)\n"
#         "K-Nearest Neighbors (KNN)\n\n"
#         "b. In a medical diagnosis system, which model would be preferable for classifying patients based on a dataset with many overlapping characteristics?"
#         "\n"
#         "Support Vector Machines (SVM)\n"
#         "K-Nearest Neighbors (KNN)\n\n"
#         "A marketing team has developed several predictive models for customer behavior."
#         "\n\n"
#         "a. To avoid overfitting, which approach should they use for model assessment?"
#         "\n"
#         "Cross-validation\n"
#         "Training on the entire dataset\n\n"
#         "b. When choosing between two different models for predicting sales, one with a lower AIC and one with a higher BIC, which model should be preferred considering both simplicity and likelihood?"
#         "\n"
#         "Model with lower AIC\n"
#         "Model with higher BIC"
#     ),
#     'correct_answer': (
#         "a. Support Vector Machines (SVM)\n\n"
#         "b. Support Vector Machines (SVM)\n\n"
#         "a. Cross-validation\n\n"
#         "b. Model with lower AIC"
#     ),
#     'explanation': (
#         "a. Support Vector Machines (SVM) offer a robust decision boundary that is effective for binary classification tasks where the cost of misclassification is high, making it suitable for loan risk assessment.\n\n"
#         "b. SVMs are also preferred in a medical diagnosis system with overlapping characteristics due to their ability to handle complex decision boundaries.\n\n"
#         "a. Cross-validation helps avoid overfitting by testing the model on different subsets of data, providing a comprehensive validation approach.\n\n"
#         "b. A model with a lower AIC (Akaike Information Criterion) is generally preferred because it balances simplicity and likelihood, reducing the risk of overfitting."
#     )
# }

# midterm_1_q20 = {
#     'question': (
#         "A pharmaceutical company produces medications in batches of 200 units, with each batch taking an average of 7 days to complete. They have data on the sequence number of each unit in the batch, the day it was completed within the batch, and the time until the first reported efficacy drop in patients. The company plans to use triple exponential smoothing to analyze patterns in the time until efficacy drop based on a unit’s sequence number in its batch."
#         "\n\n"
#         "a. The observed variable (y_t) in the context of this study:"
#         "\n- Sequence number in batch"
#         "\n- Day within batch that unit was completed"
#         "\n- Time until first reported efficacy drop\n\n"
#         "b. The seasonal length (L) in this analysis:"
#         "\n- 200 (number of units in a batch)"
#         "\n- 7 (days to complete a batch)\n\n"
#         "c. If the seasonal component (C_t) values are consistently lower towards the end of the batch, it suggests:"
#         "\n- Units produced later in a batch tend to show efficacy drop more quickly."
#         "\n- Units produced later in a batch tend to maintain efficacy longer."
#         "\n- This question is designed to assess the application of time series analysis in a production context and the interpretation of its results."
#     ),
#     'correct_answer': (
#         "a. Time until first reported efficacy drop\n\n"
#         "b. 7 (days to complete a batch)\n\n"
#         "c. Units produced later in a batch tend to show efficacy drop more quickly."
#     ),
#     'explanation': (
#         "a. The observed variable (y_t) is the primary outcome of interest, which in this context is the time until the first reported efficacy drop."
#         "\n\n"
#         "b. The seasonal length (L) in triple exponential smoothing should correspond to the recurring pattern, which in this case is the 7-day cycle to complete a batch."
#         "\n\n"
#         "c. If the seasonal component (C_t) is lower towards the end of the batch, it indicates that units produced later in the batch tend to show efficacy drop more quickly, suggesting potential issues with production consistency."
#     )
# }

# midterm_1_q17 = {
#     'question': (
#         "A regional supermarket chain has collected day-to-day data over the last five years (approximately 1800 data points)."
#         "\n\n"
#         "x1 = Number of customers visiting the store that day"
#         "\nx2 = Day of the week"
#         "\nx3 = Whether the day was part of a promotional event"
#         "\nx4 = Local unemployment rate on that day"
#         "\nx5 = Average temperature on that day"
#         "\nx6 = Local sports team win or loss on the previous day\n\n"
#         "a. Select all data that are categorical."
#         "\n"
#         "x2\n"
#         "x3\n"
#         "x6\n\n"
#         "The supermarket has built three models using the linear formula b0 + b1x1 + b2x2 + b3x3 + b4x4 + b5x5 + b6x6."
#         "\n\n"
#         "1. Linear Regression"
#         "\n2. Logistic Regression"
#         "\n3. K-nearest neighbors\n\n"
#         "b. For each of the following scenarios (i-iii), which model (1, 2, or 3) would you suggest using?"
#         "\n"
#         "i. The supermarket wants to estimate the total number of customers visiting the store each day."
#         "\n- Linear Regression\n\n"
#         "ii. The supermarket aims to predict the likelihood of having more than 500 customers in the store each day."
#         "\n- Logistic Regression\n\n"
#         "iii. The supermarket seeks to classify days into high or low customer traffic based on a threshold of 500 customers."
#         "\n- K-nearest neighbors\n\n"
#         "A regional supermarket chain has implemented a triple exponential smoothing (Holt-Winters) model to forecast the number of customers visiting the store each day. The model includes a multiplicative seasonal pattern with a weekly cycle (i.e., L=7)."
#         "\n\n"
#         "i. What should they expect the best value of α to be, considering the consistency in customer visits?"
#         "\n- 0 < α < ½"
#     ),
#     'correct_answer': (
#         "a. x2, x3, x6\n\n"
#         "i. Linear Regression\n\n"
#         "ii. Logistic Regression\n\n"
#         "iii. K-nearest neighbors\n\n"
#         "i. 0 < α < ½"
#     ),
#     'explanation': (
#         "a. The categorical variables in this dataset are the day of the week (x2), whether the day was part of a promotional event (x3), and the local sports team win or loss on the previous day (x6)."
#         "\n\n"
#         "i. Linear Regression is appropriate for estimating the total number of customers visiting the store each day, given its continuous output."
#         "\n\n"
#         "ii. Logistic Regression is suitable for predicting the likelihood of having more than 500 customers, as it involves a binary classification."
#         "\n\n"
#         "iii. K-nearest neighbors is useful for classifying days into high or low customer traffic based on a threshold, given its non-parametric nature."
#         "\n\n"
#         "i. In triple exponential smoothing, an alpha value between 0 and ½ is generally optimal when there's consistent data with minimal random variation."
#     )
# }

# midterm_1_q18 = {
#     'question': (
#         "A regional healthcare provider has collected extensive data on patient visits over the years, including patient demographics, symptoms, diagnoses, treatments, and outcomes. The organization now wants to leverage this data to predict patient readmission risks and identify key factors that contribute to higher readmission rates."
#         "\n\n"
#         "Choose the appropriate models/approaches from the list below that the healthcare provider could use for predicting patient readmissions and understanding the underlying factors."
#         "\n"
#         "- CUSUM\n"
#         "- K-nearest-neighbor classification\n"
#         "- Logistic Regression\n"
#         "- Multi-armed bandit\n"
#         "- Support Vector Machine\n"
#     ),
#     'correct_answer': (
#         "- Logistic Regression\n"
#         "- K-nearest-neighbor classification\n"
#         "- Support Vector Machine"
#     ),
#     'explanation': (
#         "Logistic Regression is suitable for predicting patient readmission risks due to its binary classification approach. It helps identify key factors contributing to higher readmission rates."
#         "\n\n"
#         "K-nearest-neighbor classification can be used to predict patient readmissions by finding the closest matches in the dataset, aiding in understanding patterns."
#         "\n\n"
#         "Support Vector Machine is also suitable for binary classification tasks, providing robust decision boundaries for predicting patient readmission risks."
#     )
# }
# midterm_1_q16 = """
# Question A1\n
# A company has noticed an increasing trend in customer service calls on Mondays over the past 15 years. The company wants to analyze whether there has been a significant change in this Monday trend in customer service calls during this period. Select all of the approaches that might reasonably be correct.
# \n
# i. Develop 15 separate logistic regression models, one for each year, with "is it Monday?" as one of the predictor variables; then apply a CUSUM analysis on the yearly coefficients for the Monday variable.
# \n
# ii. Implement time series forecasting using ARIMA, focusing on Mondays for the 780 weeks, and then use CUSUM on the forecasted values to identify any significant shifts.
# \n
# iii. Apply CUSUM directly on the volume of customer service calls received each of the 780 Mondays over the past 15 years.
# """

# midterm_1_q13 = """
# Confusion Matrix for Shoplifting Prediction Model:
# =======================================================================
#                        Predicted Not Shoplifting   Predicted Shoplifting
# Actual Not Shoplifting            1200                       300
# Actual Shoplifting                 150                       350
# =======================================================================

# This confusion matrix represents the outcomes of a shoplifting prediction model. The model predicts whether an individual is likely to commit shoplifting ('Predicted Shoplifting')
# or not ('Predicted Not Shoplifting'), and the results are compared against the actual occurrences ('Actual Shoplifting' and 'Actual Not Shoplifting').

# Questions about the Shoplifting Prediction Model's Confusion Matrix:

# Question A1:\n
# Calculate the model's accuracy (the proportion of true results among the total number of cases examined).
# \nA) (1200 + 350) / (1200 + 300 + 150 + 350)
# \nB) (1200 + 150) / (1200 + 300 + 150 + 350)
# \nC) (300 + 350) / (1200 + 300 + 150 + 350)

# Question A2:\n
# Determine the model's precision for shoplifting predictions (the proportion of correctly predicted shoplifting incidents to the total predicted as shoplifting).
# \nA) 350 / (300 + 350)
# \nB) 1200 / (1200 + 150)
# \nC) 350 / (1200 + 350)

# Question A3:\n
# Calculate the model's recall for shoplifting predictions (the ability of the model to identify actual shoplifting incidents).
# \nA) 350 / (150 + 350)
# \nB) 300 / (1200 + 300)
# \nC) 1200 / (1200 + 150)

# Question A4:\n
# Based on the confusion matrix, which statement is true regarding the model's predictions?
# \nA) The model is more accurate in predicting non-shoplifting incidents than shoplifting incidents.
# \nB) The model has the same accuracy for predicting shoplifting and non-shoplifting incidents.
# \nC) The model is more accurate in predicting shoplifting incidents than non-shoplifting incidents.

# """
# midterm_1_q14 = """
# A1\n
# Matching\n
# Choices:\n
# \nA. Classification
# \nB. Clustering
# \nC. Dimensionality Reduction
# \nD. Outlier Detection

# \nA1. Astronomers have a collection of long-exposure CCD images of distant celestial objects. They are unsure about the types of these objects and seek to group similar ones together. Which method is more suitable?
# \nA2. An astronomer has manually categorized hundreds of images and now wishes to use analytics to automatically categorize new images. Which approach is most fitting?
# \nA3. A data scientist wants to reduce the complexity of a high-dimensional dataset to visualize it more effectively, while preserving as much information as possible. Which technique should they use?
# \nA4. A financial analyst is examining a large set of transaction data to identify unusual transactions that might indicate fraudulent activity. Which method is most appropriate?
# """

# midterm_1_q1_5 = """
# Question A1:\n

# A retail company operates in various regions and is interested in optimizing its inventory management. The company has historical sales data from multiple stores and wants to predict future sales volumes for each product in different locations. This forecasting will help in efficient stock allocation and reducing inventory costs. The company also wants to understand the factors influencing sales to make strategic decisions. Which of the following models/approaches could the company use to predict future sales and understand sales dynamics?

# CUSUM:\n
# \nDiscrete event simulation: A simulation that models a system that changes when specific events occur.
# \nLinear Regression:
# \nLogistic Regression Tree:
# \nRandom Linear Regression Forest:
# """

# midterm_1_q13 = {
#     'question': (
#         "Confusion Matrix for Shoplifting Prediction Model:"
#         "\n======================================================================="
#         "\n                       Predicted Not Shoplifting   Predicted Shoplifting"
#         "\nActual Not Shoplifting            1200                       300"
#         "\nActual Shoplifting                 150                       350"
#         "\n======================================================================="
#         "\n\n"
#         "This confusion matrix represents the outcomes of a shoplifting prediction model. The model predicts whether an individual is likely to commit shoplifting ('Predicted Shoplifting') or not ('Predicted Not Shoplifting'), and the results are compared against the actual occurrences ('Actual Shoplifting' and 'Actual Not Shoplifting')."
#         "\n\n"
#         "Questions about the Shoplifting Prediction Model's Confusion Matrix:"
#         "\n\n"
#         "Question A1:\n"
#         "Calculate the model's accuracy (the proportion of true results among the total number of cases examined)."
#         "\n"
#         "A) (1200 + 350) / (1200 + 300 + 150 + 350)"
#         "\nB) (1200 + 150) / (1200 + 300 + 150 + 350)"
#         "\nC) (300 + 350) / (1200 + 300 + 150 + 350)\n\n"
#         "Question A2:\n"
#         "Determine the model's precision for shoplifting predictions (the proportion of correctly predicted shoplifting incidents to the total predicted as shoplifting)."
#         "\n"
#         "A) 350 / (300 + 350)"
#         "\nB) 1200 / (1200 + 150)"
#         "\nC) 350 / (1200 + 350)\n\n"
#         "Question A3:\n"
#         "Calculate the model's recall for shoplifting predictions (the ability of the model to identify actual shoplifting incidents)."
#         "\n"
#         "A) 350 / (150 + 350)"
#         "\nB) 300 / (1200 + 300)"
#         "\nC) 1200 / (1200 + 150)\n\n"
#         "Question A4:\n"
#         "Based on the confusion matrix, which statement is true regarding the model's predictions?"
#         "\n"
#         "A) The model is more accurate in predicting non-shoplifting incidents than shoplifting incidents."
#         "\nB) The model has the same accuracy for predicting shoplifting and non-shoplifting incidents."
#         "\nC) The model is more accurate in predicting shoplifting incidents than non-shoplifting incidents."
#     ),
#     'correct_answer': (
#         "A1: A) (1200 + 350) / (1200 + 300 + 150 + 350)\n\n"
#         "A2: A) 350 / (300 + 350)\n\n"
#         "A3: A) 350 / (150 + 350)\n\n"
#         "A4: A) The model is more accurate in predicting non-shoplifting incidents than shoplifting incidents."
#     ),
#     'explanation': (
#         "A1: The model's accuracy is the proportion of true results among the total cases, which is calculated as (1200 + 350) / (1200 + 300 + 150 + 350), yielding 1550/2000, or 77.5%."
#         "\n\n"
#         "A2: Precision for shoplifting predictions is the proportion of correctly predicted shoplifting incidents to the total predicted as shoplifting, which is calculated as 350 / (300 + 350), resulting in 350/650, or approximately 53.85%."
#         "\n\n"
#         "A3: Recall for shoplifting predictions is the ability of the model to identify actual shoplifting incidents, calculated as 350 / (150 + 350), resulting in 350/500, or 70%."
#         "\n\n"
#         "A4: The model is more accurate in predicting non-shoplifting incidents, as it correctly identifies 1200 out of 1500 cases, giving a higher accuracy compared to shoplifting predictions."
#     )
# }

# midterm_1_q14 = {
#     'question': (
#         "A1\n"
#         "Matching\n"
#         "Choices:\n"
#         "\nA. Classification"
#         "\nB. Clustering"
#         "\nC. Dimensionality Reduction"
#         "\nD. Outlier Detection\n\n"
#         "A1. Astronomers have a collection of long-exposure CCD images of distant celestial objects. They are unsure about the types of these objects and seek to group similar ones together. Which method is more suitable?"
#         "\n\n"
#         "A2. An astronomer has manually categorized hundreds of images and now wishes to use analytics to automatically categorize new images. Which approach is most fitting?"
#         "\n\n"
#         "A3. A data scientist wants to reduce the complexity of a high-dimensional dataset to visualize it more effectively, while preserving as much information as possible. Which technique should they use?"
#         "\n\n"
#         "A4. A financial analyst is examining a large set of transaction data to identify unusual transactions that might indicate fraudulent activity. Which method is most appropriate?"
#     ),
#     'correct_answer': (
#         "A1: B. Clustering\n\n"
#         "A2: A. Classification\n\n"
#         "A3: C. Dimensionality Reduction\n\n"
#         "A4: D. Outlier Detection"
#     ),
#     'explanation': (
#         "A1: Clustering is suitable for grouping similar objects, making it ideal for astronomers seeking to group celestial objects based on their characteristics."
#         "\n\n"
#         "A2: Classification is appropriate for automatically categorizing new images based on manually categorized examples."
#         "\n\n"
#         "A3: Dimensionality Reduction helps reduce the complexity of a high-dimensional dataset for visualization, preserving key information."
#         "\n\n"
#         "A4: Outlier Detection is designed to identify unusual data points or anomalies, which is ideal for identifying potential fraudulent activity in transaction data."
#     )
# }

# midterm_1_q15 = {
#     'question': (
#         "A retail company operates in various regions and is interested in optimizing its inventory management. The company has historical sales data from multiple stores and wants to predict future sales volumes for each product in different locations. This forecasting will help in efficient stock allocation and reducing inventory costs."
#         "\n\n"
#         "The company also wants to understand the factors influencing sales to make strategic decisions. Which of the following models/approaches could the company use to predict future sales and understand sales dynamics?"
#         "\n"
#         "- CUSUM\n"
#         "- Discrete event simulation\n"
#         "- Linear Regression\n"
#         "- Logistic Regression\n"
#         "- Random Forest\n"
#     ),
#     'correct_answer': (
#         "- Linear Regression\n"
#         "- Random Forest\n"
#         "- Discrete event simulation"
#     ),
#     'explanation': (
#         "Linear Regression can predict future sales volumes based on historical data, providing a straightforward regression model for forecasting."
#         "\n\n"
#         "Random Forest is useful for predicting sales dynamics, given its ensemble nature, which helps capture complex relationships."
#         "\n\n"
#         "Discrete event simulation can model changes in a system when specific events occur, useful for understanding inventory management and stock allocation."
#     )
# }

# midterm_1_q10 = {
#     'question': (
#         "Model Suitability Analysis\n\n"
#         "For each statistical and machine learning model listed below, select the type of analysis it is best suited for."
#         " There may be more than one correct answer for each model, but you need only choose one. Assume 1 of 1 attempt for each part."
#         "\n\n"
#         "A1. Ridge Regression"
#         "\n- Predicting a continuous response variable with feature data."
#         "\n- Dealing with multicollinearity in regression analysis."
#         "\n- Forecasting future values in a time-series dataset."
#         "\n- Classifying binary outcomes."
#         "\n\n"
#         "A2. Lasso Regression"
#         "\n- Selecting important features in a large dataset."
#         "\n- Predicting a numerical outcome based on feature data."
#         "\n- Analyzing patterns in time-series data."
#         "\n- Identifying categories in unstructured data."
#         "\n\n"
#         "A3. Principal Component Analysis (PCA)"
#         "\n- Reducing the dimensionality of a large dataset."
#         "\n- Forecasting trends in a time-series dataset."
#         "\n- Classifying items into categories based on feature data."
#         "\n- Detecting changes in the variance of a dataset over time."
#     ),
#     'correct_answer': (
#         "A1: Dealing with multicollinearity in regression analysis\n\n"
#         "A2: Selecting important features in a large dataset\n\n"
#         "A3: Reducing the dimensionality of a large dataset"
#     ),
#     'explanation': (
#         "A1: Ridge Regression is best suited for dealing with multicollinearity in regression analysis because it adds a regularization term that reduces the impact of correlated variables."
#         "\n\n"
#         "A2: Lasso Regression is ideal for selecting important features in a large dataset because it adds a regularization term that can shrink some coefficients to zero, effectively removing less important features."
#         "\n\n"
#         "A3: Principal Component Analysis (PCA) is designed to reduce dimensionality by transforming the original dataset into a smaller set of uncorrelated variables, retaining the most information."
#     )
# }

# midterm_1_q11 = {
#     'question': (
#         "Model Suitability Analysis\n\n"
#         "For each statistical and machine learning model listed below, select the type of analysis it is best suited for."
#         " There may be more than one correct answer for each model, but you need only choose one. Assume 1 of 1 attempt for each part."
#         "\n\n"
#         "A1. Decision Trees (e.g., CART)"
#         "\n- Predicting the category of an item based on feature data."
#         "\n- Forecasting numerical values two time periods in the future."
#         "\n- Identifying clusters in feature data."
#         "\n- Analyzing the variance in time-series data."
#         "\n\n"
#         "A2. Random Forest"
#         "\n- Predicting the likelihood of an event based on feature data."
#         "\n- Classifying items into categories based on feature data."
#         "\n- Estimating the amount of a variable two time periods in the future using time-series data."
#         "\n- Detecting patterns in large datasets with many variables."
#         "\n\n"
#         "A3. Naive Bayes Classifier"
#         "\n- Classifying text data into predefined categories."
#         "\n- Predicting future trends based on historical time-series data."
#         "\n- Estimating the probability of an event occurring in the future."
#         "\n- Analyzing variance in feature data."
#     ),
#     'correct_answer': (
#         "A1: Predicting the category of an item based on feature data\n\n"
#         "A2: Detecting patterns in large datasets with many variables\n\n"
#         "A3: Classifying text data into predefined categories"
#     ),
#     'explanation': (
#         "A1: Decision Trees are well-suited for predicting the category of an item based on feature data, providing a clear visual representation of the decision-making process."
#         "\n\n"
#         "A2: Random Forest is designed to detect patterns in large datasets with many variables, using an ensemble of decision trees to improve prediction accuracy."
#         "\n\n"
#         "A3: Naive Bayes Classifier is optimal for classifying text data into predefined categories, given its probabilistic approach that assumes independence among features."
#     )
# }

# midterm_1_q12 = {
#     'question': (
#         "A1\n"
#         "Select all of the following reasons that data should not be scaled until point outliers are removed:"
#         "\n"
#         "- If data is scaled first, the range of data after outliers are removed will be narrower than intended."
#         "\n- If data is scaled first, the range of data after outliers are removed will be wider than intended."
#         "\n- Point outliers would appear to be valid data if not removed before scaling."
#         "\n- Valid data would appear to be outliers if data is scaled first.\n\n"
#         "A2\n"
#         "Select all of the following situations in which using a variable selection approach like lasso or stepwise regression would be important:"
#         "\n"
#         "- It is too costly to create a model with a large number of variables."
#         "\n- There are too few data points to avoid overfitting if all variables are included."
#         "\n- Time-series data is being used."
#         "\n- There are fewer data points than variables."
#     ),
#     'correct_answer': (
#         "A1: If data is scaled first, the range of data after outliers are removed will be narrower than intended"
#         "\n\n"
#         "A2: It is too costly to create a model with a large number of variables"
#         "\n- There are too few data points to avoid overfitting if all variables are included"
#     ),
#     'explanation': (
#         "A1: If data is scaled first, the range of data after outliers are removed will be narrower than intended, affecting the scaling process."
#         "\n\n"
#         "A2: Using variable selection approaches like lasso or stepwise regression is important when it is too costly to create a model with many variables, or when there are too few data points to avoid overfitting."
#     )
# }


# midterm_1_q9 = {
#     'question': (
#         "Model Suitability Analysis\n\n"
#         "For each statistical and machine learning model listed below, select the type of analysis it is best suited for. "
#         "There may be more than one correct answer for each model, but you need only choose one. Assume 1 of 1 attempt for each part."
#         "\n\n"
#         "A1. Time Series Analysis (e.g., ARMA, ARIMA)"
#         "\n- Predicting future values in a time-series dataset."
#         "\n- Classifying items based on time-dependent features."
#         "\n- Analyzing the seasonal components of time-series data."
#         "\n- Estimating the probability of an event occurring in the future."
#         "\n\n"
#         "A2. k-Nearest-Neighbor Classification (kNN)"
#         "\n- Using feature data to predict whether or not something will happen two time periods in the future."
#         "\n- Using feature data to predict the probability of something happening two time periods in the future."
#         "\n- Using time-series data to predict the amount of something two time periods in the future."
#         "\n- Using time-series data to predict the variance of something two time periods in the future."
#         "\n\n"
#         "A3. Exponential Smoothing"
#         "\n- Using time-series data to predict the amount of something two time periods in the future."
#         "\n- Analyzing the seasonal components of time-series data."
#         "\n- Using time-series data to predict future trends."
#     ),
#     'correct_answer': (
#         "A1: Predicting future values in a time-series dataset\n\n"
#         "A2: Using feature data to predict whether or not something will happen two time periods in the future\n\n"
#         "A3: Using time-series data to predict the amount of something two time periods in the future"
#     ),
#     'explanation': (
#         "A1: Time Series Analysis, such as ARIMA, is most suitable for predicting future values in a time-series dataset, providing forecasting capabilities with seasonal components."
#         "\n\n"
#         "A2: k-Nearest-Neighbor Classification (kNN) is best for using feature data to predict binary outcomes, as it relies on the proximity to known data points to classify new data."
#         "\n\n"
#         "A3: Exponential Smoothing is appropriate for predicting the amount of something in a time-series dataset, offering a smoothing approach to forecast trends and address variability."
#     )
# }

# midterm_1_q8 = {
#     'question': (
#         "For each scenario, identify the most relevant statistical measure: AIC (Akaike Information Criterion), R-squared, Specificity, or Variance. Variance is included as a distractor and may not be the correct answer."
#         "\n\n"
#         "Definitions:"
#         "\nAIC (Akaike Information Criterion): Balances the model's fit with the complexity by penalizing the number of parameters."
#         "\nR-squared: Measures the proportion of variance in the dependent variable explained by the independent variables."
#         "\nSpecificity: Not relevant in this context."
#         "\nVariance: Measures the dispersion of a set of data points."
#         "\n\n"
#         "Choices:"
#         "\nA. AIC"
#         "\nB. R-squared"
#         "\nC. Specificity"
#         "\nD. Variance"
#         "\n\n"
#         "Scenarios:"
#         "\n\n"
#         "Question A1:"
#         "A researcher is assessing various linear regression models to predict future profits of a company, aiming to find a balance between model complexity and fit."
#         "\n\n"
#         "Question A2:"
#         "In a study evaluating the effect of advertising on sales, the analyst seeks to understand how changes in advertising budgets correlate with variations in sales figures."
#         "\n\n"
#         "Question A3:"
#         "An economist is choosing among different models to forecast economic growth, with a focus on avoiding overfitting in the presence of many potential predictor variables."
#     ),
#     'correct_answer': (
#         "A1: AIC\n\n"
#         "A2: R-squared\n\n"
#         "A3: AIC"
#     ),
#     'explanation': (
#         "A1: AIC is the most relevant measure for assessing various linear regression models, as it balances model fit with complexity, helping prevent overfitting."
#         "\n\n"
#         "A2: R-squared is the appropriate measure for evaluating the effect of advertising on sales, indicating the proportion of variance in the dependent variable explained by the independent variables."
#         "\n\n"
#         "A3: AIC is suitable for selecting models that avoid overfitting when forecasting economic growth, providing a balance between model complexity and fit."
#     )
# }

# midterm_1_q6 = {
#     'question': (
#         "Information for all parts of the question\n\n"
#         "Atlanta’s main library has collected the following day-by-day data over the past six years (more than 2000 data points):"
#         "\n\n"
#         "x1 = Number of books borrowed from the library on that day"
#         "\nx2 = Day of the week"
#         "\nx3 = Temperature"
#         "\nx4 = Amount of rainfall"
#         "\nx5 = Whether the library was closed that day"
#         "\nx6 = Whether public schools were open that day"
#         "\nx7 = Number of books borrowed the day before"
#         "\nt = Time"
#         "\n\n"
#         "a. Select all data that are categorical (including binary data):"
#         "\n"
#         "- Day of the week"
#         "\n- Whether the library was closed that day"
#         "\n- Whether public schools were open that day"
#         "\n\n"
#         "b. If the library is correct that on average, if more books were borrowed yesterday, more books will be borrowed today (and vice versa), what sign (positive or negative) would you expect the new predictor's coefficient β to have?"
#         "\n"
#         "- Positive, higher values of x7 increase the response (books borrowed today)"
#         "\n\n"
#         "c. Does x7 make the model autoregressive?"
#         "\n"
#         "- Yes, because the model uses day t-1 borrowing data to predict day t borrowing."
#     ),
#     'correct_answer': (
#         "a: - Day of the week, - Whether the library was closed that day, - Whether public schools were open that day\n\n"
#         "b: Positive, higher values of x7 increase the response (books borrowed today)\n\n"
#         "c: Yes, because the model uses day t-1 borrowing data to predict day t borrowing."
#     ),
#     'explanation': (
#         "a: Categorical data consists of discrete groups or binary values, which includes 'Day of the week,' 'Whether the library was closed that day,' and 'Whether public schools were open that day.'"
#         "\n\n"
#         "b: If more books borrowed yesterday leads to more books borrowed today, the expected sign for the coefficient β is positive, indicating a direct relationship."
#         "\n\n"
#         "c: A model is autoregressive if it uses previous response data to predict future responses. Using day t-1 borrowing data to predict day t borrowing makes the model autoregressive."
#     )
# }

# midterm_1_q7 = {
#     'question': (
#         "Select all of the following statements that are correct:"
#         "\n\n"
#         "- It is likely that the first principal component has much more predictive power than each of the other principal components."
#         "\n- It is likely that the first original covariate has much more predictive power than each of the other covariates."
#         "\n- It is likely that the last original covariate has much less predictive power than each of the other covariates."
#         "\n- The first principal component cannot contain information from all 7 original covariates. (correct)"
#     ),
#     'correct_answer': (
#         "- The first principal component cannot contain information from all 7 original covariates."
#     ),
#     'explanation': (
#         "The first principal component (PCA) captures the most variance in the dataset but cannot contain all the information from the original covariates. It represents a linear combination of original variables, but not the complete set."
#     )
# }


# midterm_1_q5 = {
#     'question': (
#         "An airline wants to predict airline passenger traffic for the upcoming year."
#         " For each of the specific questions (a-e) listed below, identify the question type (i-viii) it corresponds to."
#         " If a question does not match any of the listed types, leave it uncircled."
#         "\n\n"
#         "Question Types:"
#         "\n"
#         "i. Change detection"
#         "\nii. Classification"
#         "\niii. Clustering"
#         "\niv. Feature-based prediction of a value"
#         "\nv. Feature-based prediction of a probability"
#         "\nvi. Time-series-based prediction"
#         "\nvii. Validation"
#         "\nviii. Variance estimation"
#         "\n\n"
#         "Questions:"
#         "\na. What is the probability that the airline will exceed 1 million passengers next year, considering current travel trends and economic factors?"
#         "\nb. Among various forecasting models for airline passenger traffic, which one is likely to be the most accurate for the upcoming year?"
#         "\nc. Based on the past decade's data, how many passengers are expected to travel via the airline next year?"
#         "\nd. Analyzing the past fifteen years of data, has there been a significant change in passenger traffic during holiday seasons?"
#         "\ne. Considering economic indicators and travel trends over the past 25 years, which years had the most similar passenger traffic patterns?"
#     ),
#     'correct_answer': (
#         "a: v. Feature-based prediction of a probability\n\n"
#         "b: vii. Validation\n\n"
#         "c: vi. Time-series-based prediction\n\n"
#         "d: i. Change detection\n\n"
#         "e: iii. Clustering"
#     ),
#     'explanation': (
#         "a: This question involves predicting the probability of an event occurring, aligning with 'Feature-based prediction of a probability.'"
#         "\n\n"
#         "b: This question seeks the most accurate forecasting model, relating to 'Validation.'"
#         "\n\n"
#         "c: This question requires predicting a future value based on time-series data, which is 'Time-series-based prediction.'"
#         "\n\n"
#         "d: Analyzing significant changes over time corresponds to 'Change detection.'"
#         "\n\n"
#         "e: Identifying years with similar passenger traffic patterns corresponds to 'Clustering.'"
#     )
# }

# midterm_1_q3 = {
#     'question': (
#         "Question A1"
#         "\nIn the soft classification SVM model where we select coefficients to minimize the following formula:"
#         "\nΣ_{j=1}^n max{0, 1 - (Σ_{i=1}^m a_ix_ij + a_0)y_j} + C Σ_{i=1}^m a_i^2"
#         "\nSelect all of the following statements that are correct."
#         "\n\n"
#         "- Decreasing the value of C could decrease the margin."
#         "\n- Allowing a larger margin could decrease the number of classification errors in the training set."
#         "\n- Decreasing the value of C could increase the number of classification errors in the training set."
#         "\n\n"
#         "Question A2"
#         "\nIn the hard classification SVM model, it might be desirable to not put the classifier in a location that has equal margin on both sides... (select all correct answers):"
#         "\n\n"
#         "- ...because moving the classifier will usually result in fewer classification errors in the validation data."
#         "\n- ...because moving the classifier will usually result in fewer classification errors in the test data."
#         "\n- ...when the costs of misclassifying the two types of points are significantly different."
#     ),
#     'correct_answer': (
#         "A1: - Decreasing the value of C could increase the number of classification errors in the training set\n\n"
#         "A2: - ...when the costs of misclassifying the two types of points are significantly different."
#     ),
#     'explanation': (
#         "A1: In soft classification SVM, decreasing the value of C leads to a larger margin, increasing the possibility of misclassifications in the training set."
#         "\n\n"
#         "A2: In hard classification SVM, unequal margins may be desirable when the costs of misclassification differ between the two types of points."
#     )
# }

# midterm_1_q4 = {
#     'question': (
#         "Select whether a supervised learning model (like regression) is more directly appropriate than an unsupervised learning model (like dimensionality reduction)."
#         "\n\n"
#         "Definitions:"
#         "\nSupervised Learning: Machine learning where the 'correct' answer or outcome is known for each data point in the training set."
#         "\nRegression: A type of supervised learning where the model predicts a continuous outcome."
#         "\nUnsupervised Learning Model: Machine learning where the 'correct' answer is not known for the data points in the training set."
#         "\nDimensionality Reduction: A process in unsupervised learning of reducing the number of random variables under consideration, through feature selection and feature extraction."
#         "\n\n"
#         "Questions:"
#         "\n\n"
#         "- In a dataset of residential property sales, for each property, the sale price is known, and the goal is to predict prices for new listings based on various attributes like location, size, and amenities."
#         "\n\n"
#         "- In a large dataset of customer reviews, there is no specific response variable, but the goal is to understand underlying themes and patterns in the text data."
#         "\n\n"
#         "- In a clinical trial dataset, for each participant, the response to a medication is known, and the task is to predict patient outcomes based on their medical history and trial data."
#     ),
#     'correct_answer': (
#         "- In a dataset of residential property sales, for each property, the sale price is known, and the goal is to predict prices for new listings based on various attributes like location, size, and amenities."
#         "\n- In a clinical trial dataset, for each participant, the response to a medication is known, and the task is to predict patient outcomes based on their medical history and trial data."
#     ),
#     'explanation': (
#         "Supervised learning is appropriate when the 'correct' outcome is known, such as predicting property sale prices based on known values and forecasting patient outcomes based on medical history."
#         "\n\n"
#         "Unsupervised learning is appropriate when the 'correct' outcome is unknown, such as identifying themes and patterns in text data."
#     )
# }

# midterm_1_q1 = {
#     'question': (
#         "For each of the models (a-m) below, circle one type of question (i-viii) it is commonly used for."
#         " For models that have more than one correct answer, choose any one correct answer."
#         " If there is no correct answer listed, do not circle anything."
#         "\n\n"
#         "Models:"
#         "\na. ARIMA"
#         "\nb. CART"
#         "\nc. Cross validation"
#         "\nd. CUSUM"
#         "\ne. Exponential smoothing"
#         "\nf. GARCH"
#         "\ng. k-means"
#         "\nh. k-nearest-neighbor"
#         "\ni. Linear regression"
#         "\nj. Logistic regression"
#         "\nk. Principal component analysis"
#         "\nl. Random forest"
#         "\nm. Support vector machine"
#         "\n\n"
#         "Question Types:"
#         "\ni. Change detection"
#         "\nii. Classification"
#         "\niii. Clustering"
#         "\niv. Feature-based prediction of a probability"
#         "\nv. Feature-based prediction of a value"
#         "\nvi. Time-series-based prediction"
#         "\nvii. Validation"
#         "\nviii. Variance estimation"
#     ),
#     'correct_answer': (
#         "a. vi. Time-series-based prediction\n\n"
#         "b. ii. Classification\n\n"
#         "c. vii. Validation\n\n"
#         "d. i. Change detection\n\n"
#         "e. vi. Time-series-based prediction\n\n"
#         "f. viii. Variance estimation\n\n"
#         "g. iii. Clustering\n\n"
#         "h. ii. Classification\n\n"
#         "i. v. Feature-based prediction of a value\n\n"
#         "j. iv. Feature-based prediction of a probability\n\n"
#         "k. iii. Clustering\n\n"
#         "l. ii. Classification\n\n"
#         "m. ii. Classification"
#     ),
#     'explanation': (
#         "a. ARIMA is used for time-series-based prediction, analyzing data with trends and seasonality."
#         "\n\n"
#         "b. CART (Classification and Regression Trees) is commonly used for classification, creating decision trees to separate data into distinct groups."
#         "\n\n"
#         "c. Cross validation is used for validation, ensuring model robustness and preventing overfitting."
#         "\n\n"
#         "d. CUSUM (Cumulative Sum) is designed for change detection, identifying shifts in data over time."
#         "\n\n"
#         "e. Exponential smoothing is also used for time-series-based prediction, applying a smoothing factor to forecast trends."
#         "\n\n"
#         "f. GARCH (Generalized Autoregressive Conditional Heteroskedasticity) is used for variance estimation, modeling volatility in financial data."
#         "\n\n"
#         "g. k-means is a clustering method, creating groups based on data similarities."
#         "\n\n"
#         "h. k-nearest-neighbor is used for classification, classifying data based on proximity to other data points."
#         "\n\n"
#         "i. Linear regression is used for feature-based prediction of a value, predicting a continuous outcome."
#         "\n\n"
#         "j. Logistic regression is for feature-based prediction of a probability, predicting binary outcomes."
#         "\n\n"
#         "k. Principal component analysis (PCA) is typically used for clustering, reducing dimensionality and identifying key components."
#         "\n\n"
#         "l. Random forest is often used for classification, employing an ensemble of decision trees."
#         "\n\n"
#         "m. Support vector machine (SVM) is used for classification, creating decision boundaries to separate data."
#     )
# }

# midterm_1_q2 = {
#     'question': (
#         "Select all of the following models that are designed for use with attribute/feature data (i.e., not time-series data):"
#         "\n"
#         "- CUSUM"
#         "\n- Logistic regression"
#         "\n- Support vector machine"
#         "\n- GARCH"
#         "\n- Random forest"
#         "\n- k-means"
#         "\n- Linear regression"
#         "\n- k-nearest-neighbor"
#         "\n- ARIMA"
#         "\n- Principal component analysis"
#         "\n- Exponential smoothing"
#     ),
#     'correct_answer': (
#         "- CUSUM\n"
#         "- Logistic regression\n"
#         "- Support vector machine\n"
#         "- Random forest\n"
#         "- k-means\n"
#         "- Linear regression\n"
#         "- k-nearest-neighbor\n"
#         "- Principal component analysis"
#     ),
#     'explanation': (
#         "These models are designed for use with attribute/feature data, not time-series data:"
#         "\n\n"
#         "- CUSUM is for change detection in attribute/feature data."
#         "\n\n"
#         "- Logistic regression, Support vector machine, Random forest, and k-nearest-neighbor are primarily used for classification."
#         "\n\n"
#         "- Linear regression is for predicting a continuous outcome based on feature data."
#         "\n\n"
#         "- k-means and Principal component analysis (PCA) are for clustering and dimensionality reduction, respectively."
#     )
# }


OPEN_QUESTIONS = []
global_items = list(globals().items())

for name, value in global_items:
    if not name.startswith('_'):
        OPEN_QUESTIONS.append(value)

OPEN_QUESTIONS = OPEN_QUESTIONS[:-1]
