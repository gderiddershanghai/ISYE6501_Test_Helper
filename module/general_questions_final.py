
midterm_2_q1 =     """
Information for Question A:\n
There are five questions labeled "Question 1." Answer all five questions. For each of the following five questions,
select the probability distribution that could best be used to model the described scenario. Each distribution might
be used, zero, one, or more than one time in the five questions.
\n
Choose from Binomial, Poisson, Weibull, Exponential, Geometric, Bernoulli Distributions
\n
Question A1\n
Time between people entering a grocery store
\n
Question A2\n
Number of hits to a real estate web site each minute
\n
Question A3\n
Number of penalty kicks that are saved by the goalkeeper, out of the first 96 taken
\n
Question A4\n
Number of faces correctly identified by deep learning (DL) software until an error is made
\n
Question A5\n
Time between hits on a real estate web site
\n
"""

midterm_2_q2 ="""
Match the problem to the distribution:\n
Choose from Binomial, Poisson, Weibull, Exponential, Geometric, Bernoulli Distributions
\n
Question A1:\n
Lifetime of light bulbs produced in a factory.
\n
Question A2:\n
Number of customers arriving at a bank per hour.
\n
Question A3:\n
Number of correct answers in a multiple-choice exam with 50 questions, each with four options, where a student guesses all answers.
\n
Question A4:\n
Time until a newly launched website receives its first hundred visitors.
\n
Question A5:\n
The number of trials needed for a new machine to successfully complete its first task without error.

"""


midterm_2_q3 = """
Question A1: \nA retail company wants to optimize the layout of its warehouse to reduce the time it takes for workers to pick items for shipping. The layout optimization should consider the frequency of item requests and the physical distance between items.
a. Select all of the models/approaches the company could use to optimize the warehouse layout for efficient item picking, considering the frequency of item requests and physical distances:\n
1.	Logistic Regression\n
2.	Clustering\n
3.	Network Optimization\n
4.	Ridge Regression\n
5.	Integer Programming\n

Suppose the company also wants to forecast future item request patterns to further refine the warehouse layout. This forecast should be based on historical sales data, seasonal trends, and promotional activities.
b. Select all of the models/approaches the company could use to forecast future item request patterns:\n
1.	Time Series Analysis\n
2.	Support Vector Machine\n
3.	Lasso Regression\n
4.	Stochastic Optimization\n
5.	Elastic Net\n
"""

midterm_2_q4 = """
Question A1: \nA healthcare provider aims to optimize its staff scheduling to ensure adequate coverage across various departments, taking into account staff availability, skills, and shift preferences.
a. Select all of the models/approaches the provider could use for optimal staff scheduling, considering staff availability, skills, and shift preferences:\n
1.	Linear Programming\n
2.	Clustering\n
3.	Dynamic Programming\n
4.	Random Forest\n
5.	Integer Programming\n
Given the uncertainty in patient inflow and emergency cases, the healthcare provider also wants to prepare a robust staffing plan that can adapt to varying patient loads.
b. Select all of the models/approaches the provider could use to create a robust and adaptable staffing plan:\n
1.	Stochastic Optimization\n
2.	Logistic Regression\n
3.	Scenario Modeling\n
4.	Convex Optimization\n
5.	Elastic Net\n

"""
midterm_2_q5 = """
A regional airport is evaluating the need to expand its terminal facilities. The airport management team wants to use data analytics to determine the optimal number of additional gates required. They have a comprehensive dataset of flight and passenger numbers over the past twenty years. However, there is a challenge: about 5% of the records are missing key information on the number of passengers per flight.

Question A1: \nThe airport's chief strategist proposes the following approach: GIVEN the historical data of flights and passenger numbers, USE a certain statistical method TO impute the missing passenger data. Then, GIVEN the complete dataset, USE a forecasting model TO predict the number of flights and passengers for the next decade. Lastly, GIVEN these forecasts, USE a planning algorithm TO determine the minimum number of additional gates needed to handle the peak passenger traffic with a certain efficiency level. Identify and discuss the potential flaws or considerations in this proposed strategy, focusing on the choice of the statistical method for imputation, the forecasting model, and the planning algorithm.\n
A. The use of multiple regression for imputing missing passenger data may not accurately reflect the complexities of passenger behavior and flight patterns.\n
B. Forecasting the number of flights and passengers for the next decade using ARIMA might not account for unpredictable factors such as economic fluctuations or changes in travel habits.\n
C. Implementing k-nearest neighbors (KNN) for data imputation could lead to biases if the missing data is not randomly distributed.\n
D. Applying Monte Carlo simulation for planning the number of gates might not adequately consider the variability in daily flight schedules and passenger numbers.\n
"""



# midterm_2_q5 = """
# Five classification models were built for predicting whether a neighborhood will soon see a large rise in home prices,
# based on public elementary school ratings and other factors. The training data set was missing the school rating variable
# for every new school (3 percent of the data points).

# Because ratings are unavailable for newly-opened schools, it is believed that locations that have recently experienced
# high population growth are more likely to have missing school rating data.

# Model 1 used imputation, filling in the missing data with the average school rating from the rest of the data.
# Model 2 used imputation, building a regression model to fill in the missing school rating data based on other variables.
# Model 3 used imputation, first building a classification model to estimate (based on other variables) whether a new
# school is likely to have been built as a result of recent population growth (or whether it has been built for another
# purpose, e.g. to replace a very old school), and then using that classification to select one of two regression models
# to fill in an estimate of the school rating; there are two different regression models (based on other variables),
# one for neighborhoods with new schools built due to population growth, and one for neighborhoods with new schools built
# for other reasons.
# Model 4 used a binary variable to identify locations with missing information.
# Model 5 used a categorical variable: first, a classification model was used to estimate whether a new school is likely
# to have been built as a result of recent population growth; and then each neighborhood was categorized as "data available",
# "missing, population growth", or "missing, other reason".

# a. If school ratings can be reasonably well-predicted from the other factors, and new schools built due to recent
# population growth cannot be reasonably well-classified using the other factors, which model would you recommend?
# - Model 1
# - Model 2
# - Model 3
# - Model 4
# - Model 5

# b. In which of the following situations would you recommend using Model 3? [All predictions and classifications below
# are using the other factors.]
# - Ratings can be well-predicted, and reasons for building schools can be well-classified.
# - Ratings can be well-predicted, and reasons for building schools cannot be well-classified.
# - Ratings cannot be well-predicted, and reasons for building schools can be well-classified.
# - Ratings cannot be well-predicted, and reasons for building schools cannot be well-classified.
# """

midterm_2_q6 = """
Question A1\n
a.
A hospital emergency department (ED) has created a stochastic discrete-event simulation model of the ED,
including patient arrivals, resource usage (rooms, doctors, etc.), and treatment duration.

EDs are not first-come-first-served; a patient who arrives with a more-serious condition will be treated
first, ahead of even long-waiting patients with less-serious conditions.

When a patient comes in, the ED will run the simulation to quickly give the patient an estimate of the
expected wait time before being treated.

How many times does the ED need to run the simulation for each new patient (i.e., how many replications
are needed)?\n
- Once, because the outcome will be the same each time.\n
- Many times, because of the variability and randomness.\n
- Once, because each patient is unique.\n

b.

Suppose it is discovered that simulated wait times in the hospital emergency department are 50 percent higher than actual wait times, on average. What would you recommend that they do?
- Scale down all estimates by a factor of 1/1.50 to get the average simulation estimates to match the average actual wait times.\n
- Investigate to see what's wrong with the simulation, because it's a poor match to reality.\n
- Use the 50percent-higher estimates, because that's what the simulation output is.\n
"""


midterm_2_q7 = """
Information for Question A1:\n
For each of the optimization problems below, select its most precise classification. In each model, x are the variables,
all other letters (a, b, c) refer to known data, and the values of c are all positive.

There are seven questions labeled "Question 5". Answer all seven questions. Each classification might be used, zero, one,
or more than one time in the seven questions.

Choices:
- Linear Program\n
- Convex Program\n
- Convex Quadratic Program\n
- General Nonconvex Program\n
- Integer Program\n

Question a:\n
Minimize the sum of (log(c_i) * x_i), subject to the sum of (a_ij * x_i) greater than or equal to b_j for all j, and all x_i greater than or equal to 0.
\n
Question b:\n
Maximize the sum of (c_i * x_i), subject to the sum of (a_ij * x_i) greater than or equal to b_j for all j, and all x_i greater than or equal to 0.
\n
Question c:\n
Minimize the sum of (c_i * x_i^2), subject to the sum of (a_ij * x_i) greater than or equal to b_j for all j, and all x_i greater than or equal to 0.
\n
Question d:\n
Maximize the sum of (c_i * x_i), subject to the sum of (a_ij * x_i) greater than or equal to b_j for all j, and all x_i belonging to {0, 1}.
"""


midterm_2_q8 = """
Question A1\n

A supermarket is analyzing its checkout lines, to determine how many checkout lines to have open at each time.

At busy times (about 10 percent of the times), the arrival rate is 5 shoppers/minute. At other times, the arrival rate is 2 shoppers/minute.
Once a shopper starts checking out (at any time), it takes an average of 3 minutes to complete the checkout.

[NOTE: This is a simplified version of the checkout system. If you have deeper knowledge of how supermarket checkout systems work,
please do not use it for this question; you would end up making the question more complex than it is designed to be.]
\n
a. The first model the supermarket tries is a queuing model with 20 lines open at all times. What would you expect the queuing model to show?
- Wait times are low at both busy and non-busy times.\n
- Wait times are low at busy times and high at non-busy times.\n
- Wait times are low at non-busy times and high at busy times.\n
- Wait times are high at both busy and non-busy times.\n
\n
b. The second model the supermarket tries is a queuing model with 10 lines open during busy times and 4 lines open during non-busy times. What would you expect the queuing model to show?
- Wait times are low at both busy and non-busy times.\n
- Wait times are low at busy times and high at non-busy times.\n
- Wait times are low at non-busy times and high at busy times.\n
- Wait times are high at both busy and non-busy times.\n
\n
The supermarket now has decided that, when there are 5 people waiting (across all lines), the supermarket will open an express checkout line,
which stays open until nobody is left waiting.

The supermarket would like to model this new process with a Markov chain, where each state is the number of people waiting
(e.g., 0 people waiting, 1 person waiting, etc.).

Notice that now, the transition probabilities from a state like "3 people waiting" depend on how many lines are currently open,
and therefore depend on whether the system was more recently in the state "5 people waiting" or "0 people waiting".

c. Which of the following statements about the process (the checkout system) and its relation to the Markov chain's memoryless property
(previous states don't affect the probability of moving from one state to another) is true?
- The process is memoryless, so the Markov chain is an appropriate model.\n
- The process is memoryless and the Markov chain is an appropriate model only if the arrivals follow the Poisson distribution and
  the checkout times follow the Exponential distribution.\n
- The process is not memoryless, so the Markov chain model would not be not well-defined.\n
"""



midterm_2_q9 = """
Question A1\n

A charity is testing two different mailings to see whether one generates more donations than another. The charity is using A/B testing:
For each person on the charity's mailing list, the charity randomly selects one mailing or the other to send. The results after 2000 trials are shown below.

Trials      Donation rate       95 percent confidence interval
Option A    1036                4.8 percent                3.6 percent - 6.2 percent
Option B    964                 10.4 percent               8.5 percent - 12.3 percent
Note: The "donation rate" is the fraction of recipients who donate. Higher donation rates are better.

a. What should the charity do?
- Switch to exploitation (utilize Option A only; A is clearly better)\n
- Switch to exploitation (utilize Option B only; B is clearly better)\n
- More exploration (test both options; it is unclear yet which is better)\n

Later, the charity developed 7 new options, so they used a multi-armed bandit approach where each option is chosen with probability
proportional to its likelihood of being the best. The results after 2000 total trials are shown below.

Donation rate       Mean donation       Median donation
Option #1           3.2%                $112               $100
Option #2           4.2%                $98                $75
Option #3           5.2%                $174               $125
Option #4           5.5%                $153               $100
Option #5           6.5%                $122               $80
Option #6           10.8%               $132               $100
Option #7           15.0%               $106               $75

b. If the charity's main goal is to find the option that has the highest median donation, which type of tests should they use to see if the
option that appears best is significantly better than each of the other options?
- Binomial-based (e.g., McNemar's) tests\n
- Other non-parametric tests\n
- Parametric tests\n
"""

midterm_2_q10 = """
Question A1\n
For each question,
select the most appropriate model/approach to answer the question/analyze the situation described.
Each model/approach might be used zero, one, or more than one time in the five questions.
CHOICES:\n
 - Non-parametric test\n
 - Louvain algorithm\n
 - Stochastic optimization\n
 - Game theoretic analysis\n
 - Queueing\n


Question a\n
Does Lasik surgery significantly improve the median vision of people who get that surgery?
\n
Question b\n
Which groups of genetic markers often appear together in people?
\n
Question c\n
What distinct sets of recipes can be identified where there are many ingredients shared within each set?
\n
Question d\n
Determine the best marketing strategy, given that a competitor will react to your choice in his/her decisions.
\n
Question e\n
Find sets of terrorists that have a lot of communication within each set.

"""



midterm_2_q11 =  """
A1\n
For each question, you should select the most appropriate model or approach from the provided list: Non-parametric test, Louvain algorithm, Stochastic optimization, Game theoretic analysis, and Queueing.

\nQuestion a: Does the average number of daily visitors to a website significantly differ between weekdays and weekends?
\nQuestion b: In a network of scientific collaborations, which group of researchers forms the most tightly-knit community?
\nQuestion c: How should a logistics company route its trucks to minimize fuel costs, considering the varying prices and traffic conditions?
\nQuestion d: Two competing coffee shops are deciding their pricing strategies. How should each shop set its prices to maximize profit, considering the possible reactions of the other?
\nQuestion e: A hospital needs to optimize its staff scheduling to reduce patient wait times, especially during peak hours and emergencies. What approach should it use?

"""

midterm_2_q12 =  """
Question A1\n
A large retail store wants to optimize its staffing schedule based on customer footfall to minimize staffing costs while ensuring customer satisfaction. The store manager proposes the following approach:

Proposal: GIVEN past sales data and customer footfall data, USE linear regression TO predict customer footfall for each hour of the day. Then, GIVEN the predicted footfall for each hour, USE a fixed ratio of staff to customers TO determine the number of staff needed each hour. Finally, GIVEN the hourly staffing requirements, USE a scheduling algorithm TO create an optimal weekly staffing schedule.

Select all statements that indicate why the manager's proposal might be flawed:

\nA. Linear regression may not accurately capture the complex patterns in customer footfall data.
\nB. A fixed ratio of staff to customers might not account for variations in staff efficiency or customer needs.
\nD. Customer footfall could be influenced by factors not included in past sales data, making predictions unreliable.
\nE. A scheduling algorithm does not consider staff availability or preferences, potentially leading to impractical schedules.

"""

midterm_2_q13 =  """
Question A1\n
A city's transportation department wants to improve traffic flow by optimizing the timings of traffic lights. Their proposed method is as follows:

Proposal: GIVEN past traffic volume data at intersections, USE a decision tree model TO predict traffic volume for different times of the day. Then, GIVEN the predicted traffic volumes, USE a simple cycle timing formula TO set traffic light durations. Finally, GIVEN the set durations, USE simulation TO test and adjust the timings.

Select all statements that show why the department's method might be inappropriate:

\nA. A decision tree might not capture the dynamic and continuous nature of traffic flow.
\nB. Traffic volume can be influenced by unpredictable factors (like weather or accidents) not accounted for in the model.
\nC. The simple cycle timing formula might not be sufficient for complex traffic patterns and could lead to inefficiencies.
\nD. Simulation testing may not accurately represent real-world conditions, leading to suboptimal traffic light timings.
"""

midterm_2_q14 =  """
Question A1\n
Put the following seven steps in order, from what is done first to what is done last.
\n-Impute missing data values
\n-Fit lasso regression model on all variables
\n-Pick model to use based on performance on a different data set
\n-Remove outliers
\n-Fit linear regression, regression tree, and random forest models using variables chosen by lasso regression
\n-Test model on another different set of data to estimate quality
\n-Scale data
"""

midterm_2_q15 =  """
Question A1\n
Directions: Match each analytical technique or concept (listed in items 1-9) with its primary application or characteristic (options A-K). Each option can be used once, more than once, or not at all.
\n
Items:

\nClassification using SVM and KNN
\nModel Validation
\nClustering with k-means
\nOutlier Detection in Data Preparation
\nCUSUM in Change Detection
\nARIMA Models in Time Series Analysis
\nLogistic Regression in Advanced Regression
\nBox-Cox Transformation
\nNeural Networks in Advanced Models
\nOptions:
\nA. Balancing early detection of changes and avoiding false positives
\nB. Grouping similar data for unsupervised learning applications
\nC. Improving model accuracy by avoiding reliance on training data alone
\nD. Normalizing data distributions in complex datasets
\nE. Predicting probabilities within a range of 0 to 1
\nF. Categorizing data into distinct groups based on similarity
\nG. Addressing issues in datasets with cyclic patterns and trends
\nH. Identifying and handling data points that significantly differ from others
\nI. Processing large datasets with the risk of overfitting
\nJ. Utilizing in marketing effectiveness and process optimization
\nK. Enhancing decision-making in competitive scenarios

"""
midterm_2_q16 =  """
Question A1\n
A manufacturing company wants to minimize production costs while meeting the demand for its products. The production involves various constraints like machine hours, labor availability, and material costs.

a. Select all of the models/approaches the company could use to minimize production costs while meeting demand:
\n1.	Linear Programming
\n2.	Elastic Net
\n3.	Integer Programming
\n4.	Support Vector Machine
\n5.	Convex Optimization
\nThe company also wants to forecast future demand for its products to better plan production schedules and raw material purchases.
b. Select all of the models/approaches the company could use to forecast future product demand:
\n1.	Time Series Analysis
\n2.	Lasso Regression
\n3.	Stochastic Optimization
\n4.	Logistic Regression
\n5.	Random Forest
"""

midterm_2_q17 =  """
Question A1\n
An energy company is planning to optimize the distribution of electricity across a network of cities to ensure efficient power delivery and minimize losses.
\na. Select all of the models/approaches the company could use to optimize electricity distribution:
\n1.	Network Optimization
\n2.	Dynamic Programming
\n3.	Integer Programming
\n4.	Clustering
\n5.	Ridge Regression

The company also aims to predict electricity consumption patterns to better match supply with demand, especially during peak hours.
\nb. Select all of the models/approaches the company could use to predict electricity consumption patterns:
\n1.	Time Series Analysis
\n2.	Support Vector Machine
\n3.	Elastic Net
\n4.	Stochastic Optimization
\n5.	Logistic Regression
"""

midterm_2_q18 =  """
Question A1\n
An energy company is planning to optimize the distribution of electricity across a network of cities to ensure efficient power delivery and minimize losses.
a. Select all of the models/approaches the company could use to optimize electricity distribution:
\n1.	Network Optimization
\n2.	Dynamic Programming
\n3.	Integer Programming
\n4.	Clustering
\n5.	Ridge Regression

The company also aims to predict electricity consumption patterns to better match supply with demand, especially during peak hours.
b. Select all of the models/approaches the company could use to predict electricity consumption patterns:
\n1.	Time Series Analysis
\n2.	Support Vector Machine
\n3.	Elastic Net
\n4.	Stochastic Optimization
\n5.	Logistic Regression
"""

midterm_2_q19 =  """
Question A1\n

Four predictive models were developed to forecast the likelihood of a new business succeeding in a specific area, based on local economic indicators, demographic data, and other factors. The training dataset was missing the demographic diversity score for certain newly developed areas (4% of the data points).

It is hypothesized that areas with rapid industrial growth are more likely to have missing demographic diversity scores.

\nModel A used mean substitution, filling in the missing data with the average demographic diversity score from the rest of the data.
\nModel B used a predictive imputation method, creating a linear regression model to estimate the missing demographic diversity scores based on other variables.
\nModel C used a two-step approach: initially, it employed a logistic regression model to predict (based on other variables) if a new industrial area is likely due to recent industrial growth or for other reasons (like technological advancements), and then used this prediction to apply one of two different linear regression models to estimate the demographic diversity score; one model for areas developed due to industrial growth and another for areas developed for other reasons.
\nModel D introduced a categorical variable to indicate areas with missing data, categorizing them as either "data available", "missing, industrial growth", or "missing, other reasons".
\n
a. If demographic diversity scores can be accurately estimated from other factors, and new industrial areas attributed to rapid industrial growth cannot be effectively classified using the other factors, which model would you recommend?

\nModel A
\nModel B
\nModel C
\nModel D
\n
b. In which of the following situations would you recommend using Model C? [All estimations and classifications below are based on the other factors.]

\nDiversity scores can be accurately estimated, and reasons for industrial area development can be effectively classified.
\nDiversity scores can be accurately estimated, and reasons for industrial area development cannot be effectively classified.
\nDiversity scores cannot be accurately estimated, and reasons for industrial area development can be effectively classified.
\nDiversity scores cannot be accurately estimated, and reasons for industrial area development cannot be effectively classified.
"""
midterm_2_q20 =  """
Question A1\n
A financial consulting company plans to use machine learning to predict stock market trends based on a range of economic indicators. They have historical data on stock prices, various economic indicators, and trading volumes. However, they lack reliable historical data on investor sentiment, which is believed to be a key influencing factor. They have current sentiment data but do not know historical sentiment values.
Analyzing the question:

Stock prices, economic indicators, and trading volumes are numerical values.
For questions lacking reliable historical data on a key factor (investor sentiment), time series models might not be directly applicable.
a. For each of the questions below, select a model/approach that the company could use.

i. What model/approach could the company use to predict future stock market trends based on economic indicators and trading volumes?
\nOptions: Neural Networks, Support Vector Regression, or ARIMA

ii. What model/approach could the company use to select the most influential economic indicators?
\nOptions: Feature Importance in Random Forest, Ridge Regression, or Principal Component Analysis (PCA)

iii. Suppose the company wants to investigate if there was a significant change in stock market behavior after a major economic policy change. What model/approach could they use to identify if and when a significant change occurred?
\nOptions: Change Point Detection methods, Bayesian Structural Time Series, or Interrupted Time Series Analysis
"""

midterm_2_q21 = """
Question A1\n
A health care research organization aims to predict patient readmission rates based on various clinical and demographic factors. They have access to a comprehensive dataset including patient age, gender, medical history, treatment details, and socio-economic factors. However, they lack detailed data on patient lifestyle choices, which could significantly impact readmission rates.
Analyzing the question:

Patient characteristics and treatment details are categorical and numerical.
The absence of detailed lifestyle data suggests a need for models that can handle incomplete information effectively.
a. For each of the questions below, select a model/approach that the organization could use. [NOTE: CHOICES WILL BE THINGS LIKE "K-Means Clustering", "Decision Trees", "Naive Bayes", etc.]
\n
i.  What model/approach could the organization use to predict patient readmission rates based on available clinical and demographic data?
\nOptions: Logistic Regression, Decision Trees, Random Forest, Support Vector Machines, Neural Networks, K-Nearest Neighbors, Gradient Boosting
\n
ii.  What model/approach could the organization use to group patients into similar categories based on their clinical and demographic data?
\nOptions: K-Means Clustering, Random Forest, Linear Program, Poisson
\n
iii.  If the organization wanted to understand which factors are most predictive of readmission, what model/approach could they use for feature selection and importance?
\nOptions: Lasso Regression, Ridge Regression, Elastic Net, Recursive Feature Elimination, Feature Importance in Ensemble Methods, Correlation Analysis
\n
iv.  In the scenario where the organization wants to evaluate the accuracy of their predictive model, which techniques could they use?
\nOptions: Cross-Validation, Confusion Matrix, ROC Curve, Precision-Recall Curve, Brier Score, F1 Score, Log-Loss

"""

midterm_2_q22 = """
Question A1\n
For the scenarios described in your questions, selecting the appropriate probability distribution involves understanding the nature of the event being modeled and the type of data:
\n
1. Duration a customer spends shopping in a store\n
Choose from: Exponential, Normal, Weibull, Poisson, Log-Normal\n

2. Number of emails received by an office worker in an hour\n
Choose from: Poisson, Binomial, Negative Binomial, Geometric, Bernoulli\n

3. Number of customers who return a product within 30 days out of the first 200 sales\n
Choose from: Binomial, Poisson, Hypergeometric, Negative Binomial, Geometric\n

4. Number of correct answers before the first wrong attempt in a multiple-choice quiz\n
Choose from: Geometric, Binomial, Negative Binomial, Poisson, Exponential\n

5. Time taken for a website to load after clicking a link\n
Choose from: Exponential, Normal, Gamma, Weibull, Bernouilli\n

"""

midterm_2_q23 = """
Question A1\n
For each of the following scenarios, select the most appropriate model/approach to use. Note that some models/approaches may be applicable to more than one scenario.

MODELS/APPROACHES\n
\ni. Markov Decision Processes
\nii. Logistic Regression
\niii. Neural Networks
\niv. Cusum
\nv. Decision Tree Analysis
\nvi. Dynamic Programming

SCENARIOS\n
\na. Choosing the best layout for a retail store to maximize customer flow and sales, based on customer movement patterns.
\nb. Predicting the likelihood of a patient developing a specific disease based on their lifestyle choices, age, and family history.
\nc. Determining the most efficient route for a delivery truck that has multiple stops in a city, considering traffic conditions.
\nd. Automating the process of categorizing customer reviews into positive, negative, or neutral sentiments.
\ne. Deciding the optimal investment strategy for a retirement fund, given the uncertainties in the financial market.
"""

midterm_2_q24 = """
Question A1\n
\nFor each of the scenarios presented, the choice between linear regression and logistic regression depends on the nature of the outcome variable (continuous or categorical/binary). Here are the answers based on the descriptions:
\n1. Which model is more suitable to predict the final grade of a student based on their attendance, assignment scores, and class participation?

\n2. Which model is more appropriate for predicting whether a bank's loan applicant is likely to default based on their credit history and income level?

\n3. Which model would be more directly suitable for estimating the average daily electricity consumption of a household based on the number of occupants and appliance usage?

\n4. Which model is best suited for determining the probability that a customer will subscribe to a new service based on demographic data and past subscription history?
"""

midterm_2_q25 = """
Question A1\n
\nA technology company is analyzing various aspects of its operations and market dynamics. They have collected extensive data over time and are looking to apply appropriate models to gain insights.

\nFor each of the following situations, specify which model is more appropriate: ARIMA, Louvain Algorithm, or Integer Linear Programming.

\nA. The company wants to forecast its quarterly sales for the next two years based on past sales data, which shows seasonal trends and some irregular fluctuations.

\nB. The company aims to understand the structure of its internal communication network to identify tightly-knit groups or communities within its workforce.

\nC. The company needs to optimize the allocation of its limited resources across different projects, ensuring maximum efficiency while adhering to budget and manpower constraints.

"""


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
\n - Discrete event simulation: A simulation that models a system that changes when specific events occur.
\n - Linear Regression:
\n - Logistic Regression Tree:
\n - Random Linear Regression Forest:
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
\n - A linear regression model
\n - A logistic regression model
\n - A k-nearest neighbors model

\nb.  For each of the following scenarios (i-iii), which model (1, 2, or 3) would you suggest using?
\ni. The supermarket wants to estimate the total number of customers visiting the store each day.
\nii. The supermarket aims to predict the likelihood of having more than 500 customers in the store each day.
\niii. The supermarket seeks to classify days into high or low customer traffic based on a threshold of 500 customers.

A regional supermarket chain has implemented a triple exponential smoothing (Holt-Winters) model to forecast the number of customers visiting the store each day. The model includes a multiplicative seasonal pattern with a weekly cycle (i.e., L=7).

Given that the supermarket experiences regular customer patterns with minimal random day-to-day variation, they are determining the optimal value for α (the level smoothing constant).

i. What should they expect the best value of α to be, considering the consistency in customer visits?

\n - α < 0
\n - 0 < α < ½
\n - ½ < α < 1
\n - α > 1
"""

midterm_1_q18 = """
Question A1\n
A regional healthcare provider has collected extensive data on patient visits over the years, including patient demographics, symptoms, diagnoses, treatments, and outcomes. The organization now wants to leverage this data to predict patient readmission risks and identify key factors that contribute to higher readmission rates. This insight will help them in allocating resources more effectively and improving patient care strategies. Choose the appropriate models/approaches from the list below that the healthcare provider could use for predicting patient readmissions and understanding the underlying factors.

\nCUSUM: A method for change detection that compares the mean of an observed distribution with a threshold level for change, abbreviated as cumulative sum.
\n - K-nearest-neighbor classification: A non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space.
\n - Logistic Regression: A statistical model that in its basic form uses a logistic function to model a binary dependent variable, often used for binary classification.
\n - Multi-armed bandit: A problem in which a fixed limited set of resources must be allocated between competing (alternative) choices in a way that maximizes their expected gain.
\n - Support Vector Machine: A supervised learning model with associated learning algorithms that analyze data used for classification and regression analysis.
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
\n - Sequence number in batch
\n - Day within batch that unit was completed
\n - Time until first reported efficacy drop

\nb. The seasonal length (L) in this analysis:
\n - 200 (number of units in a batch)
\n - 7 (days to complete a batch)

\nc. If the seasonal component (C_t) values are consistently lower towards the end of the batch, it suggests:
\n - Units produced later in a batch tend to show efficacy drop more quickly.
\n - Units produced later in a batch tend to maintain efficacy longer.
\n - This question is designed to assess the application of time series analysis in a production context and the interpretation of its results.

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
\n - Cusum
\n - K-Nearest Neighbors
\n - Support Vector Machines

A2.For clustering patients based on similarities in their diagnosis and treatment types, which algorithm would be most effective?
\n - K-Means Clustering
\n - PCA
\n - GARCH Variance Clustering
\n
Question b: Time Series Analysis for Predicting Treatment Efficacy

The team is also interested in predicting the efficacy of treatments over time.

A3. If the team wants to forecast treatment efficacy based on past trends and seasonal variations, which model should they use?
\n - ARIMA
\n - Exponential Smoothing
\n - Random Forests

A4.To detect significant changes in treatment efficacy over time, which method would be most suitable?
\n - CUSUM
\n - Principal Component Analysis
\n - Box-Cox Transformation
"""

midterm_1_q22 = """
Question A1: \nSupport Vector Machines (SVM) and K-Nearest Neighbor (KNN) in Classification

a. A bank is developing a model to classify loan applicants as high-risk or low-risk. Given the importance of minimizing the misclassification of high-risk applicants, which model would be more suitable?
\n - SVM
\n - KNN

b. In a medical diagnosis system, which model would be preferable for classifying patients based on a dataset with many overlapping characteristics?
\n - SVM
\n - KNN

Question A2: Validation and Model Assessment
\na. A marketing team has developed several predictive models for customer behavior. To avoid overfitting, which approach should they use for model assessment?
\n - Cross-validation
\n - Training on the entire dataset
\nb. When choosing between two different models for predicting sales, one with a lower AIC and one with a higher BIC, which model should be preferred considering both simplicity and likelihood?
\n - Model with lower AIC
\n - Model with higher BIC
"""

midterm_1_q23 = """
Question A1: \nClustering and Outlier Detection in Data Analysis

\na. A retailer wants to segment their customer base for targeted marketing. Which clustering method would be best for a dataset with well-defined, separate customer groups?
\n - K-means Clustering
\n - PCA

\nb. In analyzing customer purchase data, the team identifies several extreme values. What is the most appropriate initial step in handling these outliers?
\n - Removing them from the dataset
\n - Investigating their source and context

\nQuestion A2: Time Series Analysis and Exponential Smoothing
\na. A utility company is analyzing electricity usage patterns over time. To forecast future usage that exhibits both trend and seasonality, which method would be most appropriate?
\n - ARIMA
\n - Exponential Smoothing with trend and seasonality

\nb. If the company wants to smooth out short-term fluctuations in daily usage data while giving more weight to recent observations, what should be the approach to setting the alpha value in exponential smoothing?
\n - A high alpha value
\n - A low alpha value
"""

midterm_1_q24 = """
Question A1: \nClassification Techniques

\nA financial institution is implementing a new system to classify loan applicants based on risk.

\na. Which classifier would be more effective for categorizing applicants into 'high risk' and 'low risk', considering the cost of misclassification?
\n-Linear Regression
\n-K-Nearest Neighbor (KNN)
\n-Support Vector Machine (SVM)
\n-Random Forest

\nb. In a scenario where the bank needs to identify potential fraudulent transactions, which approach should they use, given the transactions data is highly imbalanced?
\n-Hard Classifiers
\n-Soft Classifiers
\n-Decision Trees
\n-Bayesian Classifiers

\nQuestion A2: Model Validation and Testing
An e-commerce company is evaluating different models for predicting customer purchase behavior.
\na. To ensure the chosen model is not overfitting, which method should be used for validating the model's effectiveness?
\n-Cross-Validation
\n-Training on Entire Dataset
\n-AIC/BIC Comparison
\n-Holdout Method

\nb. If the model performs well on the training data but poorly on the validation data, what might this indicate?
\n-The model is underfitting
\n-The model is overfitting
\n-The model is perfectly fitted
\n-The model is not complex enough
"""

midterm_1_q25 = """
Question A1: \nClustering in Market Segmentation

A marketing agency is segmenting its audience for targeted advertising campaigns.
\na. For creating customer segments based on shopping behavior and preferences, which clustering method would be most suitable?
\n-K-means Clustering
\n-KNN Clustering
\n-PCA
\n-Poisson Variance Classification

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



FINAL_QUESTIONS = [midterm_1_q1,midterm_1_q2,midterm_1_q3,midterm_1_q4,midterm_1_q5,midterm_1_q6,
                       midterm_1_q7,midterm_1_q8,midterm_1_q9,midterm_1_q10,midterm_1_q11,midterm_1_q12,
                       midterm_1_q13,midterm_1_q14,midterm_1_q15,midterm_1_q16,midterm_1_q17,midterm_1_q18,
                       midterm_1_q19,midterm_1_q20,midterm_1_q21,midterm_1_q22,midterm_1_q23,midterm_1_q24,
                       midterm_1_q25,midterm_2_q1,midterm_2_q2,midterm_2_q3,midterm_2_q4,midterm_2_q5,midterm_2_q6,
                       midterm_2_q7,midterm_2_q8,midterm_2_q9,midterm_2_q10,midterm_2_q11,midterm_2_q12,
                       midterm_2_q13,midterm_2_q14,midterm_2_q15,midterm_2_q16,midterm_2_q17,midterm_2_q18,
                       midterm_2_q19,midterm_2_q20,midterm_2_q21,midterm_2_q22,midterm_2_q23,midterm_2_q24,
                       midterm_2_q25]
