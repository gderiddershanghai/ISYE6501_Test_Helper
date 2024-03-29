
midterm_2_q1 =     """
Information for Question A:\n
There are five questions labeled "Question 1." Answer all five questions. For each of the following five questions,
select the probability distribution that could best be used to model the described scenario. Each distribution might
be used, zero, one, or more than one time in the five questions.
\n
Choose from Binomial, Poisson, Weibull, Exponential, Geometric, Bernoulli Distributions
\n
Question A1
Time between people entering a grocery store
\n
Question A2
Number of hits to a real estate web site each minute
\n
Question A3
Number of penalty kicks that are saved by the goalkeeper, out of the first 96 taken
\n
Question A4
Number of faces correctly identified by deep learning (DL) software until an error is made
\n
Question A5
Time between hits on a real estate web site
\n
"""

midterm_2_q2 ="""
Match the problem to the distribution:\n
Choose from Binomial, Poisson, Weibull, Exponential, Geometric, Bernoulli Distributions
\n
Question A1:
Lifetime of light bulbs produced in a factory.
\n
Question A2:
Number of customers arriving at a bank per hour.
\n
Question A3:
Number of correct answers in a multiple-choice exam with 50 questions, each with four options, where a student guesses all answers.
\n
Question A4:
Time until a newly launched website receives its first hundred visitors.
\n
Question A5:
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

Question a:
Minimize the sum of (log(c_i) * x_i), subject to the sum of (a_ij * x_i) greater than or equal to b_j for all j, and all x_i greater than or equal to 0.
\n
Question b:
Maximize the sum of (c_i * x_i), subject to the sum of (a_ij * x_i) greater than or equal to b_j for all j, and all x_i greater than or equal to 0.
\n
Question c:
Minimize the sum of (c_i * x_i^2), subject to the sum of (a_ij * x_i) greater than or equal to b_j for all j, and all x_i greater than or equal to 0.
\n
Question d:
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
CHOICES:
Non-parametric test\n
Louvain algorithm\n
Stochastic optimization\n
Game theoretic analysis\n
Queueing\n


Question a
Does Lasik surgery significantly improve the median vision of people who get that surgery?
\n
Question b
Which groups of genetic markers often appear together in people?
\n
Question c
What distinct sets of recipes can be identified where there are many ingredients shared within each set?
\n
Question d
Determine the best marketing strategy, given that a competitor will react to your choice in his/her decisions.
\n
Question e
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






MIDTERM_2_QUESTIONS = [midterm_2_q1,midterm_2_q2,midterm_2_q3,midterm_2_q4,midterm_2_q5,midterm_2_q6,
                       midterm_2_q7,midterm_2_q8,midterm_2_q9,midterm_2_q10,midterm_2_q11,midterm_2_q12,
                       midterm_2_q13,midterm_2_q14,midterm_2_q15,midterm_2_q16,midterm_2_q17,midterm_2_q18,
                       midterm_2_q19,midterm_2_q20,midterm_2_q21,midterm_2_q22,midterm_2_q23,midterm_2_q24,
                       midterm_2_q25]
