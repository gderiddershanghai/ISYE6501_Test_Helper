
midterm_2_q1 =     """
Information for Question 1:
There are five questions labeled "Question 1." Answer all five questions. For each of the following five questions,
select the probability distribution that could best be used to model the described scenario. Each distribution might
be used, zero, one, or more than one time in the five questions.

Choose from Binomial, Poisson, Weibull, Exponential, Geometric, Bernoulli Distributions

Question 1
Time between people entering a grocery store

Question 2
Number of hits to a real estate web site each minute

Question 3
Number of penalty kicks that are saved by the goalkeeper, out of the first 96 taken

Question 4
Number of faces correctly identified by deep learning (DL) software until an error is made

Question 5
Time between hits on a real estate web site

"""

midterm_2_q2 ="""
Match the problem to the distribution:
Choose from Binomial, Poisson, Weibull, Exponential, Geometric, Bernoulli Distributions

Question 1:
Lifetime of light bulbs produced in a factory.

Question 2:
Number of customers arriving at a bank per hour.

Question 3:
Number of correct answers in a multiple-choice exam with 50 questions, each with four options, where a student guesses all answers.

Question 4:
Time until a newly launched website receives its first hundred visitors.

Question 5:
The number of trials needed for a new machine to successfully complete its first task without error.

"""


midterm_2_q3 = """
Question 1: A retail company wants to optimize the layout of its warehouse to reduce the time it takes for workers to pick items for shipping. The layout optimization should consider the frequency of item requests and the physical distance between items.
a. Select all of the models/approaches the company could use to optimize the warehouse layout for efficient item picking, considering the frequency of item requests and physical distances:
1.	Logistic Regression
2.	Clustering
3.	Network Optimization
4.	Ridge Regression
5.	Integer Programming

Suppose the company also wants to forecast future item request patterns to further refine the warehouse layout. This forecast should be based on historical sales data, seasonal trends, and promotional activities.
b. Select all of the models/approaches the company could use to forecast future item request patterns:
1.	Time Series Analysis
2.	Support Vector Machine
3.	Lasso Regression
4.	Stochastic Optimization
5.	Elastic Net
"""

midterm_2_q4 = """
Question 2: A healthcare provider aims to optimize its staff scheduling to ensure adequate coverage across various departments, taking into account staff availability, skills, and shift preferences.
a. Select all of the models/approaches the provider could use for optimal staff scheduling, considering staff availability, skills, and shift preferences:
1.	Linear Programming
2.	Clustering
3.	Dynamic Programming
4.	Random Forest
5.	Integer Programming
Given the uncertainty in patient inflow and emergency cases, the healthcare provider also wants to prepare a robust staffing plan that can adapt to varying patient loads.
b. Select all of the models/approaches the provider could use to create a robust and adaptable staffing plan:
1.	Stochastic Optimization
2.	Logistic Regression
3.	Scenario Modeling
4.	Convex Optimization
5.	Elastic Net


"""

midterm_2_q5 = """
Questions 2a, 2b

Five classification models were built for predicting whether a neighborhood will soon see a large rise in home prices,
based on public elementary school ratings and other factors. The training data set was missing the school rating variable
for every new school (3% of the data points).

Because ratings are unavailable for newly-opened schools, it is believed that locations that have recently experienced
high population growth are more likely to have missing school rating data.

Model 1 used imputation, filling in the missing data with the average school rating from the rest of the data.
Model 2 used imputation, building a regression model to fill in the missing school rating data based on other variables.
Model 3 used imputation, first building a classification model to estimate (based on other variables) whether a new
school is likely to have been built as a result of recent population growth (or whether it has been built for another
purpose, e.g. to replace a very old school), and then using that classification to select one of two regression models
to fill in an estimate of the school rating; there are two different regression models (based on other variables),
one for neighborhoods with new schools built due to population growth, and one for neighborhoods with new schools built
for other reasons.
Model 4 used a binary variable to identify locations with missing information.
Model 5 used a categorical variable: first, a classification model was used to estimate whether a new school is likely
to have been built as a result of recent population growth; and then each neighborhood was categorized as "data available",
"missing, population growth", or "missing, other reason".

2a. If school ratings can be reasonably well-predicted from the other factors, and new schools built due to recent
population growth cannot be reasonably well-classified using the other factors, which model would you recommend?
- Model 1
- Model 2
- Model 3
- Model 4
- Model 5

2b. In which of the following situations would you recommend using Model 3? [All predictions and classifications below
are using the other factors.]
- Ratings can be well-predicted, and reasons for building schools can be well-classified.
- Ratings can be well-predicted, and reasons for building schools cannot be well-classified.
- Ratings cannot be well-predicted, and reasons for building schools can be well-classified.
- Ratings cannot be well-predicted, and reasons for building schools cannot be well-classified.
"""

midterm_2_q6 = """
Question 4a


A hospital emergency department (ED) has created a stochastic discrete-event simulation model of the ED,
including patient arrivals, resource usage (rooms, doctors, etc.), and treatment duration.

EDs are not first-come-first-served; a patient who arrives with a more-serious condition will be treated
first, ahead of even long-waiting patients with less-serious conditions.

When a patient comes in, the ED will run the simulation to quickly give the patient an estimate of the
expected wait time before being treated.

How many times does the ED need to run the simulation for each new patient (i.e., how many replications
are needed)?
- Once, because the outcome will be the same each time.
- Many times, because of the variability and randomness.
- Once, because each patient is unique.

Question 4b

Suppose it is discovered that simulated wait times in the hospital emergency department are 50 percent higher than actual wait times, on average. What would you recommend that they do?
- Scale down all estimates by a factor of 1/1.50 to get the average simulation estimates to match the average actual wait times.
- Investigate to see what's wrong with the simulation, because it's a poor match to reality.
- Use the 50percent-higher estimates, because that's what the simulation output is.
"""


midterm_2_q7 = """
Information for Question 5:
For each of the optimization problems below, select its most precise classification. In each model, x are the variables,
all other letters (a, b, c) refer to known data, and the values of c are all positive.

There are seven questions labeled "Question 5". Answer all seven questions. Each classification might be used, zero, one,
or more than one time in the seven questions.

Choices:
- Linear Program
- Convex Program
- Convex Quadratic Program
- General Nonconvex Program
- Integer Program

Question 5:
Minimize the sum of (log(c_i) * x_i), subject to the sum of (a_ij * x_i) greater than or equal to b_j for all j, and all x_i greater than or equal to 0.

Question 5:
Maximize the sum of (c_i * x_i), subject to the sum of (a_ij * x_i) greater than or equal to b_j for all j, and all x_i greater than or equal to 0.

Question 5:
Minimize the sum of (c_i * x_i^2), subject to the sum of (a_ij * x_i) greater than or equal to b_j for all j, and all x_i greater than or equal to 0.

Question 5:
Maximize the sum of (c_i * x_i), subject to the sum of (a_ij * x_i) greater than or equal to b_j for all j, and all x_i belonging to {0, 1}.
"""


midterm_2_q8 = """
Questions 6a, 6b, 6c

A supermarket is analyzing its checkout lines, to determine how many checkout lines to have open at each time.

At busy times (about 10 percent of the times), the arrival rate is 5 shoppers/minute. At other times, the arrival rate is 2 shoppers/minute.
Once a shopper starts checking out (at any time), it takes an average of 3 minutes to complete the checkout.

[NOTE: This is a simplified version of the checkout system. If you have deeper knowledge of how supermarket checkout systems work,
please do not use it for this question; you would end up making the question more complex than it is designed to be.]

6a. The first model the supermarket tries is a queuing model with 20 lines open at all times. What would you expect the queuing model to show?
- Wait times are low at both busy and non-busy times.
- Wait times are low at busy times and high at non-busy times.
- Wait times are low at non-busy times and high at busy times.
- Wait times are high at both busy and non-busy times.

6b. The second model the supermarket tries is a queuing model with 10 lines open during busy times and 4 lines open during non-busy times. What would you expect the queuing model to show?
- Wait times are low at both busy and non-busy times.
- Wait times are low at busy times and high at non-busy times.
- Wait times are low at non-busy times and high at busy times.
- Wait times are high at both busy and non-busy times.

The supermarket now has decided that, when there are 5 people waiting (across all lines), the supermarket will open an express checkout line,
which stays open until nobody is left waiting.

The supermarket would like to model this new process with a Markov chain, where each state is the number of people waiting
(e.g., 0 people waiting, 1 person waiting, etc.).

Notice that now, the transition probabilities from a state like "3 people waiting" depend on how many lines are currently open,
and therefore depend on whether the system was more recently in the state "5 people waiting" or "0 people waiting".

6c. Which of the following statements about the process (the checkout system) and its relation to the Markov chain's memoryless property
(previous states don't affect the probability of moving from one state to another) is true?
- The process is memoryless, so the Markov chain is an appropriate model.
- The process is memoryless and the Markov chain is an appropriate model only if the arrivals follow the Poisson distribution and
  the checkout times follow the Exponential distribution.
- The process is not memoryless, so the Markov chain model would not be not well-defined.
"""

midterm_2_q9 = """
Questions 7a, 7b

A charity is testing two different mailings to see whether one generates more donations than another. The charity is using A/B testing:
For each person on the charity's mailing list, the charity randomly selects one mailing or the other to send. The results after 2000 trials are shown below.

Trials      Donation rate       95% confidence interval
Option A    1036                4.8%                3.6%-6.2%
Option B    964                 10.4%               8.5%-12.3%
Note: The "donation rate" is the fraction of recipients who donate. Higher donation rates are better.

7a. What should the charity do?
- Switch to exploitation (utilize Option A only; A is clearly better)
- Switch to exploitation (utilize Option B only; B is clearly better)
- More exploration (test both options; it is unclear yet which is better)

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

7b. If the charity's main goal is to find the option that has the highest median donation, which type of tests should they use to see if the
option that appears best is significantly better than each of the other options?
- Binomial-based (e.g., McNemar's) tests
- Other non-parametric tests
- Parametric tests
"""

midterm_2_q10 = """
There are five questions labeled "Question 9".  Answer all five questions.  For each question,
select the most appropriate model/approach to answer the question/analyze the situation described.
Each model/approach might be used zero, one, or more than one time in the five questions.
CHOICES:
Non-parametric test
Louvain algorithm
Stochastic optimization
Game theoretic analysis
Queueing


Question 9
Does Lasik surgery significantly improve the median vision of people who get that surgery?

Question 9
Which groups of genetic markers often appear together in people?

Question 9
What distinct sets of recipes can be identified where there are many ingredients shared within each set?

Question 9
Determine the best marketing strategy, given that a competitor will react to your choice in his/her decisions.

Question 9
Find sets of terrorists that have a lot of communication within each set.

"""



midterm_2_q11 =  """
For each question, you should select the most appropriate model or approach from the provided list: Non-parametric test, Louvain algorithm, Stochastic optimization, Game theoretic analysis, and Queueing.

Question 9A: Does the average number of daily visitors to a website significantly differ between weekdays and weekends?
Question 9B: In a network of scientific collaborations, which group of researchers forms the most tightly-knit community?
Question 9C: How should a logistics company route its trucks to minimize fuel costs, considering the varying prices and traffic conditions?
Question 9D: Two competing coffee shops are deciding their pricing strategies. How should each shop set its prices to maximize profit, considering the possible reactions of the other?
Question 9E: A hospital needs to optimize its staff scheduling to reduce patient wait times, especially during peak hours and emergencies. What approach should it use?

"""

midterm_2_q12 =  """
Question 1
A large retail store wants to optimize its staffing schedule based on customer footfall to minimize staffing costs while ensuring customer satisfaction. The store manager proposes the following approach:

Proposal: GIVEN past sales data and customer footfall data, USE linear regression TO predict customer footfall for each hour of the day. Then, GIVEN the predicted footfall for each hour, USE a fixed ratio of staff to customers TO determine the number of staff needed each hour. Finally, GIVEN the hourly staffing requirements, USE a scheduling algorithm TO create an optimal weekly staffing schedule.

Select all statements that indicate why the manager's proposal might be flawed:

Linear regression may not accurately capture the complex patterns in customer footfall data.
A fixed ratio of staff to customers might not account for variations in staff efficiency or customer needs.
Customer footfall could be influenced by factors not included in past sales data, making predictions unreliable.
A scheduling algorithm does not consider staff availability or preferences, potentially leading to impractical schedules.

"""

midterm_2_q13 =  """
Question 2
A city's transportation department wants to improve traffic flow by optimizing the timings of traffic lights. Their proposed method is as follows:

Proposal: GIVEN past traffic volume data at intersections, USE a decision tree model TO predict traffic volume for different times of the day. Then, GIVEN the predicted traffic volumes, USE a simple cycle timing formula TO set traffic light durations. Finally, GIVEN the set durations, USE simulation TO test and adjust the timings.

Select all statements that show why the department's method might be inappropriate:

A decision tree might not capture the dynamic and continuous nature of traffic flow.
Traffic volume can be influenced by unpredictable factors (like weather or accidents) not accounted for in the model.
The simple cycle timing formula might not be sufficient for complex traffic patterns and could lead to inefficiencies.
Simulation testing may not accurately represent real-world conditions, leading to suboptimal traffic light timings.
"""

midterm_2_q14 =  """
Put the following seven steps in order, from what is done first to what is done last.
-Impute missing data values
-Fit lasso regression model on all variables
-Pick model to use based on performance on a different data set
-Remove outliers
-Fit linear regression, regression tree, and random forest models using variables chosen by lasso regression
-Test model on another different set of data to estimate quality
-Scale data
"""

midterm_2_q15 =  """
Directions: Match each analytical technique or concept (listed in items 1-9) with its primary application or characteristic (options A-K). Each option can be used once, more than once, or not at all.

Items:

Classification using SVM and KNN
Model Validation
Clustering with k-means
Outlier Detection in Data Preparation
CUSUM in Change Detection
ARIMA Models in Time Series Analysis
Logistic Regression in Advanced Regression
Box-Cox Transformation
Neural Networks in Advanced Models
Options:
A. Balancing early detection of changes and avoiding false positives
B. Grouping similar data for unsupervised learning applications
C. Improving model accuracy by avoiding reliance on training data alone
D. Normalizing data distributions in complex datasets
E. Predicting probabilities within a range of 0 to 1
F. Categorizing data into distinct groups based on similarity
G. Addressing issues in datasets with cyclic patterns and trends
H. Identifying and handling data points that significantly differ from others
I. Processing large datasets with the risk of overfitting
J. Utilizing in marketing effectiveness and process optimization
K. Enhancing decision-making in competitive scenarios

"""


MIDTERM_2_QUESTIONS = [midterm_2_q1,midterm_2_q2,midterm_2_q3,midterm_2_q4,midterm_2_q5,midterm_2_q6,
                       midterm_2_q7,midterm_2_q8,midterm_2_q9,midterm_2_q10,midterm_2_q11,midterm_2_q12,
                       midterm_2_q13,midterm_2_q14,midterm_2_q15]
