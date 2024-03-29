kc11_question_11_1 = {
    'question': "Building simpler models with fewer factors helps avoid which problems?",
    'options_list': [
        'Overfitting',
        'Low prediction quality',
        'Bias in the most important factors',
        'Difficulty of interpretation'
    ],
    'correct_answer': 'Overfitting',
    'explanation': "Simpler models are less likely to fit the noise in the training data, which helps in avoiding overfitting. Overfitting occurs when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data.",
    'chapter_information': 'Week 11: Variable Selection / Introduction to Variable Selection'
}

kc11_question_11_2 = {
    'question': "Which of these is a key difference between stepwise regression and lasso regression?",
    'options_list': [
        'Lasso regression requires the data to first be scaled',
        'Stepwise regression gives many models to choose from, while lasso gives just one'
    ],
    'correct_answer': 'Lasso regression requires the data to first be scaled',
    'explanation': "Lasso regression penalizes the absolute size of the coefficients, which means that without scaling, variables can be penalized more or less simply because of their scale. This is why data scaling is important in lasso regression to ensure that the penalty is applied uniformly across coefficients.",
    'chapter_information': 'Week 11: Variable Selection / Models for Variable Selection'
}

kc11_question_11_3 = {
    'question': "When two predictors are highly correlated, which of the following statements is true?",
    'options_list': [
        'Lasso regression will usually have positive coefficients for both predictors',
        'Ridge regression will usually have positive coefficients for both predictors'
    ],
    'correct_answer': 'Ridge regression will usually have positive coefficients for both predictors',
    'explanation': "Ridge regression tends to shrink the coefficients of correlated predictors towards each other, which means they'll both have small but non-zero coefficients. This is a result of ridge regression's L2 penalty, which penalizes the square of coefficients and tends to keep all variables in the model with their effects shrunk.",
    'chapter_information': 'Week 11: Variable Selection / Choosing a Variable Selection Model'
}

kc2_question_12_1 = {
    'question': "If we’re testing to see whether red cars sell for higher prices than blue cars, we need to account for the type and age of the cars in our data set. This is called",
    'options_list': [
        'Controlling',
        'Comparing',
        'Combining'
    ],
    'correct_answer': 'Controlling',
    'explanation': "We need to control for the effects of type and age to isolate the impact of the car's color on its selling price.",
    'chapter_information': 'Week 12: Introduction to Design of Experiments / Introduction to Design of Experiments'
}

kc2_question_12_2 = {
    'question': "In which of these scenarios would A/B testing not be a good model to use?",
    'options_list': [
        'Data can be collected quickly',
        'The collected data is not representative of the population we want an answer about'
    ],
    'correct_answer': 'The collected data is not representative of the population we want an answer about',
    'explanation': "A/B testing requires the data to be representative of the population to provide reliable insights. If it's not, the results won't be applicable to the wider population.",
    'chapter_information': 'Week 12: Introduction to Design of Experiments / A/B Testing'
}

kc2_question_12_3 = {
    'question': "In which of these situations is a factorial design more appropriate than A/B testing?",
    'options_list': [
        'Choosing between two different banner ad designs for orange juice',
        'Picking the best-tasting of four different brands of orange juice',
        'Finding the best combination of factors in orange juice to maximize sales'
    ],
    'correct_answer': 'Finding the best combination of factors in orange juice to maximize sales',
    'explanation': "Factorial designs are suited for experiments where the aim is to understand the effect of various factors in combination, rather than just comparing individual options.",
    'chapter_information': 'Week 12: Introduction to Design of Experiments / Factorial Designs'
}

kc2_question_12_4 = {
    'question': "Which of these is a way that multi-armed bandit models deal with balancing exploration and exploitation?",
    'options_list': [
        'As we get more sure of the best answer, we’re more likely to choose to use it',
        'If we’re unsure of the best answer, we should pick one and stick with it',
        'As we get more sure of the best answer, we’re more likely to try many different ones just to make sure'
    ],
    'correct_answer': 'As we get more sure of the best answer, we’re more likely to choose to use it',
    'explanation': "Multi-armed bandit models increasingly favor the best-performing option (exploitation) as confidence in its effectiveness grows, while still allowing for exploration of other options to a lesser extent.",
    'chapter_information': 'Week 12: Introduction to Design of Experiments / Multi-Armed Bandits'
}

kc2_question_13_2_1 = {
    'question': "Why is a binomial distribution not a good model for estimating the number of days each month that the temperature is above 50 degrees Fahrenheit?",
    'options_list': [
        'The binomial distribution models the number of “yes” answers out of some number of observations',
        'The results aren’t independent – days above 50 degrees are more likely to be clumped together in the summer'
    ],
    'correct_answer': 'The results aren’t independent – days above 50 degrees are more likely to be clumped together in the summer',
    'explanation': "The binomial distribution assumes each trial is independent and the probability of success (temperature above 50 degrees) remains constant, which is not the case here as the probability changes with seasons.",
    'chapter_information': 'Week 13: Probability-Based Models / Bernoulli, Binomial and Geometric Distributions'
}

kc2_question_13_2_2 = {
    'question': "According to the geometric distribution, what is the probability of having 5 successful sales calls before the first unsuccessful call, if p is the probability of a successful call?",
    'options_list': [
        'The probability is p^5',
        'The probability is (1-p) * p^5'
    ],
    'correct_answer': '(1-p) * p^5',
    'explanation': "The geometric distribution models the probability of observing the first failure after k successes. Hence, the probability of 5 successes followed by one failure is calculated as (1-p) * p^5.",
    'chapter_information': 'Week 13: Probability-Based Models / Bernoulli, Binomial and Geometric Distributions'
}

kc2_question_13_3_1 = {
    'question': "If the time between customer arrivals to a restaurant at lunchtime fits the exponential distribution, then which of the following is true?",
    'options_list': [
        'The number of arrivals per unit time follows the Weibull distribution',
        'The number of arrivals per unit time follows the Poisson distribution'
    ],
    'correct_answer': 'The number of arrivals per unit time follows the Poisson distribution',
    'explanation': "If the interarrival times are exponentially distributed, then the count of arrivals in any given time period follows the Poisson distribution.",
    'chapter_information': 'Week 13: Probability-Based Models / Poisson, Exponential and Weibull Distributions'
}

kc2_question_13_3_2 = {
    'question': "What is the difference between the interpretations of the geometric and Weibull distributions?",
    'options_list': [
        'The geometric distribution models how many tries it takes for something to happen, while the Weibull distribution models how long it takes',
        'The Weibull distribution models how many tries it takes for something to happen, while the geometric distribution models how long it takes'
    ],
    'correct_answer': 'The geometric distribution models how many tries it takes for something to happen, while the Weibull distribution models how long it takes',
    'explanation': "The geometric distribution is discrete and models the number of trials until the first success. The Weibull distribution is continuous and can model time until an event occurs.",
    'chapter_information': 'Week 13: Probability-Based Models / Poisson, Exponential and Weibull Distributions'
}

kc2_question_13_5 = {
    'question': "Which of these is a queuing model not appropriate for?",
    'options_list': [
        'Determining the average wait time on a customer service hotline',
        'Estimating the length of the checkout lines at a grocery store',
        'Predicting the number of customers who will come to a restaurant tomorrow'
    ],
    'correct_answer': 'Predicting the number of customers who will come to a restaurant tomorrow',
    'explanation': "Queuing models are designed to handle scenarios involving waiting lines and service processes, not predicting total arrivals without a context of waiting or queue formation.",
    'chapter_information': 'Week 13: Probability-Based Models / Queuing'
}

kc2_question_13_6_1 = {
    'question': "Why should a stochastic simulation be run many times?",
    'options_list': [
        'To verify that the same thing happens each time',
        'One random outcome might not be representative of system performance in the range of different situations that could arise'
    ],
    'correct_answer': 'One random outcome might not be representative of system performance in the range of different situations that could arise',
    'explanation': "Running a stochastic simulation multiple times allows for the assessment of system performance across a variety of possible scenarios, reflecting the inherent randomness and variability.",
    'chapter_information': 'Week 13: Probability-Based Models / Simulation Basics'
}

kc2_question_13_6_2 = {
    'question': "Why is it important to validate a simulation by comparing it to real data as much as possible?",
    'options_list': [
        'If the simulation isn’t a good reflection of reality, then any insights we gain from studying the simulation might not be applicable in reality'
    ],
    'correct_answer': 'If the simulation isn’t a good reflection of reality, then any insights we gain from studying the simulation might not be applicable in reality',
    'explanation': "Validation ensures that the simulation accurately represents real-world conditions, which is crucial for the reliability and applicability of the insights derived from the simulation.",
    'chapter_information': 'Week 13: Probability-Based Models / Simulation Basics'
}

kc2_question_13_7 = {
    'question': "How can simulation be used in a prescriptive analytics way, for example to determine the right number of voting machines and poll workers at an election location?",
    'options_list': [
        'Vary parameters of interest, compare the simulated system performance, and select the setup with the best results',
        'Use the automated optimization function in simulation software',
        'Both of the answers above'
    ],
    'correct_answer': 'Both of the answers above',
    'explanation': "Both manual variation of parameters and the use of automated optimization functions in simulation software are valid approaches to finding the optimal configuration for a system.",
    'chapter_information': 'Week 13: Probability-Based Models / Prescriptive Simulation'
}

kc2_question_13_8 = {
    'question': "In analytics, the term “memoryless” means",
    'options_list': [
        'The next state of a process doesn’t depend on its current state or any of its previous states',
        'The next state of a process doesn’t depend on any of its previous states, but does depend on the current state',
        'I don’t remember. Does that mean I’m memoryless?'
    ],
    'correct_answer': 'The next state of a process doesn’t depend on any of its previous states, but does depend on the current state',
    'explanation': "A process is considered 'memoryless' if its next state depends solely on the current state and not on the sequence of events that preceded it.",
    'chapter_information': 'Week 13: Probability-Based Models / Markov Chains'
}

kc2_question_14_1 = {
    'question': "Which of these is a common reason that data sets are missing values?",
    'options_list': [
        'A person accidentally typed in the wrong value',
        'A person did not want to reveal the true value',
        'An automated system did not work correctly to record the value',
        'All of the above'
    ],
    'correct_answer': 'All of the above',
    'explanation': "Data can be missing for various reasons, including human error, reluctance to provide information, and technical failures.",
    'chapter_information': 'Week 14: Missing Data / Introduction to Missing Data'
}

kc2_question_14_2 = {
    'question': "Which of the following statements is not a situation where missing data can be biased?",
    'options_list': [
        'Due to a programming flaw, a security camera discards every 7th frame of video it takes',
        'There’s no pattern to what security-related data is kept or discarded',
        'People might be less willing to express certain political views to survey-takers, or less willing to report certain incomes',
        'GPS devices might be more likely to lose service in some areas than in others'
    ],
    'correct_answer': 'There’s no pattern to what security-related data is kept or discarded',
    'explanation': "A lack of pattern in missing data suggests randomness, whereas bias in missing data occurs when the likelihood of data being missing is related to its value or other variables.",
    'chapter_information': 'Week 14: Missing Data / Methods That Do Not Require Imputation'
}

kc2_question_14_3 = {
    'question': "Which statements are true about data imputation?",
    'options_list': [
        "Imputing more than 5% of values is usually not recommended",
        'Imputation of more than one factor value for a data point is also possible',
        'Both answers above are correct'
    ],
    'correct_answer': 'Both answers above are correct',
    'explanation': "While it's typically advised not to impute more than 5% of data, modern methods allow for imputing multiple values, expanding the scope and flexibility of data recovery.",
    'chapter_information': 'Week 14: Missing Data / Imputation Methods'
}

kc15_question_15_1 = {
    'question': "Which of the following is true about the capabilities of statistical and optimization software?",
    'options_list': [
        'Statistical software can both build and solve regression models. Optimization software only solves models; human experts are required to build optimization models.',
        'Optimization software can both build and solve models. Regression software only solves models; human experts are required to build regression models.'
    ],
    'correct_answer': 'Statistical software can both build and solve regression models. Optimization software only solves models; human experts are required to build optimization models.',
    'explanation': "While statistical software often includes automated methods for model construction and solution, optimization software typically requires a human expert to formulate the model before it can solve it.",
    'chapter_information': 'Chapter 15: Optimization / Introduction to Optimization'
}

kc15_question_15_5 = {
    'question': "Which of these statistical models does not have an underlying optimization model to find the best fit?",
    'options_list': [
        'Linear regression',
        'Logistic regression',
        'Lasso regression',
        'Exponential smoothing',
        'k-means clustering',
        'None of the above'
    ],
    'correct_answer': 'None of the above',
    'explanation': "All listed statistical models use optimization techniques to estimate their parameters and find the best fit to the data.",
    'chapter_information': 'Chapter 15: Optimization / Optimization for Statistical Models'
}

kc15_question_15_6 = {
    'question': "True or false: Requiring some variables in a linear program to take integer values can make it take a lot longer to solve.",
    'options_list': [
        'True',
        'False'
    ],
    'correct_answer': 'True',
    'explanation': "Integer variables turn a linear program into an integer program, which can be significantly more complex and time-consuming to solve due to the combinatorial nature of the solution space.",
    'chapter_information': 'Chapter 15: Optimization / Classification of Optimization Models'
}

kc15_question_15_7 = {
    'question': "True or false: Optimization models implicitly assume that we know all of the values of the input data exactly.",
    'options_list': [
        'True',
        'False'
    ],
    'correct_answer': 'True',
    'explanation': "Traditional optimization models operate under the assumption that all input data is known and precise, without accounting for uncertainty or variability in the data.",
    'chapter_information': 'Chapter 15: Optimization / Stochastic Optimization'
}

kc15_question_15_8 = {
    'question': "What are the two main steps of most optimization algorithms?",
    'options_list': [
        'Find the most important remaining variable, and assign its value to be as large as possible.',
        'Find a good direction to move from the current solution, and determine how far to go in that direction.',
        'Pick a set of variables to be zero, and find the best values of the remaining variables.'
    ],
    'correct_answer': 'Find a good direction to move from the current solution, and determine how far to go in that direction.',
    'explanation': "Most optimization algorithms iterate by identifying an improving direction from the current point and then deciding on a step size to update the solution, gradually moving towards an optimum.",
    'chapter_information': 'Chapter 15: Optimization / Basic Optimization Algorithms'
}

kc16_question_16_1 = {
    'question': "Nonparametric tests are useful when…",
    'options_list': [
        "we don’t know much about the form of the underlying distribution the data comes from, or it doesn’t fit a nice distribution",
        "it’s important to have information about the median",
        "we don’t have much data",
        "we want to know whether the means of two distributions are similar"
    ],
    'correct_answer': "we don’t know much about the form of the underlying distribution the data comes from, or it doesn’t fit a nice distribution",
    'explanation': "Nonparametric tests don't make assumptions about the data's distribution and are useful when the data’s underlying distribution is unknown or doesn't fit well-known statistical distributions.",
    'chapter_information': 'Chapter 16: Advanced Models / Non-Parametric Methods'
}

kc16_question_16_2 = {
    'question': "How can Bayesian models incorporate expert opinion when there’s not as much data to analyze as we’d like to have?",
    'options_list': [
        "In Bayes’ theorem, both A and B are parameters that can be adjusted by an expert",
        "Expert opinion can be used to define the initial distribution of P(A), and observed data about B can be used with Bayes’ theorem to obtain a revised opinion P(A|B)"
    ],
    'correct_answer': "Expert opinion can be used to define the initial distribution of P(A), and observed data about B can be used with Bayes’ theorem to obtain a revised opinion P(A|B)",
    'explanation': "In Bayesian analysis, expert opinion can be formulated as a prior distribution, which is then updated with observed data to form the posterior distribution.",
    'chapter_information': 'Chapter 16: Advanced Models / Bayesian Modeling'
}

kc16_question_16_3 = {
    'question': "(Advanced question) Suppose we have a graph where all edge weights are equal to 1. How might you split the nodes into large groups that have very few connections between them?",
    'options_list': [
        "It’s not possible using what we’ve covered in the video",
        "Change the Louvain algorithm to minimize modularity instead of maximizing it",
        "Change the graph: for every pair of nodes i and j, if there’s an edge between i and j then remove it; and if there’s not an edge between i and j, then add it. Then run the Louvain algorithm on the new graph"
    ],
    'correct_answer': "Change the graph: for every pair of nodes i and j, if there’s an edge between i and j then remove it; and if there’s not an edge between i and j, then add it. Then run the Louvain algorithm on the new graph",
    'explanation': "By inversing the connectivity of the graph and then applying the Louvain algorithm, we can identify groups with minimal connections between them in the original graph, forming independent sets.",
    'chapter_information': 'Chapter 16: Advanced Models / Communities in Graphs'
}

kc16_question_16_4 = {
    'question': "Deep learning is one of the current best approaches for which of these?",
    'options_list': [
        "Image recognition",
        "Splitting graphs into highly-connected communities",
        "Demand forecasting"
    ],
    'correct_answer': "Image recognition",
    'explanation': "Deep learning algorithms, especially convolutional neural networks, are among the most effective methods currently available for image recognition tasks.",
    'chapter_information': 'Chapter 16: Advanced Models / Neural Networks and Deep Learning'
}

kc16_question_16_5 = {
    'question': "Which of the following is a situation where competitive decision-making (e.g., a game theoretic model) would be appropriate?",
    'options_list': [
        "A company wants to optimize its production levels, based on production cost, price, and demand. The company already has estimated a function to give predicted selling price and demand as a function of the number of units produced",
        "A company wants to optimize its production levels, based on production cost, price, and demand. The company already has estimated a function to give predicted selling price and demand as a function of the number of units produced, and the number of units its competitor produces"
    ],
    'correct_answer': "A company wants to optimize its production levels, based on production cost, price, and demand. The company already has estimated a function to give predicted selling price and demand as a function of the number of units produced, and the number of units its competitor produces",
    'explanation': "When the outcomes depend not only on a company's decisions but also on the decisions of competitors, game theory provides a framework for identifying optimal strategies in competitive environments.",
    'chapter_information': 'Chapter 16: Advanced Models / Competitive Models'
}


############# GPT GENERATED####################


kgpt_question_15_1 = {
    'question': "The rare book return slot will only open if a rare book is being returned. Select the mathematical constraint that corresponds to this English sentence.",
    'options_list': [
        'x_rare_slot <= x_rare_book',
        'x_rare_book = 1 - x_rare_slot',
        'x_regular_slot >= M * x_rare_slot',
        'x_rare_book + x_regular_book <= 1'
    ],
    'correct_answer': 'x_rare_slot <= x_rare_book',
    'explanation': "This constraint ensures that the rare book return slot cannot open unless a rare book is being returned, maintaining proper usage of the return slots.",
    'chapter_information': 'Chapter 15: Optimization'
}

kgpt_question_15_2 = {
    'question': "The repeat customer discount is valid only if the first-time customer discount has not been used. Select the mathematical constraint that corresponds to this English sentence.",
    'options_list': [
        'y_repeat <= y_first_time',
        'y_first_time + y_repeat <= 1',
        'y_repeat >= M * y_first_time',
        'y_first_time = 1 - y_repeat'
    ],
    'correct_answer': 'y_first_time + y_repeat <= 1',
    'explanation': "This constraint prevents a customer from using both first-time and repeat customer discounts simultaneously, ensuring the exclusive benefit of each discount.",
    'chapter_information': 'Chapter 15: Optimization'
}

kgpt_question_15_3 = {
    'question': "The fast pass for Thunderbolt is only available to guests with a premium pass. Select the mathematical constraint that corresponds to this English sentence.",
    'options_list': [
        'x_thunderbolt_fast <= x_premium_pass',
        'x_premium_pass = 1 - x_thunderbolt_fast',
        'x_thunderbolt_fast >= M * x_premium_pass',
        'x_thunderbolt_fast + x_standard_pass <= 1'
    ],
    'correct_answer': 'x_thunderbolt_fast <= x_premium_pass',
    'explanation': "By setting this constraint, the theme park ensures that only guests with a premium pass can access the fast pass for the Thunderbolt ride.",
    'chapter_information': 'Chapter 15: Optimization'
}

kgpt_question_15_4 = {
    'question': "Subscribers can access the advanced analytics module only if they have the professional tier subscription. Select the mathematical constraint that corresponds to this English sentence.",
    'options_list': [
        'y_analytics_module <= y_pro_subscription',
        'y_pro_subscription = 1 - y_analytics_module',
        'y_analytics_module >= M * y_pro_subscription',
        'y_pro_subscription + y_basic_subscription <= 1'
    ],
    'correct_answer': 'y_analytics_module <= y_pro_subscription',
    'explanation': "This constraint specifies that the advanced analytics module is an exclusive feature available only to users with a professional tier subscription.",
    'chapter_information': 'Chapter 15: Optimization'
}

kc1_gpt_question_13_1 = {
    'question': "(GPT generated) What type of distribution models the number of sales transactions completed each day at a retail store?",
    'options_list': [
        'Binomial',
        'Exponential',
        'Geometric',
        'Poisson'
    ],
    'correct_answer': 'Poisson',
    'explanation': "The Poisson distribution is typically used to model the number of events occurring within a fixed period of time, such as sales transactions per day.",
    'chapter_information': 'Chapter 13: Probability-Based Models'
}

kc1_gpt_question_13_2 = {
    'question': "(GPT generated) Which distribution would be appropriate for modeling the number of tries until a software developer successfully compiles a program without errors?",
    'options_list': [
        'Binomial',
        'Exponential',
        'Geometric',
        'Poisson'
    ],
    'correct_answer': 'Geometric',
    'explanation': "The geometric distribution models the number of trials until the first success, which is fitting for counting the number of compile attempts until success.",
    'chapter_information': 'Chapter 13: Probability-Based Models'
}

kc1_gpt_question_13_3 = {
    'question': "(GPT generated) What distribution best describes the length of time a customer spends waiting on hold with customer service before being connected to an agent?",
    'options_list': [
        'Binomial',
        'Exponential',
        'Geometric',
        'Poisson'
    ],
    'correct_answer': 'Exponential',
    'explanation': "The exponential distribution is commonly used to model the time between occurrences in a Poisson process, such as waiting times.",
    'chapter_information': 'Chapter 13: Probability-Based Models'
}

kc1_gpt_question_13_4 = {
    'question': "(GPT generated) If you're modeling the occurrence of rare natural events in a given year, which distribution would be the most suitable?",
    'options_list': [
        'Binomial',
        'Exponential',
        'Geometric',
        'Poisson'
    ],
    'correct_answer': 'Poisson',
    'explanation': "The Poisson distribution is well-suited for modeling the count of randomly occurring rare events over a specified period of time or in a specified area.",
    'chapter_information': 'Chapter 13: Probability-Based Models'
}

kc11__gpt_question_11_4 = {
    'question': "(GPT generated) Which formula represents the objective of Ridge Regression?",
    'options_list': [
        'Minimize: ∑(yi−∑βjxij)^2 subject to ∑|βj|≤t',
        'Minimize: ∑(yi−∑βjxij)^2 + λ1∑|βj| + λ2∑βj^2',
        'Minimize: ∑(yi−∑βjxij)^22 + λ∑βj^2',
        'Minimize: ∑(yi−∑βjxij)^2 + λ∑βj'
    ],
    'correct_answer': 'Minimize: ∑(yi−∑βjxij)^2 + λ∑βj^2',
    'explanation': "Ridge Regression aims to minimize the sum of squared residuals with a penalty on the size of the coefficients, controlled by λ, to avoid overfitting.",
    'chapter_information': 'Chapter 11: Variable Selection'
}

kc11__gpt_question_11_5 = {
    'question': "(GPT generated) Which variable selection method combines both Lasso and Ridge approaches?",
    'options_list': [
        'Forward Selection',
        'Backward Elimination',
        'Stepwise Regression',
        'Elastic Net'
    ],
    'correct_answer': 'Elastic Net',
    'explanation': "Elastic Net is a variable selection method that combines the L1 penalty of Lasso for sparsity and the L2 penalty of Ridge for coefficient shrinkage.",
    'chapter_information': 'Chapter 11: Variable Selection'
}

kc11__gpt_question_11_6 = {
    'question': "(GPT generated) In Ridge Regression, what does the tuning parameter λ control?",
    'options_list': [
        'The size of coefficients.',
        'The sum of the absolute values of the coefficients.',
        'The number of factors in the model.',
        'The p-value thresholds.'
    ],
    'correct_answer': 'The size of coefficients.',
    'explanation': "In Ridge Regression, the tuning parameter λ controls the extent of shrinkage applied to the coefficients; higher values of λ lead to greater shrinkage.",
    'chapter_information': 'Chapter 11: Variable Selection'
}

kc_other_question_23 = {
    'question': "(GPT generated) In optimization models, what are the three key components?",
    'options_list': [
        'Data, constraints, and objectives',
        'Variables, constraints, and an objective function',
        'Variables, equations, and parameters',
        'Decisions, goals, and constraints'
    ],
    'correct_answer': 'Variables, constraints, and an objective function',
    'explanation': "The foundational elements of an optimization model are the decision variables that represent choices, the constraints that restrict solutions, and the objective function that defines the goal of optimization.",
    'chapter_information': 'Other'
}

kc_other_question_24 = {
    'question': "(GPT generated) What is a dynamic program in optimization modeling?",
    'options_list': [
        'A program that adapts to changing constraints in real-time.',
        'A program that uses complex mathematical equations to optimize solutions.',
        "A program that divides the system into states, decisions, and uses Bellman's equation to determine optimal decisions.",
        'A program that focuses on optimizing linear functions.'
    ],
    'correct_answer': "A program that divides the system into states, decisions, and uses Bellman's equation to determine optimal decisions.",
    'explanation': "Dynamic programming is a method for solving complex problems by breaking them down into simpler subproblems. It is applicable to problems exhibiting the properties of overlapping subproblems and optimal substructure.",
    'chapter_information': 'Other'
}

kc_other_question_25 = {
    'question': "(GPT generated) McNemar's Test is primarily designed for:",
    'options_list': [
        'Comparing the means of two independent samples.',
        'Analyzing correlations between two continuous variables.',
        'Analyzing categorical data in a 2x2 contingency table.',
        'Predicting future outcomes based on historical data.'
    ],
    'correct_answer': 'Analyzing categorical data in a 2x2 contingency table.',
    'explanation': "McNemar's Test is used in statistics to determine whether there are differences on a dichotomous dependent variable between two related groups. It is applicable to a 2x2 contingency table with a binary outcome.",
    'chapter_information': 'Other'
}

kc_other_question_26 = {
    'question': "(GPT generated) When should the Wilcoxon Signed Rank Test be used?",
    'options_list': [
        'When analyzing normally distributed data.',
        'When comparing two independent samples.',
        'When working with non-normal data in paired samples.',
        'When conducting before-and-after studies.'
    ],
    'correct_answer': 'When working with non-normal data in paired samples.',
    'explanation': "The Wilcoxon Signed Rank Test is a non-parametric statistical hypothesis test for comparing two related samples, matched samples, or repeated measurements on a single sample to assess whether their population mean ranks differ.",
    'chapter_information': 'Other'
}

kc_other_question_27 = {
    'question': "(GPT generated) The Mann-Whitney Test is particularly useful for comparing:",
    'options_list': [
        'Means from two independent samples.',
        'Medians from two independent samples.',
        'Variances from two paired samples.',
        'Proportions from two dependent samples.'
    ],
    'correct_answer': 'Medians from two independent samples.',
    'explanation': "The Mann-Whitney U Test is used to compare differences between two independent groups when the dependent variable is either ordinal or continuous, but not normally distributed.",
    'chapter_information': 'Other'
}

kc_other_question_28 = {
    'question': "(GPT generated) The Louvain Algorithm is primarily used for:",
    'options_list': [
        'Solving linear equations.',
        'Detecting communities in large networks.',
        'Analyzing textual data.',
        'Predicting stock prices.'
    ],
    'correct_answer': 'Detecting communities in large networks.',
    'explanation': "The Louvain Algorithm is a method to extract communities from large networks based on modularity optimization. It is widely used in network analysis to detect the division of nodes into groups.",
    'chapter_information': 'Other'
}

kc_other_question_29 = {
    'question': "(GPT generated) In the context of game theory, what does Nash Equilibrium refer to?",
    'options_list': [
        'The point where all players lose.',
        'The point where no player can improve their outcome by changing their strategy.',
        'The point where one player dominates all others.',
        'The point where all players cooperate fully.'
    ],
    'correct_answer': 'The point where no player can improve their outcome by changing their strategy.',
    'explanation': "A Nash Equilibrium is a solution concept in game theory where no player can benefit by unilaterally changing their strategy, given the other players' strategies remain unchanged.",
    'chapter_information': 'Other'
}

kc_other_question_30 = {
    'question': "(GPT generated) In Bayesian modeling, what is the purpose of the posterior distribution?",
    'options_list': [
        'To represent the prior beliefs and knowledge of the modeler.',
        'It describes the likelihood of observed data given the model parameters.',
        'To summarize the historical data used in the model.',
        'It quantifies the uncertainty about model parameters after considering both prior beliefs and observed data.'
    ],
    'correct_answer': 'It quantifies the uncertainty about model parameters after considering both prior beliefs and observed data.',
    'explanation': "In Bayesian inference, the posterior distribution combines the prior distribution and the likelihood of observed data to form a new distribution that reflects updated beliefs about the model parameters.",
    'chapter_information': 'Other'
}


kc11_question_11_7_gpt = {
    'question': "(GPT generated) Which variable selection technique is particularly useful when there is a risk of multicollinearity affecting the model?",
    'options_list': [
        'Forward Selection',
        'Lasso Regression',
        'Ridge Regression',
        'Elastic Net'
    ],
    'correct_answer': 'Ridge Regression',
    'explanation': "Ridge Regression is known to handle multicollinearity well by imposing a penalty on the size of coefficients, which helps in reducing the variance without substantially increasing the bias.",
    'chapter_information': 'Chapter 11: Variable Selection'
}

kc11_question_11_8_gpt = {
    'question': "(GPT generated) In the context of Elastic Net, what is the effect of combining Lasso and Ridge regularization terms in the model?",
    'options_list': [
        'It leads to selection of more variables than Lasso alone.',
        'It creates a more biased model than using Ridge alone.',
        'It balances variable selection and coefficient size reduction.',
        'It increases the complexity of the model unnecessarily.'
    ],
    'correct_answer': 'It balances variable selection and coefficient size reduction.',
    'explanation': "Elastic Net regularization takes advantage of both Lasso's variable selection capabilities and Ridge's coefficient shrinkage, leading to a model that is robust against multicollinearity while also performing variable selection.",
    'chapter_information': 'Chapter 11: Variable Selection'
}

kc11_question_11_9_gpt = {
    'question': "(GPT generated) When applying the Lasso Regression method for variable selection, what outcome is expected when the sum of the absolute values of the coefficients is constrained?",
    'options_list': [
        'All coefficients will be set to zero.',
        'Coefficients of less important variables will tend to be exactly zero.',
        'The model will retain all variables irrespective of their significance.',
        'Coefficients will not be affected by the constraint applied.'
    ],
    'correct_answer': 'Coefficients of less important variables will tend to be exactly zero.',
    'explanation': "Lasso Regression can set the coefficients of less significant variables to zero, effectively performing variable selection by excluding them from the model.",
    'chapter_information': 'Chapter 11: Variable Selection'
}

kc11_question_11_10_gpt = {
    'question': "(GPT generated) What is a primary consideration when choosing between Lasso, Ridge, and Elastic Net for variable selection?",
    'options_list': [
        'The number of predictors in the dataset.',
        'The presence of multicollinearity among the predictors.',
        'The computational resources available for model training.',
        'The preferred programming language for implementing the model.'
    ],
    'correct_answer': 'The presence of multicollinearity among the predictors.',
    'explanation': "The presence of multicollinearity is a critical factor when deciding between these methods, as Ridge can handle multicollinearity well, Lasso performs variable selection by setting some coefficients to zero, and Elastic Net balances both aspects.",
    'chapter_information': 'Chapter 11: Variable Selection'
}

kc12_gpt = {
    'question': "(GPT generated) Why is A/B testing considered a fundamental method in the design of experiments (DOE)?",
    'options_list': [
        'Because it allows for testing multiple variables simultaneously',
        'Because it identifies the best solution by comparing two variants against a single metric',
        'Because it utilizes complex statistical models to predict outcomes',
        'Because it can be used without any data'
    ],
    'correct_answer': 'Because it identifies the best solution by comparing two variants against a single metric',
    'explanation': "A/B testing is fundamental in DOE due to its straightforward approach of comparing two versions (A and B) against each other to identify which one performs better on a specific metric, making it a valuable tool for decision-making based on empirical data.",
    'chapter_information': 'Chapter 12: Design of Experiments'
}

kc12gpt1 = {
    'question': "(GPT generated) In factorial designs, what advantage does analyzing interactions between factors offer?",
    'options_list': [
        'It simplifies the model by reducing the number of factors',
        'It reveals how different factors affect each other and the outcome',
        'It guarantees that the experimental results will be statistically significant',
        'It allows for the experiment to be conducted without a control group'
    ],
    'correct_answer': 'It reveals how different factors affect each other and the outcome',
    'explanation': "Factorial designs allow for the analysis of interactions between factors, which can reveal synergies or conflicts among them, thereby offering deeper insights into how different variables collectively influence the experimental outcome.",
    'chapter_information': 'Chapter 12: Design of Experiments'
}

kc13_questsgption_13_5 = {
    'question': "(GPT generated) How does the Poisson distribution fundamentally differ from the Binomial distribution in modeling events?",
    'options_list': [
        'The Poisson distribution is used for modeling continuous events, while the Binomial is for discrete events',
        'The Poisson distribution models events in a fixed interval of time or space, whereas the Binomial models a fixed number of trials',
        'The Poisson distribution requires knowledge of previous outcomes, while the Binomial does not',
        'Both distributions are interchangeable in modeling scenarios'
    ],
    'correct_answer': 'The Poisson distribution models events in a fixed interval of time or space, whereas the Binomial models a fixed number of trials',
    'explanation': "The Poisson distribution is ideal for modeling the count of events happening within a fixed period of time or space with a known average rate and independently of the time since the last event, unlike the Binomial distribution, which is used when the number of trials and the probability of success on each trial are known.",
    'chapter_information': 'Chapter 13: Probability-Based Models'
}

kc13_quesgpt = {
    'question': "(GPT generated) What is the primary use of the Weibull distribution in reliability engineering?",
    'options_list': [
        'To predict the exact time to failure of a component',
        'To model the rate of returns in the stock market',
        'To analyze the life data of products and materials',
        'To determine the probability of success in randomized trials'
    ],
    'correct_answer': 'To analyze the life data of products and materials',
    'explanation': "The Weibull distribution is widely used in reliability engineering to analyze life data, providing a flexible model for representing the time until failure of products and materials, which can accommodate various shapes of hazard functions.",
    'chapter_information': 'Chapter 13: Probability-Based Models'
}


kc15_question_15ggg_3 = {
    'question': "(GPT generated) In optimization modeling, what role do binary variables play?",
    'options_list': [
        'They represent the continuous outcomes of a decision.',
        'They are used to optimize the objective function directly.',
        'They model yes/no decisions within the optimization problem.',
        'They determine the linear relationships between variables.'
    ],
    'correct_answer': 'They model yes/no decisions within the optimization problem.',
    'explanation': "Binary variables are crucial in optimization models for representing decisions that have two states, typically yes or no, enabling the modeling of complex decision structures.",
    'chapter_information': 'Week 10 – Module 15: Optimization'
}



KC_MPC_QUESTIONS = []
global_items = list(globals().items())
# print(global_items)

for name, value in global_items:
    if not name.startswith('_'):
        KC_MPC_QUESTIONS.append(value)

KC_MPC_QUESTIONS = KC_MPC_QUESTIONS[:-1]
