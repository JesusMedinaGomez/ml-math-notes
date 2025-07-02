# Linear Regression

Linear regression is a statistical and mathematical technique used to make predictions based on a linear model. The goal is to find a function that describes, in the simplest way possible, the relationship between a dependent variable and one or more independent variables, based on a set of observed data.

Since in practice observations are usually discrete and limited, we often lack exact records for every possible value of the independent variables. In such cases, linear regression allows us to construct an approximate function that closely resembles the behavior of the data.

When working with a single independent variable, the resulting model is a line that tries to fit the observed points. This case is known as **simple linear regression**, and is expressed by the equation:

$$
\hat{y} = \theta_0 + \theta_1 x
$$

where $ \theta_1 $ represents the slope of the line, $ \theta_0 $ is the vertical intercept, and $\hat{y} $ is the estimated value of the dependent variable. The model attempts to make the line pass as close as possible to all data points, minimizing some measure of error, such as the mean squared error.

If multiple independent variables are used, the model extends to what is known as **multiple linear regression**, fitting a plane or hyperplane that represents the linear relationship between the dependent variable and the independent variables:

$$
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n
$$
