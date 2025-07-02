# Multiple Linear Regression

When working with multiple independent variables, the scalar notation used in simple linear regression becomes cumbersome and inefficient. In such cases, it is preferable to use a **matrix representation**, which allows us to handle all data and operations compactly and efficiently.

:::{admonition} **Example**
:class: tip
A group of students participated in a study measuring their final exam scores based on **study hours per week** and **hours of sleep per night**.

| Final Score ($y$)     | 55  | 65  | 70  | 75  | 80  | 85  | 90  |
|-----------------------|-----|-----|-----|-----|-----|-----|-----|
| Study Hours ($x_1$)   | 1   | 2   | 3   | 4   | 5   | 6   | 7   |
| Sleep Hours ($x_2$)   | 6.5 | 7.0 | 6.8 | 7.2 | 7.5 | 7.8 | 8.0 |

In this case, $n = 7$, the independent variables are:
- $x_1^{(i)}$: number of study hours per week,
- $x_2^{(i)}$: average hours of sleep per night.

The dependent variable $y^{(i)}$ is the final exam score.

The regression model is:

$$
h_\theta(x^{(i)}) = \theta_0 + \theta_1 x_1^{(i)} + \theta_2 x_2^{(i)}
$$

:::


The hypothesis in matrix notation is expressed as:

$$
\hat{y} = X\theta
$$

Where:
- $n$ is the number of observations (rows in the dataset),
- $d$ is the number of independent variables (also called features),
- $X \in \mathbb{R}^{n \times (d+1)}$ is the **design matrix**, where each row represents one observation. The first column consists of ones to include the intercept term $\theta_0$, and the next $d$ columns contain the values of the independent variables:

  $$
  X = \begin{pmatrix}
  1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_d^{(1)} \\
  1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_d^{(2)} \\
  \vdots & \vdots & \vdots & \ddots & \vdots \\
  1 & x_1^{(n)} & x_2^{(n)} & \cdots & x_d^{(n)}
  \end{pmatrix}
  $$

- $\theta \in \mathbb{R}^{(d+1) \times 1}$ is the **parameter vector**:

  $$
  \theta = \begin{pmatrix}
  \theta_0 \\
  \theta_1 \\
  \vdots \\
  \theta_d
  \end{pmatrix}
  $$

- $\hat{y} \in \mathbb{R}^{n}$ is the **prediction vector**, where each entry represents the estimated output for an observation:

  $$
  \hat{y} = \begin{pmatrix}
  \hat{y}^{(1)} \\
  \hat{y}^{(2)} \\
  \vdots \\
  \hat{y}^{(n)}
  \end{pmatrix}
  $$


---

## From Scalar to Matrix Form of the Cost Function

In scalar notation, the cost function is defined as:

$$
J(\theta) = \frac{1}{2n} \sum_{i=1}^n \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

If we define the prediction vector as $\hat{y} = X\theta$, and the true output vector as $y \in \mathbb{R}^n$, then the **residual vector** is:

$$
X\theta - y
$$

This is a column vector of length $n$, and its transpose is a row vector. Multiplying the residual vector by its transpose gives:

$$
(X\theta - y)^T (X\theta - y)
$$

This expression sums the squared prediction errors:

$$
(X\theta - y)^T (X\theta - y) = \sum_{i=1}^n \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

Thus, the cost function in matrix form is:

$$
J(\theta) = \frac{1}{2n} (X\theta - y)^T (X\theta - y)
$$

---

## Minimizing the Cost Function

To find the optimal value of $\theta$, we differentiate the cost function with respect to $\theta$ and set the derivative equal to zero:

$$
\nabla_\theta J(\theta) = \frac{1}{n} X^T(X\theta - y) = 0
$$

Solving this equation leads to the **normal equations**:

$$
\boxed{\theta = (X^T X)^{-1} X^T y}
$$

This formula gives us the parameter vector $\theta$ that minimizes the mean squared error.

:::{dropdown} **Observation**
:class: tip
For this solution to exist, the matrix $X^T X$ must be invertible. In practice, when it is not, regularization techniques such as **Ridge Regression** are used to avoid singularity problems.
:::

This matrix formulation generalizes simple linear regression to the multivariable case. When $d = 1$, the model reduces to the univariate case. However, this generalization allows us to handle multiple features simultaneously, which is essential for real-world datasets with high dimensionality.
