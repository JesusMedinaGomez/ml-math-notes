# Simple Linear Regression

To use a simple linear regression model, we need a dataset $X$ with $ n $ observations, each composed of an independent variable $ x^{(i)} $ and a dependent variable $ y^{(i)} $, where $i = 1, \dots, n $. The goal is to find a linear function relating both variables so that values of $ x $ can estimate corresponding values of $ y $.

:::{admonition} **Example**
:class: tip
Seven employees’ monthly salaries and years of work experience are given. The goal is to model the salary based on experience.

| Monthly Salary (\$) | 1000 | 2000 | 2500 | 3500 | 4000 | 5000 | 5500 |
|---------------------|------|------|------|------|------|------|------|
| Experience (years)  |  0   |  1   |  3   |  5   |  9   | 12   | 15   |

In this case, $n = 7 $, the independent variable $ x^{(i)} $ is the experience, and the dependent variable $ y^{(i)} $ is the salary.  
For example, the fourth employee has $x^{(4)} = 5 $ years of experience and $ y^{(4)} = 3500 $ pesos.

A plot of the data points and a fitted line would look like this:

![Salary vs Experience](../_static/gpx1.png)
::::

To find the best-fitting line, we assume there exists a linear function \( h: X \to \mathbb{R} \) called the **hypothesis** of the model.

:::{admonition} **Definition (Hypothesis)**
:class: note
A hypothesis is a linear function that, given an input $x^{(i)}$, returns a predicted value for $y^{(i)}$:

$$
h_\theta(x^{(i)}) = \theta_0 + \theta_1 x^{(i)}
$$

where the parameter vector is $ \theta = \langle \theta_0, \theta_1 \rangle $.
:::


The difference between the hypothesis output and the true value \( y^{(i)} \) is called the **error**:

$$
h_\theta(x^{(i)}) - y^{(i)}
$$

To focus only on the magnitude of the error and avoid cancellation, we square the difference:

$$
(h_\theta(x^{(i)}) - y^{(i)})^2 \geq 0
$$

:::{dropdown} **Observation**
:class: tip
We could also use the absolute value to measure error, but the absolute function is not differentiable at 0. The square function, however, is differentiable everywhere, which makes it more suitable for optimization.
::::

To minimize the total error over all $ n $ observations, we sum all squared errors and divide by $ 2n $ to define the **cost function**.

:::{admonition} **Definition (Cost function)**
:class: note
The cost function measures the average squared error of the hypothesis with respect to the observed data:

$$
J(\theta) = \frac{1}{2n} \sum_{i=1}^n \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2
$$
::::

The cost function is a strictly convex quadratic function in $ \theta_0 $ and $ \theta_1 $, so its global minimum occurs where the gradient is zero:

$$
\nabla_\theta J(\theta) = 0 \quad \Rightarrow \quad \frac{\partial J}{\partial \theta_0} = 0, \quad \frac{\partial J}{\partial \theta_1} = 0
$$

:::{dropdown} **Observation**
:class: tip
To simplify the notation, we define the mean of the data $a^{(i)}$, for $i = 1, 2, \dots, n$, as:

$$
\bar{a} = \frac{1}{n} \sum_{i=1}^{n} a^{(i)}
$$

:::
---

## Partial derivative with respect to $ \theta_0 $

$$
\begin{align*}
\frac{\partial J}{\partial \theta_0}
&= \frac{\partial}{\partial \theta_0} \left[ \frac{1}{2n} \sum_{i=1}^n \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 \right] \\
&= \frac{1}{2n} \sum_{i=1}^n \frac{\partial}{\partial \theta_0} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 \\
&= \frac{1}{2n} \sum_{i=1}^n 2 \left( h_\theta(x^{(i)}) - y^{(i)} \right) \cdot \frac{\partial}{\partial \theta_0} \left( h_\theta(x^{(i)}) - y^{(i)} \right) \\
&= \frac{1}{n} \sum_{i=1}^n \left( h_\theta(x^{(i)}) - y^{(i)} \right) \cdot \frac{\partial}{\partial \theta_0} \left( \theta_0 + \theta_1 x^{(i)} - y^{(i)} \right) \\
&= \frac{1}{n} \sum_{i=1}^n \left( h_\theta(x^{(i)}) - y^{(i)} \right)
\end{align*}
$$

Expanding:

$$
\begin{align*}
\frac{1}{n} \sum_{i=1}^n \left( h_\theta(x^{(i)}) - y^{(i)} \right)
&= \frac{1}{n} \sum_{i=1}^n \left( \theta_0 + \theta_1 x^{(i)} - y^{(i)} \right) \\
&= \frac{1}{n} \left( \sum_{i=1}^n \theta_0 + \sum_{i=1}^n \theta_1 x^{(i)} - \sum_{i=1}^n y^{(i)} \right) \\
&= \frac{1}{n} \left( n \theta_0 + \theta_1 \sum_{i=1}^n x^{(i)} - \sum_{i=1}^n y^{(i)} \right) \\
&= \theta_0 + \theta_1 \bar{x} - \bar{y}
\end{align*}
$$


To cancel the partial derivative $\theta_0 + \theta_1 \bar{x} - \bar{y} = 0$. Solving for $ \theta_0 $:

$$
\theta_0 = \bar{y} - \theta_1 \bar{x}
$$

---

## Partial derivative with respect to $ \theta_1 $

$$
\begin{align*}
\frac{\partial J}{\partial \theta_1}
&= \frac{\partial}{\partial \theta_1} \left[ \frac{1}{2n} \sum_{i=1}^n \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 \right] \\
&= \frac{1}{2n} \sum_{i=1}^n \frac{\partial}{\partial \theta_1} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 \\
&= \frac{1}{2n} \sum_{i=1}^n 2 \left( h_\theta(x^{(i)}) - y^{(i)} \right) \cdot \frac{\partial}{\partial \theta_1} \left( h_\theta(x^{(i)}) - y^{(i)} \right) \\
&= \frac{1}{n} \sum_{i=1}^n \left( h_\theta(x^{(i)}) - y^{(i)} \right) \cdot \frac{\partial}{\partial \theta_1} \left( \theta_0 + \theta_1 x^{(i)} - y^{(i)} \right) \\
&= \frac{1}{n} \sum_{i=1}^n \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}
\end{align*}
$$

If we follow the same procedure as with $\theta_0$, then we obtain:

$$
\begin{align*}
    \frac{1}{n} \sum_{i=1}^n \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)} &= \frac{1}{n} \sum_{i=1}^n \left( \theta_0 + \theta_1 x^{(i)}  - y^{(i)} \right) x^{(i)}\\
    &= \frac{1}{n} \sum_{i=1}^n \left( \bar{y}- \theta_1 \bar{x} + \theta_1 x^{(i)}  - y^{(i)} \right) x^{(i)}\\
    &= \frac{1}{n} \sum_{i=1}^n \left( x^{(i)}\bar{y} - \theta_1 x^{(i)}\bar{x} + \theta_1x^{(i)} x^{(i)} - x^{(i)}y^{(i)} \right)  \\
    &= \frac{1}{n}  \left( \bar{y}\sum_{i=1}^n x^{(i)} - \theta_1\bar{x}\sum_{i=1}^n x^{(i)} + \theta_1\sum_{i=1}^nx^{(i)} x^{(i)} - \sum_{i=1}^nx^{(i)}y^{(i)} \right)  \\
    &=   \bar{y} \left(\frac{1}{n}\sum_{i=1}^n x^{(i)}\right) - \theta_1\bar{x}\left(\frac{1}{n}\sum_{i=1}^n x^{(i)}\right) + \theta_1\left(\frac{1}{n}\sum_{i=1}^nx^{(i)} x^{(i)}\right) - \left(\frac{1}{n}\sum_{i=1}^nx^{(i)}y^{(i)}\right)   \\
    &=   \bar{y} \bar{x} - \theta_1\bar{x}^2 + \theta_1\left(\frac{1}{n}\sum_{i=1}^n(x^{(i)})^2\right) - \left(\frac{1}{n}\sum_{i=1}^nx^{(i)}y^{(i)}\right)   \\
    &=   \bar{y} \bar{x} - \left(\frac{1}{n}\sum_{i=1}^nx^{(i)}y^{(i)}\right) + \theta_1\left( \frac{1}{n}\sum_{i=1}^n(x^{(i)})^2-\bar{x}^2 \right)   \\
    &= 0
\end{align*}
$$

Solving for $\theta_1$:

$$
\theta_1 = \frac{\frac{1}{n}\sum_{i=1}^nx^{(i)}y^{(i)}-\bar{y} \bar{x} }{  \frac{1}{n}\sum_{i=1}^n (x^{(i)})^2 - \bar{x}^2}
$$
With this, we have obtained two equations called the **normal equations**, which we use to find the linear regression parameters $\theta_0$ and $\theta_1$.
