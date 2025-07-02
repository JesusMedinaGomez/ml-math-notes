# Gradient Descent

As discussed earlier, our goal when training linear models is to find the parameter vector values $ \theta $ that minimize the cost function $ J(\theta) $, which measures the discrepancy between the model predictions and the actual observed values.

We have already seen that there are analytical formulas to obtain the optimal values of $ \theta $, such as the normal equations. However, when the number of independent variables ($ d $) or observations ($ n $) is very large, these formulas become impractical or computationally expensive. Additionally, if some variables contribute little or no information to the model, manually tuning parameters can be inefficient.

Fortunately, there is a very popular and effective optimization technique called **gradient descent**, which allows us to iteratively approximate the optimal values of $ \theta $ starting from an initial guess. This technique is particularly useful when working with large datasets or complex models.

## Fundamental Idea

To understand how gradient descent works, we first need to grasp the meaning of the gradient vector of the cost function:

$$
\nabla_{\theta} J(\theta)
$$

This vector indicates the direction in which $ J(\theta) $ increases most rapidly. Since we want to minimize $ J $, we aim to move in the opposite direction. Therefore, we use the negative gradient:

$$
-\nabla_{\theta} J(\theta),
$$

which points towards the direction in which $ J(\theta) $ decreases most rapidly.

## Gradient Descent Algorithm

The process begins by choosing an initial value for the parameters, for example:

$$
\theta^{(0)} =
\begin{pmatrix}
\theta_0^{(0)} \\
\theta_1^{(0)} \\
\vdots \\
\theta_d^{(0)}
\end{pmatrix}
=
\begin{pmatrix}
\text{random} \\
\text{random} \\
\vdots \\
\text{random}
\end{pmatrix}
$$

Since these initial values are unlikely to represent a good solution, we will adjust them step by step in the direction of the minimum. For this purpose, we define a learning rate $ \alpha > 0 $ (usually between 0.001 and 0.1), a small number that regulates the size of the steps we take.

The parameter update in each iteration is defined as:

$$
\theta^{(t+1)} = \theta^{(t)} - \alpha \nabla_{\theta} J(\theta^{(t)})
$$

This means that, at each step, we move in the direction of the negative gradient, scaled by $ \alpha $ to control the learning speed.

Each component of the vector is updated as:

$$
\theta_j^{(t+1)} = \theta_j^{(t)} - \alpha \frac{\partial J(\theta^{(t)})}{\partial \theta_j}
\quad \text{for } j = 0, 1, \ldots, d
$$

where the denominator does not carry the superscript $ t $ because the derivative is taken with respect to $ \theta_j $ evaluated at $ \theta^{(t)} $.

By differentiating $ J(\theta) $ in the linear regression case, we obtain:

$$
\theta_j^{(t+1)} =
\theta_j^{(t)} - \alpha \cdot \frac{1}{n}
\sum_{i=1}^{n}
\left( \hat{y}^{(i)} - y^{(i)} \right) x_j^{(i)}
\quad \text{for } j = 0, 1, \ldots, d
$$

Repeating this process multiple times yields a sequence of vectors:

$$
\theta^{(0)}, \quad \theta^{(1)}, \quad \theta^{(2)}, \quad \ldots
$$

that converge towards a minimum of $ J(\theta) $.

To simplify and speed up computation, we can express the update of all parameters simultaneously using matrix-vector notation:

$$
\theta^{(t+1)} = \theta^{(t)} - \alpha \cdot \frac{1}{n} X^{T} \left( \hat{y} - y \right)
$$

where $ X $ is the design matrix including all samples and features, $ \hat{y} $ is the vector of predictions, and $ y $ is the vector of actual values.

This vectorized form is not only more compact and elegant but also computationally much more efficient than calculating each component separately in a loop. This is because matrix operations are highly optimized in numerical libraries and can take advantage of parallelism and cache memory, significantly reducing computation time, especially for large datasets.

> **How do we know when to stop?**

There are several strategies to decide when to terminate gradient descent. Some of the most common are:

- **Fixed number of iterations:** the algorithm runs for a predefined number of steps.
- **Minimal change:** it stops when the change between two consecutive iterations of $ J(\theta) $ is smaller than a threshold $ \epsilon $, i.e.:

  $$
  |J(\theta^{(t+1)}) - J(\theta^{(t)})| < \epsilon
  $$
- **Gradient norm:** it stops when the magnitude of the gradient becomes very small, i.e.:

  $$
  \| \nabla_{\theta} J(\theta^{(t)}) \| < \epsilon
  $$

> **Note:**  
> The value of the learning rate $ \alpha $ is crucial. If it’s too small, the algorithm may take too long to converge. If it’s too large, it may cause oscillations or even divergence. Choosing $ \alpha $ appropriately is an important part of model design.

> **Observation:**  
> Gradient descent is the foundation of many machine learning algorithms, including training neural networks, and can be extended to more sophisticated versions such as stochastic gradient descent (SGD), mini-batch gradient descent, and adaptive methods like Adam or RMSprop.

## Gradient Descent Algorithm

In this section, we’ll play a bit with notation. For some mathematicians, seeing expressions like:

$$
x = x + 1
$$

causes a mini heart attack. From a purely mathematical perspective, this might lead us to absurd conclusions such as $ 0 = 1 $. However, in the realm of computing, this expression has a very different meaning. For example, in Python, we might write:

```python
x = 5
x = x + 1
```

As long as a value has been previously defined for `x`, there’s no error. What happens in this code is simply an **update** of the value of `x`. In the first line, `x` equals 5, and in the second line it becomes $ 5 + 1 = 6 $.

This clarification is fundamental, as in mathematics we typically use notations like:

$$
\theta_i^{(t)}
$$

whereas in programming, more practical and operational notations are common. Therefore, throughout this document, expressions such as $ x = x + 1 $ will be perfectly valid and understood as updates of a variable’s value.

The gradient descent algorithm starts from an initial value for the parameter vector, denoted as $ \theta^{(0)} $, and updates it iteratively to minimize the cost function $ J(\theta) $.

To simplify notation, we work directly with a single vector $ \theta \in \mathbb{R}^{(d+1) \times 1} $, whose components are $ \theta_j $, with $ j = 0, 1, \ldots, d $.

The update rule for each component $ \theta_j $ is expressed as:

$$
\theta_j =
\theta_j - \alpha \cdot \frac{1}{n}
\sum_{i=1}^{n}
\left( \hat{y}^{(i)} - y^{(i)} \right) x_j^{(i)}
\quad \text{for } j = 0, 1, \ldots, d.
$$
