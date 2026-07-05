- [[#1. Problem Formulation|1. Problem Formulation]]
- [[#2. Simple Linear Regression Model|2. Simple Linear Regression Model]]
- [[#3. The Least Squares Method|3. The Least Squares Method]]
- [[#4. Finding the Best Parameters (Derivations)|4. Finding the Best Parameters (Derivations)]]
- [[#5. Multiple Linear Regression (Matrix Form)|5. Multiple Linear Regression (Matrix Form)]]
- [[#6. Minimizing the Error (The Normal Equation)|6. Minimizing the Error (The Normal Equation)]]
- [[#7. Proving it is a Minimum (The Hessian Matrix)|7. Proving it is a Minimum (The Hessian Matrix)]]
- [[#1. The Column Space and The Goal|1. The Column Space and The Goal]]
- [[#2. Orthogonal Projection|2. Orthogonal Projection]]
- [[#3. The Residual Vector and the Normal Equation|3. The Residual Vector and the Normal Equation]]
- [[#1. When $X^TX$ Is Not Invertible|1. When $X^TX$ Is Not Invertible]]
- [[#2. The Null Space and Infinitely Many Solutions|2. The Null Space and Infinitely Many Solutions]]
- [[#3. The Moore-Penrose Pseudoinverse|3. The Moore-Penrose Pseudoinverse]]
- [[#4. Insight into the Role of the Bias Term ($w_0$)|4. Insight into the Role of the Bias Term ($w_0$)]]
- [[#5. Residual Sum of Squares (RSS) and Model Complexity|5. Residual Sum of Squares (RSS) and Model Complexity]]
- [[#1. The Probability Space|1. The Probability Space]]
- [[#2. Cumulative Distribution Function (CDF)|2. Cumulative Distribution Function (CDF)]]
- [[#3. Discrete Random Variables|3. Discrete Random Variables]]
- [[#4. Continuous Random Variables|4. Continuous Random Variables]]
- [[#5. Expectation (Discrete)|5. Expectation (Discrete)]]
- [[#1. Expectation (Continuous)|1. Expectation (Continuous)]]
- [[#2. Expectation of Functions|2. Expectation of Functions]]
- [[#3. Linearity of Expectation|3. Linearity of Expectation]]
- [[#4. Variance|4. Variance]]
- [[#5. Covariance and Correlation|5. Covariance and Correlation]]
- [[#6. Important Distributions|6. Important Distributions]]

## 1. Problem Formulation

**What do we have?**

A training dataset comprising $N$ observations $\{x_n\}$, where $n = 1, \dots, N$, together with corresponding labels $\{y_n\}$.

- **$y$** is a continuous numerical value (a scalar), not a discrete class label.
    
- **$x$'s** represent the features. Every individual input $x$ corresponds to a single output $y$.
    

**What is our goal?**

Our main objective is to find a function, denoted as $\hat{y}$ ($y$-hat), that can predict the best possible $y$ for any new input $x$. This means the difference between our predicted $\hat{y}$ and the true $y$ should be as small as possible.

More broadly, we want to model the predictive distribution $p(y \mid x)$—meaning we want to find the probability of $y$ given a specific $x$. Finding this $\hat{y}$ function is the core goal of the general regression problem, not just linear regression.

---

## 2. Simple Linear Regression Model

**What is the simplest function we can use for prediction?**

The simplest approach is a linear combination of the input variables $\mathbf{x} = (x_1, \dots, x_D)^T$:

$$y(\mathbf{x}, \mathbf{w}) = w_0 + w_1x_1 + \dots + w_Dx_D$$

The key property of this model is that **it is a linear function of the parameters (weights)** $w_0, \dots, w_D$.

Our task is to find these specific weights ($w$) so that the resulting predicted function is as close to the actual $y$ values as possible.

---

## 3. The Least Squares Method

Out of all the infinitely many lines we could draw through our data, we need to find the absolute best one. We do this using the **Least Squares Method**.

- **Objective:** Find the line that minimizes the total sum of squared errors (the distance from the points to the line).
    
- **Residual:** The difference between the actual observation ($y_i$) and the predicted value on the fitted line ($\hat{y}_i$).
    
- **Slope ($\hat{\beta}_1$):** The change in $y$ divided by the change in $x$.
    
- **Intercept ($\hat{\beta}_0$):** The value of $y$ when $x = 0$.
    

The prediction formula for a single variable is:

$$\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1x_i$$

Mathematically, our goal is to find the parameters $\hat{\beta}_0$ and $\hat{\beta}_1$ that minimize the sum of squared residuals:

$$\min_{\hat{\beta}_0, \hat{\beta}_1} \sum_{i=1}^N (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2$$

> **Why do we use squares instead of absolute values?**
> 
> We square the residuals primarily so that the cost function remains continuously **differentiable**. Absolute values create sharp corners in the function graph (like a V-shape), making it impossible to calculate the derivative at the minimum. Squaring creates a smooth curve (a parabola), allowing us to easily use calculus to find the exact minimum point.

### Understanding MSE (Mean Squared Error)

The average of these squared residuals is the MSE. Our goal is to find the betas that yield the smallest MSE.

- **The Intuition Catch:** While MSE is mathematically convenient because it's differentiable, it loses real-world intuition because the units are squared. For example, if you are predicting housing prices and your MSE is $100$, you must take the square root (Root Mean Squared Error, RMSE) to understand that your actual average error distance is $10$.
    

---

## 4. Finding the Best Parameters (Derivations)

**How do we find the minimum?**

In calculus, for a single variable, we find the minimum or maximum by taking the first derivative and setting it to zero. Then, we check the second derivative:

- If positive $\rightarrow$ it's a minimum point.
    
- If negative $\rightarrow$ it's a maximum point.
    
- If zero $\rightarrow$ it's undetermined.
    

Since we are dealing with multiple variables ($\hat{\beta}_0$ and $\hat{\beta}_1$), we calculate the **gradients** (partial derivatives) and set them to zero.

### Step 1: Partial Derivative with respect to the Intercept ($\hat{\beta}_0$)

$$\frac{\partial}{\partial \hat{\beta}_0} \sum_{i=1}^N (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2 = -2 \sum_{i=1}^N (y_i - (\hat{\beta}_0 + \hat{\beta}_1 x_i)) = 0$$

Dividing by -2 and expanding the sum:

$$\sum_{i=1}^N y_i - \sum_{i=1}^N \hat{\beta}_0 - \sum_{i=1}^N \hat{\beta}_1 x_i = 0$$

Since summing a constant $\hat{\beta}_0$ for $N$ times gives $N\hat{\beta}_0$:

$$\sum_{i=1}^N y_i - N\hat{\beta}_0 - \hat{\beta}_1 \sum_{i=1}^N x_i = 0$$

**Final equation for Intercept:**

$$\hat{\beta}_0 = \frac{\sum_{i=1}^N y_i - \hat{\beta}_1 \sum_{i=1}^N x_i}{N}$$

### Step 2: Partial Derivative with respect to the Slope ($\hat{\beta}_1$)

$$\frac{\partial}{\partial \hat{\beta}_1} \sum_{i=1}^N (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2 = -2 \sum_{i=1}^N x_i (y_i - (\hat{\beta}_0 + \hat{\beta}_1 x_i)) = 0$$

Expanding the terms and substituting the $\hat{\beta}_0$ formula we just found:

$$\sum_{i=1}^N x_i y_i - \hat{\beta}_0 \sum_{i=1}^N x_i - \hat{\beta}_1 \sum_{i=1}^N x_i^2 = 0$$

$$\sum_{i=1}^N x_i y_i - \left( \frac{\sum y_i - \hat{\beta}_1 \sum x_i}{N} \right) \sum_{i=1}^N x_i - \hat{\beta}_1 \sum_{i=1}^N x_i^2 = 0$$

Grouping the terms with $\hat{\beta}_1$ gives us the **Final equation for Slope:**

$$\hat{\beta}_1 = \frac{\sum_{i=1}^N x_i y_i - \frac{1}{N} \sum_{i=1}^N y_i \sum_{i=1}^N x_i}{\sum_{i=1}^N x_i^2 - \frac{1}{N} \left(\sum_{i=1}^N x_i\right)^2}$$

---

## 5. Multiple Linear Regression (Matrix Form)

Now, consider the case where we have multiple variables. We have $N$ data points, and each point is described by $K$ features. How do we find the betas now? We use the exact same logic, but we express it using linear algebra.

For $i = 1, \dots, N$:

$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_k x_{ik} + u_i$$

Each $y_i$ is equal to the dot product of the beta vector and the feature vector $x_i$, plus an error term $u_i$. Our goal is to minimize this error term. We can represent this entire system using matrices:

$$\begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix} = \begin{bmatrix} 1 & x_{11} & x_{12} & \dots & x_{1k} \\ 1 & x_{21} & x_{22} & \dots & x_{2k} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & x_{n2} & \dots & x_{nk} \end{bmatrix} \begin{bmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \\ \vdots \\ \beta_k \end{bmatrix} + \begin{bmatrix} u_1 \\ u_2 \\ \vdots \\ u_n \end{bmatrix}$$

This simplifies to:

$$Y = X\beta + u$$

$$u = Y - X\beta$$

---

## 6. Minimizing the Error (The Normal Equation)

We need to minimize the squared deviation for all points simultaneously. In matrix notation, this squared error is represented as $u^T u$. We must find the vector $\beta$ that minimizes $u^T u$, so we differentiate it.

First, expand $u^T u$:

$$u^T u = (Y - X\beta)^T (Y - X\beta)$$

$$= Y^T Y - Y^T X\beta - X\beta Y^T + X\beta^T X^T \beta$$

Since $\beta^T X^T Y$ is a scalar, it equals its transpose $Y^T X \beta$. Combining the middle terms:

$$u^T u = Y^T Y - 2Y^T X\beta + \beta^T X^T X\beta$$

Take the derivative with respect to the vector $\beta$ and set it to 0:

$$\frac{\partial u^T u}{\partial \beta} = -2X^T Y + 2X^T X\beta = 0$$

This gives us the **Normal Equation** for linear regression, which directly calculates the best $\beta$:

$$X^T X\beta = X^T Y$$

$$\beta = (X^T X)^{-1} X^T Y$$

> **Important Caveat:** We cannot always assume that the matrix $X^T X$ is invertible. If the determinant of $X^T X$ is zero (which happens if features are perfectly correlated/linearly dependent), the inverse does not exist, and there are infinitely many solutions for $\beta$.

---

## 7. Proving it is a Minimum (The Hessian Matrix)

We set the derivative to zero, but how do we definitively prove this point is a minimum and not a maximum?

We must calculate the second-order derivative, which for multiple variables is a matrix of partial derivatives called the **Hessian Matrix** ($H$).

$$H_f = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \dots \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \dots \end{bmatrix}$$

Let's compute the Hessian of our cost function $J$ with respect to $\beta$:

$$\nabla^2 J = \frac{2}{N} X^T X$$

To prove this matrix corresponds to a local minimum, we must show it is **positive semi-definite**. For any arbitrary vector $z$:

$$z^T(X^T X)z = (Xz)^T(Xz) = ||Xz||^2 \ge 0$$

Because the length squared of any vector ($||Xz||^2$) is always greater than or equal to zero:

1. $X^T X$ is positive semi-definite.
    
2. If the columns of $X$ are linearly independent (no perfect multicollinearity), then $X^T X$ is **strictly positive definite**.
    
3. A positive definite Hessian guarantees that the point we found is a **unique global minimum**.
    

---

# Geometric Interpretation of Multivariate Linear Regression

To truly understand what linear regression is doing behind the scenes, we can look at it through the lens of geometry and linear algebra.

## 1. The Column Space and The Goal

In linear regression, we are trying to minimize the squared distance between our actual target values ($y$) and our predicted values ($\hat{y}$). Mathematically, we minimize:

$$\min_w ||y - Xw||^2$$

Let's define our predicted values as:

$$\hat{y} = Xw$$

**Key Observation:**

Because $\hat{y}$ is formed by multiplying the matrix $X$ by the weight vector $w$, $\hat{y}$ is simply a linear combination of the columns of $X$.

Therefore, we can say that $\hat{y}$ belongs to the **Column Space of $X$**, denoted as $\text{Col}(X)$.

$$\hat{y} \in \text{Col}(X)$$

_Geometric Intuition:_ Our actual $y$ vector lives somewhere in a high-dimensional space, but our predictions ($\hat{y}$) are restricted to live _only_ within the subspace (the "plane") created by the columns of $X$. Our goal is to geometrically search the column space of $X$ to find the specific point that is **closest** to $y$.

## 2. Orthogonal Projection

What is the shortest distance from a point to a plane? It is the perpendicular line dropping straight down.

Therefore, on a geometric plane, finding the closest point in $\text{Col}(X)$ to $y$ is exactly the same as finding the **orthogonal projection** of $y$ onto $\text{Col}(X)$.

**Linear regression = Orthogonal projection of $y$ onto $\text{Col}(X)$.**

### Visualizing the Projection (Explaining the Diagrams)

Imagine a 3D space where our features form a flat 2D plane (this plane represents $\text{Col}(X)$).

1. **The Subspace ($\text{Col}(X)$):** The green/blue plane is spanned by the vectors of our features (e.g., $X_1$ and $X_2$). Any point on this plane is a possible prediction $\hat{y}$.
    
2. **The Target ($y$):** The vector $y$ points somewhere off into the 3D space, away from the plane.
    
3. **The Projection ($X\hat{\beta}$):** We shine a light straight down from $y$ onto the plane. The shadow it casts is $\hat{y}$ (or $X\hat{\beta}$). This is the best prediction linear regression can make.
    
4. **The Error ($\hat{\varepsilon}$):** The dotted line connecting our projection on the plane to the actual $y$ in space is the error (or residual). Because we dropped straight down, this line forms a perfect **90-degree right angle** with the plane.
    

## 3. The Residual Vector and the Normal Equation

Let's define the **residual** (the error vector) as the difference between the actual value and the prediction:

$$r = y - Xw$$

As we established visually, at the minimum distance, the residual vector must be completely orthogonal (perpendicular) to the subspace $\text{Col}(X)$. If it is orthogonal to the subspace, it must be orthogonal to _every_ column vector inside $X$.

In linear algebra, two vectors are orthogonal if their dot product is zero. So, at the minimum point:

$$X^Tr = 0$$

Let's substitute our definition of $r$ back into this equation:

$$X^T(y - Xw) = 0$$

$$X^Ty - X^TXw = 0$$

$$X^TXw = X^Ty$$

**This is the Normal Equation.** **Geometric Meaning:** The residual vector is orthogonal to every column of $X$ ($r \perp \text{Col}(X)$). The error vector being perpendicular to the subspace is exactly the mathematical condition required for an orthogonal projection. By solving this equation, we are finding the precise weights ($w$ or $\beta$) that project our data onto the closest point possible in the feature space!

---

# Handling Singular Matrices in Linear Regression

## 1. When $X^TX$ Is Not Invertible

To find the optimal weights in multivariate linear regression, we rely on the Normal Equation:

$$(X^TX)w = X^Ty$$

To solve for $w$, we typically multiply both sides by the inverse of $(X^TX)$, giving us $w = (X^TX)^{-1}X^Ty$. However, a major problem arises **if $X^TX$ is singular (non-invertible)**. In this case, $(X^TX)^{-1}$ simply does not exist, and we cannot compute $w$ using the standard formula.

The matrix $X^TX$ is singular if the feature matrix $X$ is **rank deficient**, meaning $\text{rank}(X) < D$ (where $D$ is the number of features). This occurs under the following conditions:

- **Perfect multicollinearity:** Some features are exact linear combinations of others.
    
- **Identical columns:** Two or more features are exactly the same.
    
- **More features than observations ($D > N$):** You have a highly wide dataset where the number of dimensions strictly exceeds the number of data points.
    

### Does the minimum disappear?

**No.** The core optimization problem remains exactly the same: $\min_w ||y - Xw||^2$. The loss function we are trying to minimize is still a convex shape (like a bowl), meaning a global minimum point definitely still exists.

The only difference is that **the minimum is no longer unique**. Instead of finding one single optimal weight vector $w$, we will find that there are infinitely many valid weight vectors that result in the exact same lowest possible error.

---

## 2. The Null Space and Infinitely Many Solutions

To understand why there are infinitely many solutions, we must look at the **Null Space** (or Kernel) of $X$.

The null space of $X$, denoted as $\text{Null}(X)$, is the set of all vectors $z$ that satisfy the equation $Xz = 0$.

- If a matrix is fully invertible, its null space contains only the trivial zero vector ($z = 0$).
    
- Because our matrix is singular (non-invertible), it has a **non-trivial null space**, meaning there are non-zero vectors $z$ that result in $Xz = 0$.
    

Let's see what happens to our loss function if we add a vector $z$ from the null space to our optimal weights $w$:

$$X(w + z) = Xw + Xz$$

Since $z$ is in the null space, $Xz = 0$, so:

$$X(w + z) = Xw + 0 = Xw$$

If we plug this into our loss function:

$$||y - X(w + z)||^2 = ||y - Xw||^2$$

**Conclusion:** We can add any vector from the null space to our weights, and the predictions (and thus the loss) will not change at all. The loss function is completely flat along the direction of the null space, resulting in infinitely many minimizers.

---

## 3. The Moore-Penrose Pseudoinverse

So, what do we do when we can't use the standard inverse? We use a mathematical fallback called the **Moore-Penrose Pseudoinverse**, denoted as $X^+$.

Instead of $w = (X^TX)^{-1}X^Ty$, we define our optimal weights as:

$$w^* = X^+y$$

The pseudoinverse $X^+$ acts just like a regular inverse because it mathematically satisfies four strict properties:

1. $XX^+X = X$ (Consistency on the column space)
    
2. $X^+XX^+ = X^+$ (Consistency on the row space)
    
3. $(XX^+)^T = XX^+$ (Symmetric projection onto $\text{Col}(X)$)
    
4. $(X^+X)^T = X^+X$ (Symmetric projection onto $\text{Row}(X)$)
    

### What does the Pseudoinverse actually do?

It solves the optimization problem exactly ($\min_w ||y - Xw||^2$). It is **not** an approximation; it finds an exact least-squares minimizer.

Because we established there are infinitely many solutions, **which one does the pseudoinverse choose?**

The pseudoinverse is designed to choose the solution with the **minimum norm**:

$$w^* = \text{argmin}_w ||w|| \quad \text{subject to minimizing } ||y - Xw||^2$$

**Geometric Interpretation:** Since all possible solutions differ only by vectors sitting inside the null space, the pseudoinverse specifically picks the unique solution that is purely **orthogonal to the null space** of $X$. It gives us the "shortest" possible weight vector that still completely minimizes the error.

---

## 4. Insight into the Role of the Bias Term ($w_0$)

The formula for the bias term is:

$$w_0 = \bar{y} - \sum_{j=1}^D w_j \bar{x}_j$$

**What does this physically mean?**

The bias $w_0$ compensates for the baseline difference between the target values and the input features. For example, if your input features ($x$) have very small numerical averages, but the target values ($y$) you are trying to predict are massive numbers, the weights alone might struggle to scale up. The bias completely absorbs this difference between the averages.

**Geometric Interpretation:**

Thanks to the bias term, the regression line shifts up or down exactly as much as needed. Because of this compensation, the linear regression line is mathematically guaranteed to **always pass through the exact center of mass of the data**, which is the point of the averages: $(\bar{x}_1, \dots, \bar{x}_D, \bar{y})$.

---

## 5. Residual Sum of Squares (RSS) and Model Complexity

To evaluate our model, we look at the Residual Sum of Squares (RSS), which is simply the total sum of the squared errors:

$$\text{RSS} = \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

_(Note: Mean Squared Error is just this value divided by $N$: $\text{MSE} = \frac{1}{N}\text{RSS}$)_

A smaller RSS indicates a better fit to the training data.

### Why does adding more features _always_ result in a smaller (or equal) RSS?

When you add a new feature to a linear regression model, you are giving the optimization algorithm a new "degree of freedom" (a new weight parameter, $w_{new}$, to adjust).

1. **If the new feature is completely useless:** The optimization algorithm will simply set its weight to $0$. In this worst-case scenario, the model remains exactly the same as before, and the RSS stays exactly the same.
    
2. **If the new feature has even a tiny bit of correlation with the target:** (Even if it is just pure, random noise that happens to align with the training data), the algorithm will assign it a non-zero weight to fit the training data just a little bit better.
    

Therefore, mathematically, **the RSS on the training data can never increase when you add features**. It is physically impossible. It will always either stay flat or decrease. This is why looking solely at RSS on training data is a highly deceptive way to evaluate a model—it constantly encourages you to add useless features, directly leading to severe overfitting.

---

# Model Evaluation: R-Squared ($R^2$)

To understand how well or poorly our regression model has fit the data, we need specific evaluation metrics. While Mean Squared Error (MSE) gives us an absolute measure of distance, another crucial metric is **R-squared ($R^2$)**.

R-squared represents the **proportion of variance explained** by the model. It essentially tells us what percentage of the total variation in the data our model was able to successfully capture and explain.

### The Formula

To calculate $R^2$, we look at two types of variance:

1. **Unexplained variance:** The sum of squared residuals (the errors our model made).
    
    $$\text{Unexplained variance} = \sum_{i=1}^n (\hat{y}_i - y_i)^2$$
    
2. **Total variance:** The variance of the data around a simple average line (as if we just predicted the mean of $y$ for every point).
    
    $$\text{Total variance} = \sum_{i=1}^n (y_i - \text{avg}(y))^2$$
    

The $R^2$ formula is:

$$R^2 = 1 - \frac{\text{Unexplained variance}}{\text{Total variance}}$$

### Interpreting R-squared

- **$R^2 = 1$:** The model explains 100% of the variance. It's a perfect fit.
    
- **$R^2 = 0$:** The model explains absolutely nothing. It is completely random or no better than just drawing a flat line at the average of $y$.
    
- **Can it be negative?** Yes, mathematically, $R^2$ can be negative if your model's predictions are arbitrarily worse than simply guessing the mean every time. However, in practice, a model that performs worse than the simple average is usually discarded, which is why we typically say $R^2$ falls within the range of **$[0, 1]$**.
    

**What is a "good" R-squared value?** It is hard to say definitively. A "good" value depends entirely on the domain (e.g., predicting human behavior might yield a low $R^2$, while physics experiments might yield a very high $R^2$).

**The Trap of More Features:**

Just like with the Residual Sum of Squares (RSS), adding more features to your model will mathematically _always_ result in a higher (or equal) $R^2$ on the training data. Because of this, simply looking for the highest $R^2$ is **not a reliable approach** for choosing the best model, as it encourages overfitting.

---

# Probability Theory Basics for Machine Learning

To build statistical models and quantify uncertainty in Machine Learning, we rely on Probability Theory.

## 1. The Probability Space

We begin with a formalized probability space defined by three components:

$$(\Omega, \mathcal{F}, P)$$

- **$\Omega$ (Sample Space):** The set of all possible outcomes.
    
- **$\mathcal{F}$ (Events):** The set of outcomes we are interested in measuring.
    
- **$P$ (Probability Measure):** The function that assigns a probability to those events.
    

Probability assigns numbers to _events_, but to do math easily, we use **Random Variables**, which assign numbers to _outcomes_.

A random variable $X$ is simply a function that maps the sample space to a real number:

$$X : \Omega \to \mathbb{R}$$

**Why do we need this?** They allow us to:

- Quantify uncertainty.
    
- Compute averages (Expectations).
    
- Define variance and covariance.
    
- Build statistical models.
    
- Define likelihoods in Machine Learning.
    

---

## 2. Cumulative Distribution Function (CDF)

For _any_ random variable (whether discrete or continuous), we can define a Cumulative Distribution Function (CDF). It tells us the probability that a random variable $X$ will take a value less than or equal to $x$.

$$F_X(x) = P(X \le x)$$

**Key Properties of the CDF:**

- **Non-decreasing:** It never goes down; as $x$ increases, accumulated probability only grows.
    
- **Right-continuous:** It has no left-facing gaps.
    
- **Lower Bound:** $\lim_{x \to -\infty} F(x) = 0$ (Probability cannot be less than 0).
    
- **Upper Bound:** $\lim_{x \to \infty} F(x) = 1$ (Total probability must cap at 100%).
    

### Discrete vs. Continuous via the CDF

We can easily tell if a random variable is discrete or continuous just by looking at the shape of its CDF:

- **Discrete RV:** The CDF has sudden jumps (it looks like a staircase / step-function).
    
- **Continuous RV:** The CDF is a completely smooth curve.
    
- **Mixed RV:** A combination of both smooth slopes and sudden jumps.
    

---

## 3. Discrete Random Variables

A random variable is discrete if it takes a countable number of specific values (e.g., rolling a die: 1, 2, 3, 4, 5, 6).

It is described by a **Probability Mass Function (PMF)**, which gives the exact probability of hitting a specific value:

$$p(x) = P(X = x)$$

**Rules for PMF:**

- $p(x) \ge 0$ (No negative probabilities).
    
- $\sum_x p(x) = 1$ (The sum of all probabilities must equal 1).
    

**Relation between PMF and CDF:**

The CDF is just the running total of the PMF. The height of the "jumps" in the discrete CDF graph is exactly equal to the PMF at that point.

$$F(x) = \sum_{t \le x} p(t)$$

---

## 4. Continuous Random Variables

A random variable is continuous if it can take any value within a continuous range (e.g., exactly measuring someone's height).

Instead of a PMF, we use a **Probability Density Function (PDF)**, denoted as $f(x)$. For continuous variables, the probability of hitting one _exact_ specific number is mathematically zero. Instead, we measure the probability of the variable falling within a specific _range_ $[a, b]$ by calculating the area under the PDF curve:

$$P(a \le X \le b) = \int_a^b f(x) dx$$

**Rules for PDF:**

- $f(x) \ge 0$ (Density cannot be negative).
    
- $\int_{-\infty}^\infty f(x) dx = 1$ (The total area under the entire curve must equal 1).
    

**Relation between PDF and CDF:**

The CDF is the integral (the accumulated area) of the PDF from negative infinity up to $x$:

$$F(x) = \int_{-\infty}^x f(t) dt$$

If the CDF is differentiable, the reverse is also true—the PDF is the derivative of the CDF:

$$f(x) = F'(x)$$

---

## 5. Expectation (Discrete)

The Expectation, or Expected Value $E[X]$, is simply the theoretical mean or average of the random variable.

For a discrete random variable, it is calculated as the sum of all possible values, each multiplied by its probability of occurring:

$$E[X] = \sum_x x \cdot p(x)$$

**Interpretation:** It acts as a **weighted average**, pulling the expected outcome toward the values that have the highest probability of occurring.

---

# Probability Theory: Expectation, Variance, and Distributions

## 1. Expectation (Continuous)

For a continuous random variable, we cannot simply sum up discrete probabilities. Instead, we use integration over the Probability Density Function (PDF). The expected value is the integral of the value $x$ multiplied by its density $f(x)$ over the entire range:

$$E[X] = \int_{-\infty}^{\infty} x f(x) dx$$

**Unified view:** Expectation is essentially integration with respect to probability.

---

## 2. Expectation of Functions

If we apply a function $g$ to a random variable $X$, we can find the expected value of that new function $E[g(X)]$ without needing to find the new probability distribution of $g(X)$ itself. We simply plug $g(x)$ into our standard expectation formulas.

For any function $g$:

**Discrete:**

$$E[g(X)] = \sum_x g(x)p(x)$$

**Continuous:**

$$E[g(X)] = \int_{-\infty}^{\infty} g(x)f(x) dx$$

---

## 3. Linearity of Expectation

One of the most powerful and heavily used properties in probability and machine learning is the **Linearity of Expectation**.

The expected value of a linear combination of random variables is equal to the linear combination of their expected values:

$$E[aX + bY] = aE[X] + bE[Y]$$

**Crucial Note:** This holds true regardless of whether the variables $X$ and $Y$ are independent or completely correlated. **No independence is required** for linearity of expectation to work.

---

## 4. Variance

Variance measures how much the values of a random variable $X$ are spread out around their expected value (the mean). It is defined as the expected value of the squared deviation from the mean:

$$\text{Var}(X) = E[(X - E[X])^2]$$

By expanding the square and applying the linearity of expectation, we get a much more mathematically convenient equivalent formula:

$$\text{Var}(X) = E[X^2] - (E[X])^2$$

_(The expectation of the square minus the square of the expectation)._

---

## 5. Covariance and Correlation

While variance looks at a single variable, **Covariance** measures the joint variability of two random variables—how much they change together.

$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])]$$

- Notice that the variance of a variable is simply its covariance with itself: $\text{Var}(X) = \text{Cov}(X, X)$.
    
- **If Independent:** If $X$ and $Y$ are completely independent, their covariance is $0$, which leads to the property: $E[XY] = E[X]E[Y]$.
    

### Correlation ($\rho$)

Covariance can be any number, making it hard to interpret the strength of the relationship. We normalize covariance by dividing it by the product of the standard deviations to get the **Correlation Coefficient ($\rho$)**:

$$\rho = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}}$$

The correlation coefficient strictly bounds the relationship:

$$-1 \le \rho \le 1$$

- **Positive Correlation ($\rho > 0$):** As $X$ increases, $Y$ increases.
    
- **Negative Correlation ($\rho < 0$):** As $X$ increases, $Y$ decreases.
    
- **No Correlation ($\rho \approx 0$):** No linear relationship exists between $X$ and $Y$.
    

---

## 6. Important Distributions

![[Pasted image 20260330225203.png]]

### Bernoulli Distribution

**What it is:** Models a single binary trial (a success or a failure, a coin flip).

- Probability of success (1) is $p$.
    
- Probability of failure (0) is $1 - p$.
    

$$P(X = 1) = p, \quad P(X = 0) = 1 - p$$

- **Expectation:** $E[X] = p$
    
- **Variance:** $\text{Var}(X) = p(1 - p)$
    

_(The PMF graph for a Bernoulli distribution simply shows two vertical bars at $x=0$ and $x=1$, representing the respective probabilities)._

### Normal (Gaussian) Distribution

**What it is:** Models symmetric noise and naturally emerges in many natural phenomena due to the Central Limit Theorem.

The Probability Density Function (PDF) forms the classic "bell curve":

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

- **Expectation (Mean):** $E[X] = \mu$
    
- **Variance:** $\text{Var}(X) = \sigma^2$
    

_(The PDF graph shows symmetric bell curves centered at $\mu$, with wider curves representing larger variance $\sigma^2$. The CDF graph shows smooth, S-shaped sigmoidal curves)._

> **Crucial Connection to Machine Learning (Maximum Likelihood):**
> 
> Եթե մենք ենթադրում ենք, որ մեր տվյալներում առկա աղմուկը (noise) ունի Նորմալ բաշխում, ապա Maximum Likelihood Estimation (MLE) սկզբունքով պարամետրերը օպտիմիզացնելիս, մաթեմատիկորեն ապացուցվում է, որ մեր խնդիրը բերվում է **քառակուսային սխալանքի (Mean Squared Error) մինիմիզացմանը**:
> 
_