- [[#1. When Standard Linear Models Go Wrong|1. When Standard Linear Models Go Wrong]]
- [[#2. Generalized Linear Regression Model|2. Generalized Linear Regression Model]]
- [[#3. Least Squares vs. Maximum Likelihood|3. Least Squares vs. Maximum Likelihood]]
- [[#4. The Likelihood Function and the Logarithm Trick|4. The Likelihood Function and the Logarithm Trick]]
- [[#5. The Mathematical Equivalence|5. The Mathematical Equivalence]]
- [[#6. The Design Matrix ($\mathbf{\Phi}$)|6. The Design Matrix ($\mathbf{\Phi}$)]]
- [[#1. Insight into the Role of the Bias Term ($w_0$)|1. Insight into the Role of the Bias Term ($w_0$)]]
- [[#2. Noise Variance and Precision ($\beta$)|2. Noise Variance and Precision ($\beta$)]]
- [[#3. Shifting the Problem Formulation|3. Shifting the Problem Formulation]]
- [[#4. The Polynomial Regression Model|4. The Polynomial Regression Model]]
- [[#5. The Design Matrix (Vandermonde Matrix)|5. The Design Matrix (Vandermonde Matrix)]]
- [[#1. The Design Matrix (Vandermonde Matrix)|1. The Design Matrix (Vandermonde Matrix)]]
- [[#2. The Interpolation Theorem|2. The Interpolation Theorem]]
- [[#3. The Problem of Ill-Conditioning|3. The Problem of Ill-Conditioning]]
- [[#4. The Condition Number $\kappa(A)$|4. The Condition Number $\kappa(A)$]]
- [[#1. Why the Condition Number Matters|1. Why the Condition Number Matters]]
- [[#2. Ill-Conditioning vs. Overfitting|2. Ill-Conditioning vs. Overfitting]]
- [[#1. The Overfitting Problem|1. The Overfitting Problem]]
- [[#2. Real-World Example: Hardwood Data|2. Real-World Example: Hardwood Data]]
- [[#3. Theoretical Connection: Polynomial Regression vs. Taylor Series|3. Theoretical Connection: Polynomial Regression vs. Taylor Series]]
- [[#1. Choosing the Right Polynomial Degree ($n$)|1. Choosing the Right Polynomial Degree ($n$)]]
- [[#2. Crucial Limitations of Polynomial Models|2. Crucial Limitations of Polynomial Models]]
- [[#3. Runge's Phenomenon|3. Runge's Phenomenon]]
- [[#1. Hierarchical Models|1. Hierarchical Models]]
- [[#2. Introduction to Regularization (Weight Decay)|2. Introduction to Regularization (Weight Decay)]]

## 1. When Standard Linear Models Go Wrong

Standard linear regression assumes a strictly straight-line relationship between features and the target. As the first slide illustrates, this assumption can fail depending on the true underlying data distribution:
![[Pasted image 20260401011012.png]]
* **Area vs. House Price:** This relationship is largely *monotonic* (as area increases, price increases) and can be *approximately linear*. A standard linear model works reasonably well here.
* **Body Temperature vs. Probability of Death:** This forms a U-shaped curve. It is *neither monotonic nor linear*. A straight line would completely fail to model the extremes (both freezing and a high fever increase the risk of death). Feature engineering is required here, such as calculating the absolute deviation from the normal temperature: $\Delta T = \text{abs}(T - T_{\text{normal}})$.
* **Pixel Value vs. Probability of "cat":** This is highly chaotic and noisy. It is neither monotonic nor linear. Simple feature engineering cannot easily fix this; it requires highly complex non-linear models.

---

## 2. Generalized Linear Regression Model

To handle non-linear data without abandoning the mathematical elegance of linear regression, we introduce a powerful concept: **Basis Functions**.

We want to generalize linear regression while still preserving a linear element. To do this, we take our $D$-dimensional vector from the original input space ($\mathbf{x}$), transition it into a new $M$-dimensional feature space ($\boldsymbol{\phi}$), and perform linear regression *there*. 

Essentially, we are introducing non-linearity into the space itself, rather than the function. This gives us a model that is non-linear with respect to the original input variables, but strictly linear with respect to the weights ($\mathbf{w}$). *Note: This concept is mathematically very similar to the Kernel Trick used in Support Vector Machines.*
An equation is considered linear with respect to a specific variable if that variable only ever appears to the power of 1. It cannot be squared, cubed, put inside a square root, or multiplied by another instance of itself.
When we say "map it to X-space," we are talking about bringing our model back down to the single dimension of our original dataset: the regular $x$-axis.

In our multi-dimensional feature space ($Z$-space), the equation $y = w_0 + w_1z_1 + w_2z_2 + w_3z_3$ creates a perfectly flat, stiff, multi-dimensional plane. But because our $Z$ variables are tied strictly to $x$ ($z_1=x$, $z_2=x^2$, $z_3=x^3$), we are forcing that flat plane to intersect with the natural curves of those math functions.

When we project that intersection back down to our standard 2D graph ($x$ vs $y$), the "flat" plane appears to us as a bending, twisting curve!
Let's break down the polynomial regression model:
$$y = w_0 + w_1x + w_2x^2 + w_3x^3$$
<iframe src="file:///home/blueberry/Picsart-Academy/ML/Obsidian Vault/polynomial_viz.html" width="100%" height="600" frameborder="0"></iframe>
<iframe src="file:///home/blueberry/Picsart-Academy/ML/Obsidian Vault/zspace_viz.html" width="100%" height="750" frameborder="0"></iframe>

Instead of $y(\mathbf{x}, \mathbf{w}) = w_0 + w_1x_1 + \dots + w_Dx_D$, we apply fixed non-linear basis functions $\phi_j(\mathbf{x})$:
$$y(\mathbf{x}, \mathbf{w}) = \sum_{j=0}^{M-1} w_j\phi_j(\mathbf{x}) = \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x})$$

* $\mathbf{w} = (w_0, \dots, w_{M-1})^T$ is our weight vector.
* $\boldsymbol{\phi} = (\phi_0, \dots, \phi_{M-1})^T$ is our basis function vector. The function $\boldsymbol{\phi}$ takes an input vector and returns a newly transformed vector.
* We introduce a dummy function $\phi_0(\mathbf{x}) = 1$ specifically to act as the multiplier for the bias parameter $w_0$.

### Common Basis Functions
What can we use as our basis function?
1.  **Polynomial:** $\phi_j(x) = x^j$. (This is the most commonly used basis function, creating Polynomial Regression).
2.  **Gaussian:** $\phi_j(x) = \exp\{-\frac{(x-\mu_j)^2}{2s^2}\}$
3.  **Sigmoidal:** $\phi_j(x) = \sigma(\frac{x-\mu_j}{s})$

---

## 3. Least Squares vs. Maximum Likelihood

But how do we find these optimal $\mathbf{w}$ weights? 
In previous lectures, we used the **Least Squares method**—calculating geometric distances (errors), taking the derivative, and setting it to zero. 

**The core question:** Is minimizing this distance exactly the same as assuming all data points came from a Gaussian distribution, and trying to find a line that perfectly represents the *means* of those Gaussians? 
We are fundamentally changing the formulation of the regression problem: instead of finding the line that minimizes *error*, we want to find the line that maximizes the *probability* (Likelihood) of observing our specific data points. 

Let's prove this equivalence. Assume our target variable $t$ is generated by our deterministic function $y(\mathbf{x}, \mathbf{w})$ plus some random Gaussian noise $\epsilon$:
$$t = y(\mathbf{x}, \mathbf{w}) + \epsilon$$

Since $\epsilon$ is a zero-mean Gaussian random variable with precision $\beta$ (where precision is the inverse of variance, $\beta = 1/\sigma^2$), the conditional probability distribution of $t$ is a Gaussian centered exactly on our regression line:
$$p(t | \mathbf{x}, \mathbf{w}, \beta) = \mathcal{N}(t | \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}), \beta^{-1})$$
To visualize this equation, you have to shift how you think about regression lines.

Normally, we think of a regression line as a rigid stick that predicts one exact target value ($t$) for every input ($x$). But in the Maximum Likelihood/Probabilistic view, the regression line isn't a rigid predictor—it is just the **center of a probability cloud**.

Here is how to break down the visual components of that equation:

1. **The Mean ($\mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})$):** This is your standard regression line (or curve). It acts as the "anchor" or the peak of the probability distribution.
    
2. **The Noise ($\epsilon$):** Because real-world data is messy, we assume the actual target $t$ will bounce up or down away from that anchor line.
    
3. **The Precision ($\beta^{-1}$):** This is the variance ($\sigma^2$). It dictates how "fuzzy" or "thick" the probability cloud is around the line. If $\beta$ is very high (high precision), the variance is tiny, and the data hugs the line tightly. If $\beta$ is low, the data spreads out wildly.
    

### The Mental Image

Imagine slicing the 2D graph vertically at a specific $x$-coordinate. Instead of finding a single dot on the line, you find a **vertical Gaussian bell curve** sitting exactly on top of the line. The highest probability (the peak of the bell) is directly on the regression line, and the probability fades away as you move vertically up or down.

<iframe src="file:///home/blueberry/Picsart-Academy/ML/Obsidian Vault/probability_viz.html" width="100%" height="750" frameborder="0"></iframe>


### Why is this view so powerful?

By treating the prediction as a probability distribution rather than a single point, the model isn't just giving you an answer—it is giving you **confidence bounds**.

When a machine learning model says "The house price will be $300k," that's moderately helpful. But when the probabilistic model says "The expected mean is $300$k, but based on the precision ($\beta$), there is a 95% chance it falls between $280k and $320k," that is incredibly valuable for real-world decision-making!

---

## 4. The Likelihood Function and the Logarithm Trick

**Likelihood** is a form of conditional probability, closely related to Bayes' Theorem. It measures how likely it is to observe our specific data given a set of parameters.

For the model to be optimal, the probabilities for all observed points must be maximally large. Assuming all $N$ data points are drawn independently, the joint probability (the Likelihood function) is the *product* of all individual probabilities:
$$p(\mathbf{t} | \mathbf{X}, \mathbf{w}, \beta) = \prod_{n=1}^N \mathcal{N}(t_n | \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n), \beta^{-1})$$

We must find the Gaussian parameters that make this product as large as possible. However, taking the derivative of a massive product ($\prod$) is mathematically exhausting. 

**The Log Trick:** If we know we have to differentiate a complex product, we take its natural logarithm. Because the logarithm is a strictly increasing (monotonic) function, the parameters that maximize the log-likelihood will perfectly match the parameters that maximize the raw likelihood. The logarithm magically turns the product into a sum ($\sum$) and cancels out the exponential $e$ inside the Gaussian formula, making optimization vastly easier:

$$\ln p(\mathbf{t} | \mathbf{w}, \beta) = \sum_{n=1}^N \ln \mathcal{N}(t_n | \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n), \beta^{-1})$$
$$= \frac{N}{2}\ln \beta - \frac{N}{2}\ln(2\pi) - \beta E_D(\mathbf{w})$$

Where $E_D(\mathbf{w})$ is our familiar sum-of-squares error function:
$$E_D(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^N \{t_n - \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)\}^2$$

---

## 5. The Mathematical Equivalence

The slides mathematically prove that **Maximum Likelihood corresponds exactly to the Least Squares error.** Look at the final log-likelihood equation above. The only term that actually contains the weights ($\mathbf{w}$) is $-\beta E_D(\mathbf{w})$. Because this term is *negative*, maximizing the overall likelihood is mathematically identical to **minimizing** the sum-of-squares error $E_D(\mathbf{w})$. 

*This is exactly why we use squared errors instead of linear/absolute errors in regression—the squared term is naturally embedded inside the exponent of the Normal (Gaussian) distribution's error function!*

Once this is established, we take the gradient (derivative) of the log-likelihood, set it to zero, and solve for $\mathbf{w}$ to obtain the generalized normal equation:
$$\mathbf{w}_{ML} = (\mathbf{\Phi}^T \mathbf{\Phi})^{-1} \mathbf{\Phi}^T \mathbf{t}$$

---

## 6. The Design Matrix ($\mathbf{\Phi}$)

In the generalized normal equation, $\mathbf{\Phi}$ is an $N \times M$ matrix called the **Design Matrix**. 
$$
\mathbf{\Phi} = 
\begin{pmatrix}
\phi_0(\mathbf{x}_1) & \phi_1(\mathbf{x}_1) & \dots & \phi_{M-1}(\mathbf{x}_1) \\
\phi_0(\mathbf{x}_2) & \phi_1(\mathbf{x}_2) & \dots & \phi_{M-1}(\mathbf{x}_2) \\
\vdots & \vdots & \ddots & \vdots \\
\phi_0(\mathbf{x}_N) & \phi_1(\mathbf{x}_N) & \dots & \phi_{M-1}(\mathbf{x}_N)
\end{pmatrix}
$$

**How to read this matrix:**
If we choose $M$ basis vectors (the $\phi$ functions) and we have $N$ total data points ($x$), then every individual element $\Phi_{nj} = \phi_j(\mathbf{x}_n)$ shows the value of the $j$-th basis vector evaluated at the $n$-th data point.
* **Rows:** Represent individual data points.
* **Columns:** Represent the different basis functions applied to the data.

Just like in standard linear regression, if the term $(\mathbf{\Phi}^T \mathbf{\Phi})$ is singular (an inverse does not exist), we bypass the issue by using the **Moore-Penrose pseudo-inverse**, denoted as $\mathbf{\Phi}^\dagger$.

---

# Deep Dive: Bias, Precision, and Polynomial Regression

## 1. Insight into the Role of the Bias Term ($w_0$)

Let's make the bias parameter ($w_0$) explicit in our sum-of-squares error function to see exactly what it does:

$$E_D(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N \left\{ t_n - w_0 - \sum_{j=1}^{M-1} w_j \phi_j(\mathbf{x}_n) \right\}^2$$

If we take the derivative of this error function with respect to $w_0$, set it to zero, and solve for $w_0$, we obtain the following definitions:

- **Average of targets:** $\bar{t} = \frac{1}{N} \sum_{n=1}^N t_n$
    
- **Average of basis functions:** $\bar{\phi}_j = \frac{1}{N} \sum_{n=1}^N \phi_j(\mathbf{x}_n)$
    
- **The Bias Formula:** $w_0 = \bar{t} - \sum_{j=1}^{M-1} w_j \bar{\phi}_j$
    

**What this means practically:**

The bias represents the difference between the averages of our targets and the weighted averages of our features. Roughly speaking, it shifts our regression line up and down to ensure that it naturally compensates for this difference. Because of this adjustment, the regression line is mathematically guaranteed to pass exactly through the center point of the averages: $(\bar{\phi}, \bar{t})$.

---

## 2. Noise Variance and Precision ($\beta$)

We can also maximize the log-likelihood function with respect to the noise precision parameter $\beta$. When we do this, we get:

$$\frac{1}{\beta_{ML}} = \frac{1}{N} \sum_{n=1}^N \{t_n - \mathbf{w}_{ML}^T \boldsymbol{\phi}(\mathbf{x}_n)\}^2$$

**What this means practically:**

The squared deviation of the noise represents the actual variance of the noise. Here, we clearly see that the inverse of the maximum-likelihood noise precision ($1/\beta_{ML}$) is exactly equal to the residual variance (the mean squared error) of the target values around our fitted regression function.

---

## 3. Shifting the Problem Formulation

Let's change the problem formulation. Suppose we have $N$ data points (observations $\{x_n\}$), and each one has a corresponding target value $\{t_n\}$. Our ultimate goal is to predict the value of $t$ for any new $x$.

If we plot this data and notice it is not a straight line, we realize that a simple linear connection is extremely unlikely in the real world. However, we still want to apply linear mathematical methods. How do we resolve this?

We assume that $t$ can be modeled as a nonlinear but smooth function of $x$—specifically, a polynomial. By doing this, we can easily capture the curved relationships in the data.

---

## 4. The Polynomial Regression Model

How do we move beyond linearity? The main idea of polynomial regression is to introduce completely new features based on the original input $x$. Instead of just giving the model $x$, we add $x^2, x^3$, and so on, as entirely new variables.

Let's denote the $m$-th transformation of $X$ as $h_m(X)$ and assign it a weight variable $\beta_m$. Thus, our model function becomes:

$$f(X) = \sum_{m=1}^M \beta_m h_m(X)$$

Here, $f(X)$ is a linear basis expansion in the basis of $X$. **The magic of this trick:** Once we define and calculate these basis functions ($x^2, x^3$, etc.), the model treats them as standard, independent variables. The model is completely _linear_ with respect to these new variables, and the fitting algorithm proceeds exactly the same way as standard linear regression!

---

## 5. The Design Matrix (Vandermonde Matrix)

If we specifically choose our basis functions to be standard polynomials ($1, x, x^2, x^3 \dots$), then our Design Matrix $\mathbf{\Phi}$ takes on a very famous mathematical form known as the **Vandermonde matrix**:

$$V = V(x_0, x_1, \dots, x_m) = \begin{pmatrix} 1 & x_0 & x_0^2 & \dots & x_0^n \\ 1 & x_1 & x_1^2 & \dots & x_1^n \\ 1 & x_2 & x_2^2 & \dots & x_2^n \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_m & x_m^2 & \dots & x_m^n \end{pmatrix}$$

Using this Vandermonde matrix in our normal equation, we can find the absolute best set of $\beta$ weights that will result in the smallest possible squared deviation from the matrix.

### The Interpolation Theorem

A powerful (and sometimes dangerous) mathematical fact about polynomials:

If we have $N$ distinct points $\{x_i\}$, there always exists a polynomial of degree at most $N-1$ such that:

$$\hat{y}(x_i) = y_i \quad \forall i$$

This means if you have 10 data points, a 9th-degree polynomial will pass _exactly_ through every single point. It will memorize the data perfectly with zero training error (though this usually results in catastrophic overfitting for any new predictions!).


# Polynomial Regression and Matrix Stability

## 1. The Design Matrix (Vandermonde Matrix)

In polynomial regression, we transform our 1D input $x$ into a multidimensional feature space by taking its powers. The design matrix $\mathbf{\Phi}$ (often denoted as $V$ for Vandermonde) is constructed as follows:

$$V = V(x_0, x_1, \dots, x_m) = \begin{pmatrix} 1 & x_0 & x_0^2 & \dots & x_0^n \\ 1 & x_1 & x_1^2 & \dots & x_1^n \\ 1 & x_2 & x_2^2 & \dots & x_2^n \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_m & x_m^2 & \dots & x_m^n \end{pmatrix}$$

**How it works:** For every individual data point (each row), we take its value $x$, calculate its first $n$ powers, and place them side-by-side as columns. Using this specific design matrix, we can use the Normal Equation to find the exact $\beta$ weights that yield the minimum possible squared deviation.

---

## 2. The Interpolation Theorem

**Theorem:** _If we have $N$ distinct points $\{x_i\}$, there exists a polynomial of degree at most $N - 1$ such that $\hat{y}(x_i) = y_i \quad \forall i$._

This theorem directly stems from the following logic shown. If we want to find a polynomial curve that perfectly passes through our points, we can set up a system of linear equations (we solve it for C-s and not x-s):

$$\hat{y}_1 = C_0 + C_1x_1 + C_2x_1^2 + \dots + C_{n-1}x_1^{n-1} = y_1$$

$$\hat{y}_2 = C_0 + C_1x_2 + C_2x_2^2 + \dots + C_{n-1}x_2^{n-1} = y_2$$

$$\dots$$

Because all of our input $x$ values are distinct (different from one another), the columns of our resulting system are **linearly independent**.

If we use a polynomial of degree $N-1$ for $N$ points, we get a perfectly square $N \times N$ matrix. Because its columns are linearly independent, the matrix is fully **invertible**. Therefore, we can find one single, unique solution for the coefficients $C$.

Because the line hits every single target point exactly, the minimum value of our error function is exactly **0**. _(A corollary to this mathematical property is that an $n$-th degree polynomial has exactly $n$ roots)._
This means if you have 10 data points, a 9th-degree polynomial will pass _exactly_ through every single point. It will memorize the data perfectly with zero training error (though this usually results in catastrophic overfitting for any new predictions!).

---

## 3. The Problem of Ill-Conditioning

While finding a zero-error polynomial sounds great in theory, it is mathematically disastrous in practice.

Imagine our input features ($x$) are normalized to fall between $0$ and $1$ (i.e., $x \in [0, 1]$). As we take higher and higher powers of these fractions, the numbers get vanishingly small and incredibly close to one another:

$$x^5 \approx x^4 \approx x^3$$

Because these higher-power columns become almost identical, they become **nearly linearly dependent**. Consequently, the matrix product $\mathbf{\Phi}^T\mathbf{\Phi}$ becomes almost singular—its determinant approaches zero, meaning it is practically non-invertible.

**Why is this bad?** We do not want our weights to make massive, unpredictable jumps just because of tiny, microscopic changes in the input data. When a matrix is nearly singular, calculating its inverse becomes numerically unstable. The computer's rounding errors explode, and the resulting polynomial curve will violently oscillate up and down. This numerical instability is known as **Ill-Conditioning**.

---

## 4. The Condition Number $\kappa(A)$

To measure exactly how unstable (ill-conditioned) a matrix is, we define the **Condition Number**, denoted as $\kappa(A)$.

$$\kappa(A) = \frac{\sigma_{\text{max}}(A)}{\sigma_{\text{min}}(A)}$$

Where:

- $\sigma_{\text{max}} =$ the largest singular value.
    
- $\sigma_{\text{min}} =$ the smallest singular value.
    
    _(Note: The singular values of a matrix $A$ are defined as the non-negative square roots of the eigenvalues of $A^TA$ or $AA^T$)._
    

**Interpretation:**

- $\kappa(A) \approx 1 \rightarrow$ **Well-conditioned:** The matrix is stable, and inverses are reliable.
    
- Large $\kappa(A) \rightarrow$ **Ill-conditioned:** The matrix is highly sensitive to noise.
    
- If $\sigma_{\text{min}} \rightarrow 0$, then $\kappa(A) \rightarrow \infty$ (The matrix is completely singular/non-invertible).
    

### Geometric Intuition: Why do we need it?

To intuitively understand what the Condition Number represents, we have to look at it geometrically.

Any matrix $A$ represents a geometric transformation. Imagine a perfect, round 2D circle made of data points. If we multiply that circle by a matrix $A$, it will stretch and squash into an **ellipse**.

- The **largest singular value ($\sigma_{\text{max}}$)** is the length of the ellipse's longest axis (how much the matrix stretched the space).
    
- The **smallest singular value ($\sigma_{\text{min}}$)** is the length of the ellipse's shortest axis (how much the matrix squashed the space).
    

The condition number is the ratio between these two. If the condition number is massive, it means the circle was squashed flat into a pancake. Reversing that process (inverting the matrix) means trying to perfectly inflate a microscopic pancake back into a perfect circle. If there is even $0.0001\%$ noise in the data, the "inflation" process will magnify that noise infinitely, completely destroying your regression weights.

This visualization dynamically builds a Vandermonde matrix, calculates the inverse, and shows exactly how the weights explode into massive numbers as you increase the polynomial degree and introduce a tiny bit of noise.

<iframe src="file:///home/blueberry/Picsart-Academy/ML/Obsidian Vault/polynomial_crash_viz.html" width="100%" height="750" frameborder="0"></iframe>

**Geometric Interpretation of Condition Number**
This visualization shows how a matrix physically stretches and squashes space, and calculates the exact Singular Values ($\sigma$) and Condition Number ($\kappa$).

<iframe src="file:///home/blueberry/Picsart-Academy/ML/Obsidian Vault/condition_number_viz.html" width="100%" height="750" frameborder="0"></iframe>

---

# Polynomial Regression: The Condition Number $\kappa$

## 1. Why the Condition Number Matters

In Polynomial Regression, our ultimate goal is to solve the Normal Equation for the optimal weights $w$:

$$(\Phi^T \Phi)w = \Phi^T y$$

To isolate and solve for $w$, our very first mathematical assumption is that we can simply multiply both sides by the inverse of $(\Phi^T \Phi)$. But as we've learned, we cannot always do this cleanly.

**The Role of Kappa ($\kappa$):**

The condition number $\kappa$ allows us to mathematically quantify exactly how close (or how far) our matrix is from being completely singular (non-invertible).

If the condition number $\kappa(\Phi^T \Phi)$ is very large, the matrix is "ill-conditioned." This means the mathematical foundation of our model is incredibly fragile. Under these conditions, the following triggers:

- A tiny amount of noise in the target $y$
    
- Small floating-point rounding errors inside the computer's memory
    
- A microscopic perturbation in the input data
    

...will result in a **huge, catastrophic change in the calculated weights ($w$)**.

### The Problem with Instability

When building a machine learning model, predictability and stability are paramount. If we have two data points that are very similar to each other, we absolutely want our model to classify or predict them in a very similar way.

If the model is unstable due to a large condition number, those slightly different inputs will result in wildly different weight calculations and entirely different predictions. Therefore, we must pay close attention to $\kappa(\Phi^T \Phi)$ to ensure we have a "well-conditioned" matrix before blindly trusting the calculated weights.

_(Fortunately, modern machine learning libraries like `scikit-learn` handle this intelligently. If you pass an ill-conditioned matrix to `scikit-learn`, it will throw a warning alerting you that the features are highly dependent and the standard mathematical solution is unstable)._

---

## 2. Ill-Conditioning vs. Overfitting

It is crucial to understand that **Ill-Conditioning is NOT Overfitting.**

- **Ill-Conditioning** is a _numerical stability issue_. It is a problem with how computers handle the physics of floating-point math when dealing with nearly-parallel vectors. The wild oscillations happen because of calculation errors and the amplification of noise.
    
- **Overfitting** is a _statistical learning issue_. It happens when a model is too complex and perfectly memorizes the training data (including its noise), resulting in terrible generalization to unseen data.
    

**The Intersection:**

While they are fundamentally different problems, **they almost always happen at the exact same time when using high-degree polynomials.**

### Why does this happen? (The Role of the Hyperparameter $n$)

Let's think about what controls this problem. We are given the data points ($x$ and $y$); we cannot change them. The optimization algorithm calculates the coefficients ($C$ or $w$).

So what is the one thing _we_ control? **The degree of the polynomial ($n$).** The polynomial degree $n$ is a **hyperparameter** that we must actively choose. Even if you only have a single input feature $x$, you can choose to expand it into 13 features ($x, x^2, x^3 \dots x^{13}$).

When we choose a very large $n$:

1. **It causes Ill-Conditioning:** The higher powers ($x^{12}, x^{13}$) become virtually identical, making the matrix columns linearly dependent and spiking the condition number.
    
2. **It causes Overfitting:** As per the Interpolation Theorem, a high-degree polynomial has enough flexibility to pass perfectly through every single training point. It learns the noise perfectly, resulting in massive weight values and terrible generalization to new data.
    

### The Solution

Because both of these severe issues are triggered by the exact same hyperparameter, the primary solution is straightforward: **Avoid using excessively high-degree polynomials.** Ultimately, we must remember that Polynomial Regression is still just standard Linear Regression. We are simply performing that linear regression in a newly manufactured, higher-dimensional space of polynomial features rather than the original input space.

---

# Polynomial Regression: Illustrations and Challenges

## 1. The Overfitting Problem

As discussed previously, while ill-conditioning is a problem of numerical stability, simply fitting high-degree polynomials to any given dataset creates a massive statistical problem: **Overfitting**.

This stems from the Interpolation Theorem. If all our input $x$ values are distinct, we can solve the system of linear equations with arbitrary precision. By simply adding more and more polynomial features ($x^2, x^3, \dots, x^{n}$), we constantly expand the column space of our design matrix. Because we are projecting our target $y$ onto an ever-expanding column space, the residual error will artificially shrink closer and closer to zero.
![[Pasted image 20260401181315.png]]
### The Behavior of High-Degree Polynomials

Higher-degree polynomials have large, highly sensitive coefficients. If we force a high-degree polynomial (like $n=9$) to fit a small set of data points, it will pass through those points perfectly. However, to achieve this, the curve must take wildly exaggerated paths between the data points.
![[Pasted image 20260401180256.png]]
If we look at a 9th-degree polynomial fit, it hits the training points well. But between the points, the curve shoots off into completely arbitrary, extreme values. This is a terrible model. If we were to receive a new data point right in the middle of two training points, the model would predict a massive, nonsensical value instead of a logical interpolation. This extreme oscillation between points is the hallmark of overfitting in polynomial regression.

---

## 2. Real-World Example: Hardwood Data

Let's look at a concrete example choosing the basis function $h_m(X) = X^m$. We are expanding the simple model:

$$f(X) = \beta_0 + \beta_1x + \beta_2x^2 + \dots + \beta_{M-1}x^{M-1}$$

We have 19 observations concerning the strength of kraft paper ($y$) and the percentage of hardwood ($x$) used in the pulp. The goal is to predict paper strength based on the hardwood percentage.

### Analyzing the Fits
![[Pasted image 20260401180346.png]]
If we look at the raw scatterplot, we see a clear non-monotonic trend. Around 10% hardwood, the paper strength peaks. If we add more hardwood, the strength decreases. If we use too little, it is also weak.

We tested four models and evaluated them using $R^2$:

- **Linear Model ($m=1$):** A straight line cannot capture this change in direction. It simply predicts that strength constantly increases. As expected, its $R^2$ is very poor (0.3054).
    
- **Quadratic Model ($m=2$):** A parabola catches the general U-shape and the change in monotonicity. However, because the data is slightly skewed (more points bunched up on one side), the symmetric parabola misses the true peak and falls off incorrectly. The $R^2$ jumps significantly to 0.9085, but the visual fit is still slightly off.
    
- **Cubic Model ($m=3$):** A cubic curve provides an almost ideal fit to this specific dataset. It captures the skewness and passes beautifully through the core trend. The $R^2$ is an excellent 0.9707.
    

### What if we kept going? (Interactive Demonstration)

If we were to add a 4th, 5th, or 10th-degree polynomial, the $R^2$ on this training set would technically keep getting closer to 1.0. But visually, the curve would start vibrating wildly to hit every single dot.

_Try this interactive widget to see exactly what happens when you push the polynomial degree too high on a similar dataset. Notice how the training error drops, but the curve becomes entirely useless for predicting anything between the dots!_
<iframe src="file:///home/blueberry/Picsart-Academy/ML/Obsidian Vault/polynomial_overfitting_viz.html" width="100%" height="750" frameborder="0"></iframe>

Cross-referencing internal data

If we performed a standard Train/Test split, the Cubic model ($m=3$) would likely yield the best test score, while the 10th-degree model would fail catastrophically on the test data due to this exact overfitting.

---

## 3. Theoretical Connection: Polynomial Regression vs. Taylor Series

A great theoretical question arises: _Can't any function theoretically be represented as an infinite polynomial series?_

Yes, this brings us to the concept of **Taylor Series**. A Taylor series allows us to represent complex functions (like $e^x$ or $\sin(x)$) as an infinite sum of polynomial terms.

So why does polynomial regression fail and oscillate wildly, while a Taylor series converges beautifully?

**The critical difference is Infinity vs. Finiteness:**

- **Taylor Series (Infinite):** In a converging Taylor Series, the coefficients (derived from the exact mathematical derivatives) naturally approach zero as the polynomial degree approaches infinity. The higher-order terms gradually fade out, allowing the infinite sum to converge smoothly and perfectly to the true underlying function.
    
- **Polynomial Regression (Finite):** In machine learning, we are strictly bound by a finite number of data points. Because we only have a finite set of points, we cannot let the series run to infinity to naturally zero out the coefficients. Instead, the optimization algorithm is forced to assign massive, finite numbers to those higher-order coefficients just to forcefully bend the curve through the few discrete points we gave it.
    

If we theoretically had an infinite amount of data points everywhere along the curve, we could fit an infinite-degree polynomial regression, and it would perfectly discover the underlying Taylor Series of the true function. But because we only ever have a finite dataset, we are blocked from the mathematical beauty of the Taylor Series. We are stuck with volatile, finite polynomials, making the careful selection of the hyperparameter $n$ (the degree) absolutely critical to avoid overfitting.

---

# Advanced Polynomial Regression: Model Selection and Runge's Phenomenon

## 1. Choosing the Right Polynomial Degree ($n$)

We know that if we have $N$ data points, a polynomial of degree $N-1$ will perfectly memorize the training data (producing a $0$ training error). However, as we established, this model will be catastrophically overfitted and will perform terribly on a test set.

Therefore, we must purposefully select a degree $n$ that balances learning the trend with generalizing to new data. There are two primary strategies for finding this optimal degree:

1. **Forward Selection (Bottom-Up):** Start with the simplest model (a linear model, $n=1$). Gradually add higher-degree components (quadratic, then cubic, etc.). Stop increasing the degree when the loss (error) on a validation set stops improving significantly.
    
2. **Backward Elimination (Top-Down):** Start with a high-degree polynomial (e.g., $N-1$). Gradually remove the highest-order terms one by one. Stop removing terms when the $t$-statistic indicates that the removed term had a significant impact (i.e., its weight was not close to $0$, meaning taking it away severely hurts the model).
    

_(Note: Forward selection is generally the safer and more common approach in practical machine learning)._

---

## 2. Crucial Limitations of Polynomial Models

Before deploying polynomial regression, there are two major limitations we must understand: **Extrapolation** and **Hierarchy**.

### A. The Danger of Extrapolation

Polynomials exhibit incredibly extreme and unpredictable behavior outside the specific range of data they were trained on.

**Example:** Imagine fitting a downward-facing parabola (a negative quadratic function) to a cluster of data points. Within that specific domain, the curve models the trend perfectly.

However, if you try to **extrapolate**—meaning you try to predict the value for a new $x$ that is far to the right of your training data—the parabola will continue plunging rapidly toward negative infinity. In reality, the true trend of the data might have leveled off or gone back up.

**Rule of Thumb:** Never use polynomial models for extrapolation. You should only trust their predictions _within the exact range_ (the specific domain) of the data they were trained on.

### B. The Rule of Hierarchy

A polynomial model is considered **hierarchical** if it includes _all_ lower-degree terms up to the chosen degree $n$.

For example, a valid hierarchical cubic model is: $y = \beta_0 + \beta_1x + \beta_2x^2 + \beta_3x^3$.

If a model is _not_ hierarchical (e.g., you skip the linear and quadratic terms and just use $y = \beta_0 + \beta_3x^3$), the model will lose the mathematical property of **invariance to linear transformations**. This means if you simply shifted your data (e.g., measuring temperature in Kelvin instead of Celsius by adding $273.15$ to $x$), the non-hierarchical model would completely break and yield entirely different predictions. To keep the math stable, always include all lower-degree terms.

---

## 3. Runge's Phenomenon

Even if you follow all the rules, fitting high-degree polynomials to a set of equally spaced data points leads to a specific, mathematically guaranteed disaster known as **Runge's Phenomenon**.

### What happens?

When you try to force a high-degree polynomial to pass perfectly through all training points (exact interpolation):

1. The system solves for an exact fit, resulting in zero training error.
    
2. To achieve this, the mathematical constraints force the coefficients to become incredibly large.
    
3. These massive coefficients must alternate in sign (positive, negative, positive) to forcefully bend the curve back and forth through the data points.
    
4. **The Result:** The function begins to fluctuate dramatically. While it might look relatively stable in the center of the dataset, these oscillations are massively amplified at the **boundary regions** (the far left and right edges of the data).
    

### Why does this happen? (The Geometric & Global View)

The root cause is that polynomial basis functions ($x^2, x^3, \dots, x^p$) are **global**.

Unlike a localized function, changing just one coefficient in a polynomial equation alters the shape of the curve across the _entire_ domain from $-\infty$ to $\infty$. There is no "locality."

If the optimization algorithm tweaks a weight to make the curve hit a data point cleanly in the middle of the graph, that exact same tweak will cause the tail ends of the curve (at the boundaries) to whip violently up or down. Geometrically, as the degree increases, the column space of the design matrix ($\Phi$) enlarges so much that the projection fits the data exactly, but the space _between_ the points is left entirely unconstrained and fluctuates wildly.

### Visual Example: The Runge Function

Look at the classic demonstration of Runge's Phenomenon in your final slide:

- **The Green Dotted Line:** The true underlying function is a simple, smooth bell curve: $f(x) = \frac{1}{1 + 25x^2}$.
    
- **The Blue Dots:** We sample specific, equally spaced data points from that smooth curve.
    
- **The Orange Line:** We fit a high-degree interpolating polynomial through those exact points.
    

Even though the orange line successfully hits every single blue dot, it fails miserably at modeling the true function. Near the edges (around $x = -1.0$ and $x = 1.0$), the orange polynomial shoots up to massive, nonsensical values, completely diverging from the true green line.

---
### See it in Action: Runge's Phenomenon Explorer

_Use this interactive widget to physically experience Runge's Phenomenon. Notice how increasing the polynomial degree initially helps fit the center, but quickly causes the edges (boundaries) to whip out of control!_

<iframe src="file:///home/blueberry/Picsart-Academy/ML/Obsidian Vault/runge_phenomenon_viz.html" width="100%" height="750" frameborder="0"></iframe>

---

# Hierarchical Models and Regularization

## 1. Hierarchical Models

When building polynomial regression models, maintaining a strict mathematical structure is essential for stability. This structure is known as **Hierarchy**.

A polynomial model is considered **hierarchical** if, when a term of degree $p$ is included, absolutely all lower-degree terms are also included in the equation.

**Examples:**

- **Valid (Hierarchical):** $f(x) = w_0 + w_1x + w_2x^2 + w_3x^3$
    
    _(This is valid because the highest degree is 3, and degrees 2, 1, and 0 are all present)._
    
- **Not Hierarchical:** $f(x) = w_0 + w_3x^3$
    
    _(This is invalid because it is missing the first-order $x$ and second-order $x^2$ lower terms)._
    

### Why Hierarchy Matters

The fundamental reason we must use hierarchical models is that only hierarchical polynomials are **invariant under linear transformations**.

If a model is not invariant to a linear shift, it means simply changing the starting point or scale of your data (like changing from Celsius to Fahrenheit, or starting a timer at $t=5$ instead of $t=0$) will completely break the model and yield entirely different predictions.

**Mathematical Proof of the Shift:**

Suppose we want to shift our input data $x$ by a constant $c$. We define a new variable $z = x + c$.

If we have a non-hierarchical term like $x^3$, and we substitute our shifted variable, we get:

$$x^3 = (z - c)^3$$

If we expand this binomial, we get:

$$z^3 - 3cz^2 + 3c^2z - c^3$$

Look at what happened during the expansion! Simply shifting the data naturally generated a $z^2$ term, a $z$ term, and a constant term.

If our original model _lacked_ those lower-degree terms (like the "Not Hierarchical" example above), the model has no place to store or compute these newly generated values. The model literally **cannot represent the exact same function after a simple data shift**.

However, if the model _was_ hierarchical, it already possessed $w_1x$ and $w_2x^2$ terms. The new constants generated by the shift simply get absorbed into the existing weights ($w$), and the cubic shape of the function is perfectly preserved.

---

## 2. Introduction to Regularization (Weight Decay)

As we have seen, using higher-degree polynomials (or having a highly variant dataset) often leads to **Overfitting** and **Ill-Conditioning**. The optimization algorithm tries so hard to minimize the error that it assigns massive, unstable values to the weights ($w_j$).

If we don't want to manually guess the perfect, small polynomial degree, how can we mathematically force the model to keep its weights small and stable? We use **Regularization**.

### The Regularized Loss Function

Up until now, our goal was solely to minimize the Least Squares error ($E_D$), which measures the difference between our predictions and the actual data:

$$E_D(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^N \{t_n - \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}_n)\}^2$$

Regularization works by adding a **Penalty Term** (also called a regularization term, $E_W$) to the loss function. We are essentially telling the optimization algorithm: _"I want you to minimize the data error, BUT I will punish you if you use excessively large weights to do it."_

The total loss function becomes:

$$\text{Total Loss} = E_D(\mathbf{w}) + \lambda E_W(\mathbf{w})$$

- $E_D(\mathbf{w})$ is the data error.
    
- $E_W(\mathbf{w})$ is the weight penalty.
    
- $\lambda$ (Lambda) is the **Regularization Coefficient** (a hyperparameter we choose). If $\lambda$ is large, we punish large weights heavily. If $\lambda$ is $0$, there is no penalty, and we are back to standard linear regression.
    

### L2 Regularization (Ridge Regression)

The most common form of regularization is L2 regularization (often called Ridge Regression or Weight Decay). In L2, our penalty is the squared magnitude of the weight vector:

$$E_W(\mathbf{w}) = \frac{1}{2} \mathbf{w}^T\mathbf{w} = \frac{1}{2} \sum_{j=0}^{M-1} w_j^2$$

_(We include the $1/2$ fraction purely for mathematical convenience, so it cancels out cleanly when we take the derivative)._

By adding this term, the optimization algorithm must now balance a trade-off. If it wants to use a massive weight like $10,000,000$ to perfectly hit a single data point, the squared penalty ($10,000,000^2$) will cause the Total Loss to explode. Therefore, the algorithm is forced to keep the weights small and stable, effectively "decaying" them toward zero and smoothing out the wild oscillations of overfitting!

### Solving L2 Regularization

Because the L2 penalty is just a simple square, it is easily differentiable. If we take the derivative of the new Total Loss function with respect to $\mathbf{w}$ and set it to zero, we get the modified Normal Equation for Ridge Regression:

$$\mathbf{w}_{\text{ridge}} = (\lambda \mathbf{I} + \mathbf{\Phi}^T \mathbf{\Phi})^{-1} \mathbf{\Phi}^T \mathbf{t}$$

_(Where $\mathbf{I}$ is the Identity matrix)._

This tiny addition of $\lambda \mathbf{I}$ to our design matrix is incredibly powerful. Not only does it shrink the weights, but it also mathematically "cures" ill-conditioning. By adding a positive number to the diagonal of the matrix, we ensure that the matrix can never be strictly singular, making the inversion mathematically stable every single time!

---

### See it in Action: Regularization Explorer

_Use this interactive widget to physically see how L2 Regularization tames an overfitted curve. Watch the "Calculated Weights" panel—as you increase the Lambda ($\lambda$) penalty, the massive weights will shrink down, and the wild oscillations will flatten out into a smooth, generalized curve._

<iframe src="file:///home/blueberry/Picsart-Academy/ML/Obsidian Vault/regularization_viz.html" width="100%" height="750" frameborder="0"></iframe>
