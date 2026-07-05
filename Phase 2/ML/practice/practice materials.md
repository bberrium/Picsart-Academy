
## GLM
<iframe style="border:none" width="800" height="450" src="https://whimsical.com/embed/GGaoZxP8BQCCCpQWd1Ri9h@NKBbAEvLSyiLA6QfySiSESFyYijS5nseo"></iframe>
### 1. $h$ stands for **Hypothesis**

In machine learning, "hypothesis" is just a fancy word for **"the prediction rule."** * Think of $h$ as the actual software or mathematical formula you deploy to the real world after training is finished.

- You feed it a new piece of data ($x$), and $h(x)$ spits out your final guess.
    
- In the specific case of logistic regression, $h(x)$ doesn't output a hard "Yes" or "No." It outputs a probability between 0 and 1 (like 0.85, meaning an 85% chance the answer is "Yes").
    

### 2. $E$ stands for **Expected Value**

In probability, Expected Value is the **"average expected outcome."** * Imagine you have a loaded coin that lands on Heads ($1$) 80% of the time, and Tails ($0$) 20% of the time.

- If you flip it once, you can't get a 0.8. You only get a 0 or a 1.
    
- But if you flip it a million times, the _average_ of all those 0s and 1s will be exactly 0.8. Therefore, the "Expected Value" is 0.8.
    
- In binary classification (where $y$ is only 0 or 1), the Expected Value ($E$) is always exactly equal to the probability of getting a 1.
    

### Putting it together: Why $h = E$

The block of equations in your text is just a formal way of stating the following logic:

1. **The Goal:** We want to build a prediction machine ($h$).
    
2. **The Logic:** What should our machine predict? It should predict the most mathematically sound guess, which is the Expected Value ($E$).
    
3. **The Math:** Since our data is binary (0 or 1), the Expected Value ($E$) is just the probability of getting a 1 ($\phi$).
    
4. **The Result:** Therefore, our prediction machine ($h$) is exactly equal to the formula for that probability: $1 / (1 + e^{-\theta^T x})$.

---
### 1. The Indicator Function: $1\{\cdot\}$

The indicator function is just a mathematical "True/False" switch. Think of it like a bouncer asking a single yes/no question:

- If the statement inside the brackets is **True**, the bouncer hands you a **1**.
    
- If the statement inside the brackets is **False**, the bouncer hands you a **0**.
    

_Example:_ Let's say $y$ represents the weather today, where $1$ = Sunny, $2$ = Rainy, and $3$ = Snowy. If today is Rainy ($y = 2$):

- $1\{y = 1\}$ evaluates to **0** (Is 2 equal to 1? False.)
    
- $1\{y = 2\}$ evaluates to **1** (Is 2 equal to 2? True.)
    

### 2. The Relationship: $(T(y))_i = 1\{y = i\}$

In machine learning, we often can't feed raw categories (like "Rainy" or the number "2") into our math equations because the math might think 2 is "twice as big" as 1, which isn't true for categories. Instead, we use a technique called **One-Hot Encoding**.

The equation $(T(y))_i = 1\{y = i\}$ is just the formal mathematical way to write "Turn this category into a One-Hot Encoded vector."

Let's stick with our weather example ($y=2$ for Rainy). $T(y)$ becomes a list (or vector) representing all possible weather states. To fill out this list, we run the indicator function for every possible position $i$:

- Position 1 ($i=1$): $(T(y))_1 = 1\{2 = 1\} = \mathbf{0}$
    
- Position 2 ($i=2$): $(T(y))_2 = 1\{2 = 2\} = \mathbf{1}$
    
- Position 3 ($i=3$): $(T(y))_3 = 1\{2 = 3\} = \mathbf{0}$
    

So, $T(y)$ becomes the array **`[0, 1, 0]`**. The $1$ "indicates" which category is currently active.

### 3. The Expected Value: $E[(T(y))_i] = P(y = i) = \phi_i$

Now that we have our list of 0s and 1s, what happens if we take the Expected Value (the long-term average)?

Imagine you live in a city where it rains 30% of the time.

- $P(y = 2)$ is the probability of rain, which is $0.30$. This is our $\phi_2$.
    
- If you record the weather every day for 100 days, you will write down a `1` in the rainy column about 30 times, and a `0` about 70 times.
    
- If you calculate the average (the Expected Value, $E$) of all those 1s and 0s in the rainy column, you get: $(30 \times 1 + 70 \times 0) / 100 = \mathbf{0.30}$.
    

Therefore, the average of an indicator function ($E[(T(y))_i]$) is always exactly equal to the probability of that event happening ($\phi_i$).

**Summary of the paragraph:**

To predict multiple categories, we convert our target $y$ into a list of 0s and 1s (One-Hot Encoding) using an indicator function. The average value of any slot in that list is simply the probability of that specific category occurring.


## Residual Analysis
<iframe src="file:///home/blueberry/Picsart-Academy/ML/Obsidian Vault/residuals.html" width="100%" height="750" frameborder="0"></iframe>
### Worked Example: Univariate Quadratic Fit

**Scenario:** We have a real estate dataset with 10 properties. We want to predict the house price ($y$) from the square footage ($x$). An initial scatter plot suggests a curved relationship (a quadratic form).

**Data:**

| **x (sqft, in 1000s)** | **y (price, in 1000s)** |
| ---------------------- | ----------------------- |
| 1.0                    | 100                     |
| 1.5                    | 150                     |
| 2.0                    | 230                     |
| 2.5                    | 320                     |
| 3.0                    | 430                     |
| 3.5                    | 550                     |
| 4.0                    | 680                     |
| 4.5                    | 820                     |
| 5.0                    | 980                     |
| 5.5                    | 1150                    |

**Observation:** Prices increase rapidly (price differences widen as size increases), suggesting quadratic growth where $y \propto x^2$.

---

#### Step 1: Create the Polynomial Feature Matrix

For a degree $d = 2$, we create three columns representing the constant, linear, and quadratic terms:

$$X_{poly} = \begin{bmatrix} 1 & 1.0 & 1.0^2 \\ 1 & 1.5 & 1.5^2 \\ 1 & 2.0 & 2.0^2 \\ 1 & 2.5 & 2.5^2 \\ 1 & 3.0 & 3.0^2 \\ 1 & 3.5 & 3.5^2 \\ 1 & 4.0 & 4.0^2 \\ 1 & 4.5 & 4.5^2 \\ 1 & 5.0 & 5.0^2 \\ 1 & 5.5 & 5.5^2 \end{bmatrix} = \begin{bmatrix} 1 & 1.0 & 1.00 \\ 1 & 1.5 & 2.25 \\ 1 & 2.0 & 4.00 \\ 1 & 2.5 & 6.25 \\ 1 & 3.0 & 9.00 \\ 1 & 3.5 & 12.25 \\ 1 & 4.0 & 16.00 \\ 1 & 4.5 & 20.25 \\ 1 & 5.0 & 25.00 \\ 1 & 5.5 & 30.25 \end{bmatrix}$$

**Result:** A $10 \times 3$ matrix (10 samples, 3 features including the bias term).

---

#### Step 2: Solve the Normal Equation

Compute the Ordinary Least Squares (OLS) coefficients:

$$\beta = (X_{poly}^T X_{poly})^{-1} X_{poly}^T y$$

_(Computational details are omitted, as in practice this is handled by libraries like `numpy` or `scikit-learn`.)_

Suppose the calculated solution is:

$$\beta = \begin{bmatrix} -50 \\ 30 \\ 90 \end{bmatrix}$$

**Interpretation of Parameters:**

- **$\beta_0 = -50$:** The baseline or intercept (has limited physical meaning here, as a 0 sqft house doesn't exist).
    
- **$\beta_1 = 30$:** The linear effect of $x$ (adds roughly 30k per 1,000 sqft linearly).
    
- **$\beta_2 = 90$:** The quadratic effect (the nonlinear acceleration; price grows faster at larger sizes).
    

---

#### Step 3: Write the Fitted Model

Using our solved coefficients, the final prediction equation is:

$$\hat{y} = -50 + 30x + 90x^2$$

---

#### Step 4: Make Predictions

**On training data (Checking fit quality):**

Let's test the prediction for $x = 2.5$:

$$\hat{y} = -50 + 30(2.5) + 90(2.5)^2$$

$$\hat{y} = -50 + 75 + 90(6.25)$$

$$\hat{y} = -50 + 75 + 562.5 = 587.5$$

- **Actual value:** $y = 320$
    
- **Residual ($r$):** $320 - 587.5 = -267.5$
    
    _(Note: This is a very large residual, indicating this specific "supposed" $\beta$ model underfits this point heavily)._
    

**On new data (Interpolation example):**

Predict the price for $x = 3.2$ (a value not in our training set, but within our range limits):

$$\hat{y} = -50 + 30(3.2) + 90(3.2)^2$$

$$\hat{y} = -50 + 96 + 90(10.24)$$

$$\hat{y} = -50 + 96 + 921.6 = 967.6$$

**Prediction:** A 3,200 sqft house is estimated to cost **$967,600**.

#### **Step 5: Goodness-of-Fit Assessment**

Compute $R^2$ (fraction of variance explained):

$$
\text{SS}_{\text{res}} = \sum_i (y_i - \hat{y}_i)^2, \quad \text{SS}_{\text{tot}} = \sum_i (y_i - \bar{y})^2
$$

_(Calculations depend on actual fitted values; let's say we get_ $R^2 = 0.92$_)_

Interpretation: **Model explains 92% of price variance**, which is quite good. The 8% unexplained is likely due to omitted factors (location, condition, etc.).
<iframe style="border:none" width="800" height="450" src="https://whimsical.com/embed/LTfiZS6uYZ1WGLTNRhQnPt"></iframe>

