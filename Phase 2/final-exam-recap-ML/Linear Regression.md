
- [[#Comprehensive Practice Exam WITH ANSWERS|Comprehensive Practice Exam WITH ANSWERS]]
	- [[#Comprehensive Practice Exam WITH ANSWERS#Linear Regression — Complete Question Bank|Linear Regression — Complete Question Bank]]
- [[#SECTION 1: Problem Formulation & The Simple Linear Model|SECTION 1: Problem Formulation & The Simple Linear Model]]
		- [[#Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 2: The Least Squares Method & MSE|SECTION 2: The Least Squares Method & MSE]]
		- [[#Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 3: Multivariate Linear Regression|SECTION 3: Multivariate Linear Regression]]
		- [[#Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 4: When XᵀX Is Not Invertible|SECTION 4: When XᵀX Is Not Invertible]]
		- [[#Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 5: Model Evaluation|SECTION 5: Model Evaluation]]
		- [[#Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 6: The Bias Term & Regularization Connection|SECTION 6: The Bias Term & Regularization Connection]]
		- [[#Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
	- [[#SECTION 6: The Bias Term & Regularization Connection#BONUS CHALLENGE QUESTIONS|BONUS CHALLENGE QUESTIONS]]
		- [[#BONUS CHALLENGE QUESTIONS#✅ Answer|✅ Answer]]
	- [[#SECTION 6: The Bias Term & Regularization Connection#Complete Answer Reference Table|Complete Answer Reference Table]]
	- [[#SECTION 6: The Bias Term & Regularization Connection#Top 6 Most Likely Exam Questions From This Topic|Top 6 Most Likely Exam Questions From This Topic]]

# Comprehensive Practice Exam WITH ANSWERS
## Linear Regression — Complete Question Bank

---

# SECTION 1: Problem Formulation & The Simple Linear Model

---

**Question 1**
You are introduced to the regression problem for the first time.

1. Formally define the regression problem. What do we have, what is our goal, and how do we approach it? Distinguish between predicting a single value ŷ(x) and modeling the full predictive distribution p(y|x).
2. Write the equation for the simple linear regression model. Explain every component and what "linear in the parameters" means specifically.
3. Explain the geometric interpretation of simple linear regression in 2D. What do the slope and intercept represent physically, and what is a residual?

### ✅ Answer

**Part 1:**
**Formal definition of the regression problem:**

**What we have:**
- A training dataset of N observations: {xₙ, yₙ} for n = 1, ..., N
- Each xₙ is an input vector of D features: xₙ = (xₙ₁, xₙ₂, ..., xₙD)ᵀ
- Each yₙ is a continuous real-valued target/label

**What is our goal:**
To predict the value of y for a new, unseen value of x — generalization from training examples to new inputs.

**Two levels of the goal:**
1. **Point prediction:** Construct a function ŷ(x) that produces a single predicted value for any new input x. This gives "the most likely" prediction but no uncertainty.
2. **Distributional prediction:** Model the full predictive distribution p(y|x) — not just the most likely value, but the entire probability distribution of y given x. This captures uncertainty — how confident we are about the prediction, and what range of values is plausible.

The distributional approach is richer: for example, p(y|x) = N(ŷ(x), σ²) says the prediction is a Gaussian centered at ŷ(x) with variance σ² — giving both a best guess and a confidence interval.

**Part 2:**
**Simple linear regression model:**

ŷ(x) = w₀ + w₁x₁ + w₂x₂ + ... + wDxD = w₀ + Σⱼ₌₁ᴰ wⱼxⱼ

In compact vector notation: ŷ(x) = wᵀx (with bias absorbed into w)

Every component:
- **ŷ(x):** The predicted output value for input x
- **x = (x₁, x₂, ..., xD)ᵀ:** The input feature vector with D features
- **w₀:** The bias (intercept) — the predicted value when all features are zero
- **w₁, ..., wD:** The weights (coefficients) — one per feature
- **wⱼ:** The weight for feature j — measures how much ŷ changes per unit increase in xⱼ, holding all other features constant

**"Linear in the parameters" means:**
The model is a linear function of w₀, w₁, ..., wD — the parameters appear only as first-degree terms, never squared or in products with each other. If you double all weights, the output doubles. This is critical because it makes the optimization problem convex (has a unique global minimum with a closed-form solution).

Importantly: the model CAN be nonlinear in x (e.g., you can include x², x³ as features) and it is still "linear regression" — because linearity refers to the parameters w, not the inputs x.

**Part 3:**
**Geometric interpretation in 2D (one feature x, one output y):**

The model ŷ = w₀ + w₁x defines a **straight line** in the x-y plane.

**Slope (w₁):**
The rate of change of ŷ with respect to x — (change in y)/(change in x). A slope of 2 means: for every 1 unit increase in x, the prediction increases by 2 units. A positive slope means x and y move together; negative slope means they move in opposite directions.

**Intercept (w₀):**
The predicted value of y when x = 0. Geometrically, where the line crosses the y-axis. It shifts the entire line up or down without changing its angle.

**Residual:**
For each training point (xₙ, yₙ), the residual is:
rₙ = yₙ - ŷ(xₙ) = yₙ - (w₀ + w₁xₙ)

The residual is the **vertical distance** from the actual data point to the fitted line. Positive residual means the point is above the line (actual > predicted). Negative residual means the point is below the line (actual < predicted). A perfect fit has all residuals = 0 (all points on the line).

---

# SECTION 2: The Least Squares Method & MSE

---

**Question 2**
The least squares method is the foundation of linear regression.

1. Explain the principle of least squares. What exactly are we minimizing and why do we square the residuals rather than just summing them or using absolute values?
2. Write the Mean Squared Error (MSE) formula. Explain every component and both its key advantage and key disadvantage.
3. Explain why MSE is not robust to outliers. Give a concrete numerical example showing how one extreme outlier dramatically changes the MSE compared to the case without it.

### ✅ Answer

**Part 1:**
**Principle of Least Squares:**

Least squares minimizes the **sum of squared residuals (RSS)**:

RSS = Σₙ₌₁ᴺ (yₙ - ŷ(xₙ))² = Σₙ₌₁ᴺ rₙ²

We find the parameters w₀, w₁, ..., wD that make this sum as small as possible.

**Why square the residuals rather than just summing them?**
1. **Cancellation problem:** Positive and negative residuals cancel. A line that is +10 above half the points and -10 below the other half has sum of residuals = 0 — appearing perfect when it is actually terrible. Squaring makes all residuals positive, preventing cancellation.

**Why not absolute values |rₙ|?**
1. **Differentiability:** |rₙ| is not differentiable at rₙ = 0 — it has a sharp corner. This makes optimization harder (no smooth gradient at zero). MSE is differentiable everywhere, making calculus-based optimization straightforward.
2. **Closed-form solution:** The squared loss leads to a system of linear equations (Normal Equations) with an exact analytical solution. Absolute value loss (L1) has no closed-form solution and requires iterative methods.
3. **Statistical properties:** Squaring connects to maximum likelihood estimation under Gaussian noise — the least squares solution is the Maximum Likelihood Estimator when errors are Gaussian.

**Why not higher powers (e.g., 4th power)?**
Higher powers penalize large residuals even more severely, becoming even less robust to outliers. Squaring is the sweet spot: eliminates cancellation, differentiable, leads to clean linear algebra, has Gaussian MLE interpretation.

**Part 2:**
**MSE Formula:**

MSE = (1/N) Σₙ₌₁ᴺ (yₙ - ŷₙ)²

Every component:
- **N:** Total number of training observations
- **1/N:** Averaging factor — divides by N to make MSE independent of dataset size (comparable across datasets of different sizes)
- **yₙ:** The actual/true target value for observation n
- **ŷₙ:** The predicted value for observation n
- **(yₙ - ŷₙ):** The residual for observation n — prediction error
- **(yₙ - ŷₙ)²:** Squared error — always positive, penalizes large errors more than small ones
- **Σ:** Sum over all N training examples

**Key advantage:**
The MSE graph is a **differentiable, smooth, convex function** of the parameters w. This means:
- Calculus-based optimization (gradient descent, closed-form derivatives) works perfectly
- There are no local minima — any minimum found is the global minimum
- The derivative points clearly in the direction of steepest descent

**Key disadvantage:**
The value is in **squared units of the output**. If you are predicting house prices in dollars, MSE is in dollars² — not directly interpretable. To get interpretable error units, you take the square root (RMSE = √MSE). Additionally, MSE is **not robust to outliers** (see Part 3).

**Part 3:**
Concrete numerical example of MSE sensitivity to outliers:

**Dataset without outlier:** Predicted = 10 for all, Actual values: [9, 11, 10, 10, 10]
- Residuals: [-1, +1, 0, 0, 0]
- MSE = (1 + 1 + 0 + 0 + 0)/5 = **0.4**

**Same dataset WITH one outlier:** Replace last value with 50
Actual values: [9, 11, 10, 10, 50], Predicted = 10 for all
- Residuals: [-1, +1, 0, 0, +40]
- Squared residuals: [1, 1, 0, 0, 1600]
- MSE = (1 + 1 + 0 + 0 + 1600)/5 = **320.4**

One outlier increased MSE from **0.4 to 320.4** — an 800× increase. The squaring operation amplifies the outlier's residual (40²= 1600) to completely dominate the MSE, making the metric misrepresent the model's performance on the other 4 perfectly reasonable points.

This is why MSE is not robust to outliers — a single extreme value can make a good model look terrible, or conversely, pull the fitted line toward itself (distorting the weights) to reduce its own squared residual at the expense of all other points.

---

**Question 3**
Derive the optimal parameters for 1D linear regression analytically.

1. Write the MSE loss function for 1D linear regression (y = w₀ + w₁x). Explain the derivation strategy: why do we take partial derivatives and set them to zero?
2. Derive the optimal w₁ (slope) step by step. Show all algebraic manipulations clearly.
3. Explain what the resulting closed-form solution tells us. What does the formula for w₁ reveal about the relationship between the optimal slope and the data?

### ✅ Answer

**Part 1:**
**MSE for 1D linear regression:**

J(w₀, w₁) = (1/N) Σₙ₌₁ᴺ (yₙ - w₀ - w₁xₙ)²

**Why take partial derivatives and set to zero:**
J is a smooth, convex function of w₀ and w₁ (it is a quadratic polynomial in these parameters — a paraboloid in 2D parameter space). For a convex function, the global minimum occurs exactly where the gradient equals zero — where the function is flat in all directions.

Setting ∂J/∂w₀ = 0 and ∂J/∂w₁ = 0 simultaneously gives us the system of equations whose solution is the unique global minimum. This is guaranteed to be the global minimum (not just a local minimum or saddle point) because J is strictly convex when the xₙ values are not all identical.

**Strategy:**
1. Take partial derivative with respect to w₀
2. Take partial derivative with respect to w₁
3. Set both equal to zero
4. Solve the resulting 2×2 linear system

**Part 2:**
**Full derivation:**

**Step 1 — Partial derivative w.r.t. w₀:**
∂J/∂w₀ = (1/N) Σₙ 2(yₙ - w₀ - w₁xₙ)(-1) = 0

Simplify:
Σₙ (yₙ - w₀ - w₁xₙ) = 0

Σₙ yₙ - Nw₀ - w₁Σₙ xₙ = 0

Divide by N:
ȳ - w₀ - w₁x̄ = 0

**→ w₀ = ȳ - w₁x̄** ... (equation 1)

Where ȳ = (1/N)Σyₙ and x̄ = (1/N)Σxₙ are the sample means.

**Step 2 — Partial derivative w.r.t. w₁:**
∂J/∂w₁ = (1/N) Σₙ 2(yₙ - w₀ - w₁xₙ)(-xₙ) = 0

Σₙ xₙ(yₙ - w₀ - w₁xₙ) = 0

Σₙ xₙyₙ - w₀ Σₙ xₙ - w₁ Σₙ xₙ² = 0 ... (equation 2)

**Step 3 — Substitute equation 1 into equation 2:**
Substitute w₀ = ȳ - w₁x̄:

Σₙ xₙyₙ - (ȳ - w₁x̄)Σₙ xₙ - w₁ Σₙ xₙ² = 0

Σₙ xₙyₙ - ȳΣₙ xₙ - w₁x̄Σₙ xₙ + w₁ Σₙ xₙ² = 0 (note sign error corrected: -w₁x̄Σxₙ should be +w₁x̄Σxₙ... let me redo cleanly)

Σₙ xₙyₙ - (ȳ - w₁x̄)(Nx̄) - w₁ Σₙ xₙ² = 0

Σₙ xₙyₙ - Nȳx̄ + Nw₁x̄² - w₁ Σₙ xₙ² = 0

Σₙ xₙyₙ - Nȳx̄ = w₁(Σₙ xₙ² - Nx̄²)

**→ w₁ = (Σₙ xₙyₙ - Nȳx̄) / (Σₙ xₙ² - Nx̄²)**

This can be rewritten as:
**w₁ = Σₙ(xₙ - x̄)(yₙ - ȳ) / Σₙ(xₙ - x̄)²**

**Part 3:**
**What the formula reveals:**

The optimal slope w₁ = Σ(xₙ-x̄)(yₙ-ȳ) / Σ(xₙ-x̄)² has a deep statistical interpretation:

- **Numerator: Σ(xₙ-x̄)(yₙ-ȳ)** = the sample **covariance** (unnormalized) between x and y. It measures how much x and y vary together. Positive = they tend to increase together. Negative = one increases as the other decreases.

- **Denominator: Σ(xₙ-x̄)²** = the sample **variance** of x (unnormalized). It measures how spread out the x values are.

Therefore: **w₁ = Cov(x,y) / Var(x)**

This is exactly the **Pearson correlation coefficient scaled by the ratio of standard deviations**: w₁ = ρ × (σy/σx)

**Interpretations:**
1. If x and y are highly positively correlated → large positive w₁
2. If x and y are uncorrelated → w₁ ≈ 0 → flat line at ȳ
3. The slope adjusts for the scale of x: if x values are spread far apart (large Var(x)), you need a smaller slope to explain the same covariance

Also from equation 1: **w₀ = ȳ - w₁x̄**
The regression line always passes through the point (x̄, ȳ) — the mean of x and mean of y. The intercept adjusts the line to pass through the centroid of the data.

---

# SECTION 3: Multivariate Linear Regression

---

**Question 4**
Multivariate linear regression extends to multiple features using matrix notation.

1. Write the multivariate linear regression model in both scalar form (sum over features) and matrix/vector form. Explain the dimensions of every matrix and vector.
2. Derive the matrix form of the MSE loss function J(w) = (1/N)||y - Xw||². Explain why expanding this expression leads to the quadratic form it takes.
3. Derive the Normal Equations by taking the gradient of J with respect to w and setting it to zero. Write the final solution w* = (XᵀX)⁻¹Xᵀy and explain every component.

### ✅ Answer

**Part 1:**
**Scalar form:**
ŷₙ = w₀ + w₁xₙ₁ + w₂xₙ₂ + ... + wkxₙk = Σⱼ₌₀ᵏ wⱼxₙⱼ

(where xₙ₀ = 1 for all n to absorb the bias w₀)

**Matrix/Vector form:**
ŷ = Xw

**Dimensions of every component:**

- **X** (design matrix): N × (k+1) — N rows (one per observation), k+1 columns (k features + 1 bias column of all 1s)
  - Row n: [1, xₙ₁, xₙ₂, ..., xₙk] — the feature vector of observation n with prepended 1 for bias
  
- **w** (weight vector): (k+1) × 1 — one weight per feature plus one bias weight
  - [w₀, w₁, w₂, ..., wk]ᵀ

- **ŷ** (prediction vector): N × 1 — one predicted value per observation
  - Matrix multiplication: (N × (k+1)) × ((k+1) × 1) = N × 1 ✓

- **y** (target vector): N × 1 — actual observed values

- **ε** (error vector): N × 1 — residuals for each observation: y = Xw + ε

**Part 2:**
**Matrix form of MSE:**

J(w) = (1/N)||y - Xw||²

Expanding the squared norm:
||y - Xw||² = (y - Xw)ᵀ(y - Xw)

= yᵀy - yᵀ(Xw) - (Xw)ᵀy + (Xw)ᵀ(Xw)

Since yᵀ(Xw) is a scalar and (Xw)ᵀy is its transpose, and the transpose of a scalar equals itself:
yᵀ(Xw) = (Xw)ᵀy

Therefore:
= yᵀy - 2(Xw)ᵀy + wᵀXᵀXw

= yᵀy - 2wᵀXᵀy + wᵀXᵀXw

**Why this is quadratic:**
The term wᵀ(XᵀX)w is a **quadratic form** in w — it contains products of weights with each other (wᵢwⱼ terms). The term -2wᵀXᵀy is linear in w. yᵀy is a constant. So J(w) is a quadratic function of w — like a multidimensional parabola. This guarantees a unique global minimum (if XᵀX is positive definite).

**Part 3:**
**Deriving the Normal Equations:**

Take the gradient of J with respect to w:

∇ᵥJ(w) = ∇ᵥ[(1/N)(yᵀy - 2wᵀXᵀy + wᵀXᵀXw)]

= (1/N)(-2Xᵀy + 2XᵀXw)

Set equal to zero:
-2Xᵀy + 2XᵀXw = 0

**XᵀXw = Xᵀy** ← The Normal Equations

Solving for w (assuming XᵀX is invertible):
**w* = (XᵀX)⁻¹Xᵀy**

**Explanation of every component:**

- **Xᵀ:** Transpose of the design matrix, shape (k+1) × N
- **XᵀX:** Shape (k+1) × (k+1) — a square matrix containing dot products of all feature pairs. Entry (i,j) = Σₙ xₙᵢxₙⱼ — the inner product of feature column i with feature column j. Measures feature covariances.
- **(XᵀX)⁻¹:** The inverse of XᵀX — exists when features are linearly independent
- **Xᵀy:** Shape (k+1) × 1 — dot product of each feature column with the target vector. Entry j = Σₙ xₙⱼyₙ — how much feature j covaries with y
- **(XᵀX)⁻¹Xᵀ:** Called the **pseudoinverse** (when XᵀX is invertible) — shape (k+1) × N
- **w* = (XᵀX)⁻¹Xᵀy:** The optimal weight vector — a closed-form analytical solution. No iteration needed.

---

**Question 5**
The geometric interpretation of linear regression provides deep insight.

1. Explain the geometric interpretation of linear regression as an orthogonal projection. What is the column space of X, and why is ŷ = Xw constrained to lie in it?
2. Explain why the residual vector r = y - Xw must be orthogonal to every column of X at the optimal solution. Derive the Normal Equations from this geometric condition.
3. Explain what it means visually when XᵀXw = Xᵀy is called the "Normal Equation." Why "normal"?

### ✅ Answer

**Part 1:**
**Linear regression as orthogonal projection:**

The prediction ŷ = Xw is a linear combination of the columns of X:
ŷ = w₀×(column 0 of X) + w₁×(column 1 of X) + ... + wk×(column k of X)

By definition, any linear combination of the columns of X lies in the **column space of X** (also called the range or image of X) — the set of all vectors that can be expressed as Xw for some w.

Therefore: **ŷ ∈ Col(X)** — the prediction is always constrained to lie in this subspace, regardless of what w we choose.

**The geometric problem:**
The target vector y lives in N-dimensional space. The column space Col(X) is a (k+1)-dimensional subspace of that N-dimensional space (assuming k+1 < N). Generally, y does NOT lie in Col(X) — there is no w that makes Xw = y exactly (the system is overdetermined).

**The solution:**
We want to find the point ŷ* in Col(X) that is **closest to y** — minimizing the distance ||y - ŷ||² = ||y - Xw||². The closest point in a subspace to a point outside it is obtained by **orthogonal projection** — dropping a perpendicular from y onto Col(X).

Therefore: **Linear regression = orthogonal projection of y onto Col(X)**

**Part 2:**
**Why the residual must be orthogonal to Col(X):**

At the optimal solution, we have found the closest point ŷ* in Col(X) to y. The residual vector r = y - ŷ* = y - Xw* is the vector from ŷ* to y.

For ŷ* to be the closest point in Col(X) to y, the residual r must be **perpendicular to the subspace Col(X)**. If r were not perpendicular — if r had any component pointing within Col(X) — then we could move ŷ* a tiny bit in that direction within Col(X) and get even closer to y. This would contradict ŷ* being the minimum.

**Deriving Normal Equations from this:**
r ⊥ Col(X) means r is orthogonal to every column of X:
Xᵀr = 0 (each column of X has zero dot product with r)

Substituting r = y - Xw:
Xᵀ(y - Xw) = 0
Xᵀy - XᵀXw = 0
**XᵀXw = Xᵀy** ← Normal Equations derived geometrically

The Normal Equations are not just an algebraic trick — they are the mathematical expression of the geometric condition that the residual must be orthogonal to the feature subspace.

**Part 3:**
**Why "Normal" Equations:**

The word "normal" comes from geometry — in geometry, a **normal** to a surface is a vector **perpendicular (orthogonal)** to it. In Latin, "norma" means a right angle.

The Normal Equations encode the condition that the residual vector r = y - Xw is **normal (perpendicular)** to the column space of X:

r ⊥ Col(X) ↔ Xᵀr = 0 ↔ XᵀXw = Xᵀy

The equations are called "Normal" because they express the normality (perpendicularity) condition of the residual to the feature subspace. This is not "normal" in the sense of "ordinary" — it specifically refers to the geometric concept of orthogonality (normal vectors).

**Visual interpretation:**
Imagine y as a point floating above a 2D plane (Col(X)) embedded in 3D space. The optimal prediction ŷ is the shadow of y on the plane — the point directly below y on the plane. The residual r is the vertical line from ŷ to y, which is perfectly perpendicular (normal) to the plane. The Normal Equations express exactly this perpendicularity condition.

---

# SECTION 4: When XᵀX Is Not Invertible

---

**Question 6**
The Normal Equations require XᵀX to be invertible, which is not always the case.

1. List four specific conditions under which XᵀX is NOT invertible (singular). Explain what each condition means in practical terms.
2. If XᵀX is singular, does the optimization problem min||y - Xw||² still have a solution? Explain carefully. What changes about the nature of the solution?
3. Explain the concept of the null space of X in this context. If z is in the null space of X, why does w + z give the same predictions as w?

### ✅ Answer

**Part 1:**
Four conditions making XᵀX singular:

1. **Some features are linear combinations of others (multicollinearity):**
Example: Feature 3 = 2 × Feature 1 + Feature 2. The column for Feature 3 is redundant — it adds no new information. The columns of X are linearly dependent, so rank(X) < k+1 and XᵀX is singular.
Practical meaning: You have redundant features that carry the same information. The model cannot uniquely determine which features "deserve" the credit for predicting y.

2. **Two columns are identical:**
The most extreme case of multicollinearity — the same feature appears twice. Column i = Column j. Clearly redundant, rank deficient. In practice, this happens when features are accidentally duplicated or when one-hot encoding is done incorrectly.

3. **D > N (more features than observations):**
You have more parameters to estimate than data points. With N observations, you can solve at most N independent equations — having more than N parameters means the system is underdetermined. There are infinitely many weight vectors that perfectly fit the N training points.
Practical example: A genomics study with 100 patients but 50,000 gene expression features — massively underdetermined.

4. **Perfect multicollinearity:**
A generalization of case 1: some subset of features forms a perfect linear relationship. This means the feature matrix has rank less than k+1, making XᵀX singular.

**Part 2:**
**Does the minimum still exist when XᵀX is singular?**

**Yes, the minimum still exists.** The optimization problem:
min_w ||y - Xw||²

is still a convex optimization problem with a convex, differentiable objective function. Convex functions always have global minima. The loss surface is still a bowl shape (or flat-bottomed bowl).

**What changes:** The minimum is **not unique** — there are infinitely many weight vectors w* that all achieve the same minimum loss.

Why infinitely many? Because if z is in the null space of X (Xz = 0), then:
||y - X(w + z)||² = ||y - Xw - Xz||² = ||y - Xw - 0||² = ||y - Xw||²

Adding any null space vector to a solution gives another solution with identical loss. The loss function is completely **flat along the null space** — you can slide along the null space directions without changing the loss at all.

**Practical consequence:**
- We cannot use w* = (XᵀX)⁻¹Xᵀy because (XᵀX)⁻¹ doesn't exist
- We need either: (1) the Moore-Penrose pseudoinverse to pick the minimum-norm solution, or (2) regularization (Ridge regression) to make the problem uniquely solvable

**Part 3:**
**Null space explanation:**

The null space of X (also called the kernel of X) is the set of all vectors z such that:
**Xz = 0**

These are directions in weight space that produce zero change in the predictions.

**Why w + z gives same predictions as w:**
ŷ(w + z) = X(w + z) = Xw + Xz = Xw + 0 = Xw = ŷ(w)

Adding a null space vector z to the weights changes the weights but produces IDENTICAL predictions for every training point. The null space represents "directions of irrelevance" — weight components that are invisible to the model's predictions.

**Geometric meaning:**
The loss landscape is flat (constant value) along the null space. If the null space is 1-dimensional, the minimum is a line of solutions. If 2-dimensional, the minimum is a plane of solutions. The number of dimensions of the null space equals the degree of rank deficiency (k+1 - rank(X)).

**Practical example:**
If Feature 3 = Feature 1 + Feature 2, then z = [0, 1, 1, -1, 0, ...] is in the null space (adding 1 to w₁, 1 to w₂, and subtracting 1 from w₃ produces identical predictions because the net change to any prediction is xₙ₁(1) + xₙ₂(1) + xₙ₃(-1) = xₙ₁ + xₙ₂ - (xₙ₁+xₙ₂) = 0).

---

**Question 7**
The Moore-Penrose Pseudoinverse provides a principled solution when XᵀX is singular.

1. Explain what the Moore-Penrose Pseudoinverse X⁺ is and what it computes. How does it differ from the regular inverse?
2. Among infinitely many solutions, which one does X⁺ select? Explain both algebraically and geometrically why this is the "natural" choice.
3. Explain the geometric meaning: "The pseudoinverse picks the solution orthogonal to the null space." Why is this geometrically sensible?

### ✅ Answer

**Part 1:**
**Moore-Penrose Pseudoinverse:**

When XᵀX is invertible, the unique solution is:
w* = (XᵀX)⁻¹Xᵀy

When XᵀX is singular (not invertible), we replace this with:
**w* = X⁺y**

where X⁺ is the Moore-Penrose pseudoinverse of X.

**What it computes:**
X⁺ still solves **exactly** the least squares problem:
min_w ||y - Xw||²

It does not approximate — it finds a genuine minimizer of the squared loss, just like the regular inverse. The difference is in WHICH minimizer it selects (see Part 2).

**How it differs from regular inverse:**
- Regular inverse: Only exists when X is square and full rank. If it exists, w = X⁻¹y gives the unique exact solution to Xw = y.
- Pseudoinverse: Always exists for any matrix X (any shape, any rank). When X has full rank, X⁺ = (XᵀX)⁻¹Xᵀ (reduces to the regular solution). When X is rank-deficient, X⁺ still exists and gives the minimum-norm least-squares solution.

The pseudoinverse is computed via Singular Value Decomposition (SVD): if X = UΣVᵀ, then X⁺ = VΣ⁺Uᵀ where Σ⁺ replaces non-zero singular values σᵢ with 1/σᵢ and keeps zeros as zeros.

**Part 2:**
**Which solution does the pseudoinverse select?**

Among all weight vectors w that minimize ||y - Xw||², the pseudoinverse selects:

**w* = argmin ||w||² subject to w minimizes ||y - Xw||²**

It selects the **minimum-norm solution** — the solution with the smallest possible magnitude ||w||.

**Why this is natural:**

**Algebraically:** The minimum-norm solution contains no unnecessary components — it uses only as much "weight" as is strictly needed to achieve the best possible fit. Any other solution w + z (where z is a null space vector) has ||w + z||² = ||w||² + ||z||² ≥ ||w||² (since w is orthogonal to z — see Part 3). So the pseudoinverse solution is uniquely the smallest.

**Practically:** The minimum-norm solution represents the most "parsimonious" explanation — it achieves the same predictions with the smallest weight magnitudes. This connects to regularization: Ridge regression (L2 regularization) also produces minimum-norm solutions as a limit, and the pseudoinverse is the natural unregularized counterpart.

**Part 3:**
**Geometric meaning — orthogonal to null space:**

The solution space is an affine subspace of the form: {w* + z : z ∈ Null(X)} — a translated copy of the null space passing through any particular solution w*.

The pseudoinverse selects the unique point in this solution affine subspace that is **orthogonal to the null space** — the point with no component in the null space direction.

**Why geometrically sensible:**
Think of the simplest 2D case: the null space is a 1D line. The set of all solutions forms a 1D line (parallel to the null space). The pseudoinverse picks the point on this line that is closest to the origin — which is the perpendicular foot from the origin to the solution line.

This is sensible because:
1. **Uniqueness:** There is exactly one point in a convex set closest to the origin — the pseudoinverse is unique
2. **No "arbitrary" components:** The null space represents weight directions with zero effect on predictions. Including null space components in w is arbitrary — any amount can be added or subtracted without changing predictions. The pseudoinverse removes all these arbitrary components by projecting orthogonally out of the null space, leaving only the "essential" weights.
3. **Parsimony principle:** When multiple explanations fit equally well, prefer the simplest (smallest) one

---

# SECTION 5: Model Evaluation

---

**Question 8**
Evaluating regression models requires careful choice of metrics.

1. Explain Residual Sum of Squares (RSS). Write the formula, explain what "smaller RSS = better fit" means, and explain the critical problem with using RSS alone to select the best model.
2. Explain R-squared (R²). What does it measure, what is its range, and why is it described as the "proportion of variance explained"?
3. Explain why more features always increases R² (or keeps it equal) but does NOT mean the model is better. What does this reveal about R² as a model selection criterion?

### ✅ Answer

**Part 1:**
**Residual Sum of Squares (RSS):**

RSS = Σₙ₌₁ᴺ (yₙ - ŷₙ)² = ||y - Xw||²

**Smaller RSS = better fit** because: RSS measures the total squared distance between the model's predictions and the actual observed values. A smaller RSS means the fitted line/hyperplane is closer to the actual data points on average. RSS = 0 means perfect prediction (every prediction exactly correct).

**Critical problem with using RSS alone:**

**More features always decreases RSS (or keeps it equal):**

When you add a new feature to a linear regression model, the model gains one additional degree of freedom. In the worst case, the new feature's optimal weight is zero (feature is useless) — giving exactly the same RSS as before. In practice, the optimizer will find some non-zero weight that reduces RSS slightly, even if the feature is pure noise.

Therefore: comparing models with different numbers of features using RSS is meaningless. A model with 50 useless noise features will always have lower RSS than a model with 5 genuinely informative features — not because it is better, but because it has more parameters to exploit.

This is precisely why you need metrics that penalize complexity: Adjusted R², AIC, BIC, or cross-validation.

**Part 2:**
**R-squared (R²) — Coefficient of Determination:**

R² = 1 - RSS/TSS = 1 - Σ(yₙ - ŷₙ)² / Σ(yₙ - ȳ)²

Where TSS (Total Sum of Squares) = Σ(yₙ - ȳ)² is the total variance of y.

**What it measures:**
The proportion of the total variance in y that is "explained" by (accounted for by) the model's predictions. It measures how much better the model's predictions are compared to simply predicting ȳ (the mean) for every input.

**Range:** R² ∈ [0, 1] for reasonable models
- **R² = 1:** Perfect fit — model explains 100% of variance, every prediction exact
- **R² = 0:** Model explains nothing — no better than predicting the mean every time
- **R² < 0:** Possible but indicates the model performs worse than predicting the mean (very bad model)

**"Proportion of variance explained":**
- TSS measures total variance of y: how much y varies around its mean
- RSS measures unexplained variance: how much of that variation remains after fitting the model
- RSS/TSS = fraction of variance the model FAILED to explain
- 1 - RSS/TSS = fraction the model DID explain = R²

Example: R² = 0.85 means the model's features account for 85% of the variation in y; 15% remains unexplained (noise, missing features, etc.).

**Part 3:**
**Why more features always increases R²:**

Algebraically: Adding a feature to a linear regression model either:
1. **Reduces RSS** (if the feature has any correlation with y) → R² increases
2. **Keeps RSS identical** (if the feature has exactly zero correlation with y) → R² unchanged

R² can never decrease when adding features, regardless of whether those features are genuinely useful or pure random noise. This is because the model can always set the new feature's weight to zero, replicating the previous model exactly, and then only accept the feature if it helps.

**What this reveals about R² as a model selection criterion:**

R² is fundamentally a measure of training fit, not generalization quality. Using R² alone to choose between models with different numbers of features will always favor the model with more features — even if those extra features are pure noise that will hurt performance on new data.

This is a manifestation of overfitting: a model with more parameters always fits training data better (or equally), but this improvement in training fit comes at the cost of higher variance and worse generalization.

**The slides explicitly state:** "More features → higher R² → not a reliable approach for choosing the best model."

**Proper alternatives:**
- **Adjusted R²:** Penalizes for the number of features — R² only increases if the new feature's contribution exceeds what would be expected by chance
- **Cross-validation:** Directly estimates generalization performance on held-out data
- **AIC/BIC:** Information criteria that penalize model complexity
- **F-test:** Statistical significance test for whether added features significantly improve fit

---

# SECTION 6: The Bias Term & Regularization Connection

---

**Question 9**
The role of the bias term in linear regression.

1. Write the insight about the bias term w₀ from the slides. Explain what it means that "the bias compensates for the difference between the averages of target values and weighted sum of averages of input values."
2. Show algebraically why the regression line must pass through the point (x̄, ȳ) — the centroid of the data. What does this tell you about the role of w₀?
3. If you forgot to include the bias term (forced w₀ = 0), what specific problem would occur? Give a concrete example where this would produce dramatically wrong predictions.

### ✅ Answer

**Part 1:**
**The bias insight from the slides:**

At the optimal solution, taking the partial derivative w.r.t. w₀ and setting it to zero gives:
w₀ = ȳ - Σⱼ wⱼx̄ⱼ

**Interpretation:**
The bias w₀ absorbs the difference between:
- **ȳ** — the average target value in the training set
- **Σⱼ wⱼx̄ⱼ** — the weighted sum of the average feature values (what the model would predict at the feature means using only the weighted features)

In other words: the bias ensures that when you plug in the average values of all features, you get the average target value. It "calibrates" the model to the mean of the data.

Without the bias, the model would be forced to predict 0 when all features are 0 — which is almost never what the data actually shows. The bias lifts or lowers the entire prediction surface to match the data's overall level.

**Part 2:**
**Why the regression line passes through (x̄, ȳ):**

From the derivation of 1D regression:
- **w₀ = ȳ - w₁x̄**

Now evaluate the model at x = x̄:
ŷ(x̄) = w₀ + w₁x̄ = (ȳ - w₁x̄) + w₁x̄ = ȳ

Therefore ŷ(x̄) = ȳ — the model's prediction at the mean input equals the mean output.

**Geometrically:** The regression line always pivots around the centroid (x̄, ȳ) of the data. The slope w₁ determines the angle, and the bias w₀ is then set to ensure the line passes through (x̄, ȳ).

**What this tells us about w₀:**
The bias is not a free parameter — it is completely determined by ȳ, w₁, and x̄. Once you know the slope and the data's centroid, the intercept is fixed. The bias exists to anchor the line to the data's center of mass, not to provide additional model flexibility.

**Part 3:**
**Problem when w₀ = 0 is forced:**

If w₀ = 0, the model is: ŷ = w₁x₁ + w₂x₂ + ... — a hyperplane passing through the origin.

**The specific problem:**
The model cannot represent any relationship where the output is non-zero when all inputs are zero. It is forced to predict ŷ = 0 whenever all features are 0, regardless of whether that makes sense.

**Concrete example:**
Predicting house prices with features like square footage and number of rooms. A house with 0 square footage and 0 rooms would be "predicted" to cost $0 — which is the forced constraint. But for any real house, the intercept might capture baseline costs: land value, fixed construction costs, location value. A house with 1000 sq ft might cost $200,000, but if forced through the origin, the model would predict: price = w₁ × 1000. The weight w₁ would need to simultaneously capture both the per-square-foot value AND the baseline cost, leading to systematically wrong predictions for all house sizes — too high for small houses, too low for large ones (or vice versa).

The regression line would be tilted awkwardly to pass through the origin even though the data's centroid is far from the origin, causing systematic prediction errors throughout the data range.

---

**Question 10**
Connect the concepts learned in linear regression to the broader ML concepts from earlier slides.

1. Linear regression has a closed-form solution (Normal Equations). Why don't we always use this closed-form solution in practice? What are its limitations compared to iterative optimization (Gradient Descent)?
2. Explain how adding polynomial features (x², x³, x₁x₂) to a linear regression model connects to the Kernel Trick from SVMs. What is fundamentally similar about these two approaches?
3. You train a linear regression model on 1000 samples with 5 features and get MSE = 2.1 on training and MSE = 2.3 on test. You then train a degree-10 polynomial regression on the same data and get MSE = 0.01 on training and MSE = 147 on test. Diagnose both models using ALL relevant concepts: Bias-Variance, overfitting/underfitting, regularization implications.

### ✅ Answer

**Part 1:**
**Limitations of the closed-form solution:**

The Normal Equations: w* = (XᵀX)⁻¹Xᵀy

**Computational limitations:**
1. **Matrix inversion cost:** Inverting a (k+1) × (k+1) matrix requires O(k³) operations. With k = 10,000 features (common in modern ML), this is 10¹² operations — computationally infeasible.
2. **Memory:** Storing XᵀX requires O(k²) memory. For k = 100,000 features, XᵀX alone requires 80 GB of memory (100,000² × 8 bytes = 80 GB).
3. **Large datasets:** Computing XᵀX requires a full pass over all N data points simultaneously. For N = 1 billion, holding the entire X in memory is infeasible. Gradient Descent can process data in mini-batches.

**Numerical limitations:**
4. **Ill-conditioning:** When XᵀX is nearly singular (multicollinearity), its inverse is numerically unstable — tiny errors in input cause huge errors in the solution. Iterative methods can be more numerically stable.

**Gradient Descent advantages:**
- Works with any differentiable loss (Normal Equations only for squared loss)
- Scales to massive datasets via mini-batch / stochastic gradient descent
- Can incorporate regularization naturally (just add regularization term to gradient)
- Finds solutions even when XᵀX is singular (with appropriate step size)
- Memory efficient — processes one batch at a time

**Part 2:**
**Connection between polynomial features and the Kernel Trick:**

**Polynomial features in linear regression:**
Instead of fitting ŷ = w₀ + w₁x, you create new features: x², x³, x₁x₂ and fit:
ŷ = w₀ + w₁x + w₂x² + w₃x³ + ...

This is still "linear regression" (linear in w) but the input has been mapped to a higher-dimensional feature space: φ(x) = [1, x, x², x³, ...]. The model is actually nonlinear in the original feature x.

**What is fundamentally similar to the Kernel Trick:**
Both approaches map data to a higher-dimensional feature space to capture nonlinear relationships:

| Property | Polynomial Features | Kernel Trick (SVM) |
|----------|--------------------|--------------------|
| Core idea | Explicitly create φ(x) | Implicitly compute φ(x)ᵀφ(y) via K(x,y) |
| Feature space | Finite dimensional | Can be infinite dimensional |
| Computation | Explicit — must compute all features | Implicit — never compute φ(x) directly |
| Boundary | Polynomial in original space | Same, but without explicit feature computation |

**The key difference:**
Polynomial features **explicitly compute** the high-dimensional representation — you literally create all the x², x₁x₂ columns. This is feasible for low degree but explodes for high degree or high input dimension.

The Kernel Trick **implicitly** performs the same operation — computing the dot product in feature space without ever computing the features. For infinite-dimensional spaces (RBF kernel), explicit computation is impossible. The Kernel Trick makes it feasible.

**Conclusion:** Polynomial feature expansion is the explicit version of the Kernel Trick. The Kernel Trick generalizes and makes computationally tractable what polynomial feature expansion does explicitly.

**Part 3:**
**Complete diagnosis:**

**Linear regression (MSE_train = 2.1, MSE_test = 2.3):**

- **Overfitting/Underfitting:** Neither — this is a well-fitted model. Training and test errors are close (gap of 0.2), both at reasonable levels.
- **Bias-Variance:** **Balanced** — low enough bias (captures the underlying pattern, not too simple) and low enough variance (stable generalization, small train-test gap)
- **Bias-Variance tradeoff:** The model is appropriately complex for 1000 samples and 5 features — enough capacity to fit the pattern without memorizing noise
- **Regularization:** Probably does not need regularization — the model generalizes well without it. Adding regularization might increase MSE_train slightly but would not significantly improve test MSE
- **Assessment:** This is a good model. The small train-test gap indicates low variance, and the absolute error level indicates acceptable bias.

**Degree-10 polynomial regression (MSE_train = 0.01, MSE_test = 147):**

- **Overfitting/Underfitting:** Severe **overfitting** — classic pattern of very low training error (0.01 ≈ near perfect fit) with catastrophically high test error (147 >> 2.3)
- **Bias-Variance:** **Low Bias, Very High Variance** — the polynomial model is flexible enough to fit the training data almost perfectly (low bias), but this flexibility means it captures noise and memorizes specific training patterns that don't appear in new data (very high variance)
- **Mechanism:** A degree-10 polynomial with 5 features creates an enormous number of features: all monomials up to degree 10. With 1000 samples, this could easily create more features than needed, giving the model more than enough capacity to memorize all training patterns
- **Train-test gap:** Gap of ~147 vs ~0.2 for linear regression — the polynomial model's test error is 60× the linear model's test error, despite having 200× lower training error
- **Regularization implications:**
  - **L2 (Ridge) regularization** would shrink all polynomial coefficients toward zero, reducing variance while accepting slightly higher bias. Would dramatically reduce overfitting.
  - **L1 (Lasso) regularization** would drive many polynomial coefficients to exactly zero — effectively performing feature selection among the polynomial terms, keeping only the most important ones
  - **Cross-validation** should be used to select the regularization strength λ
  - **Model selection:** Even with regularization, degree-10 polynomial may be overkill for this data — a lower-degree polynomial (degree 2 or 3) combined with cross-validation would likely perform better than heavily regularizing degree-10

---

## BONUS CHALLENGE QUESTIONS

---

**Question 11**
Cross-topic synthesis.

1. You have a dataset with 5 features and 4 observations (D=5, N=4). Explain from three angles why linear regression will fail: (a) algebraically using the Normal Equations, (b) geometrically using the column space interpretation, (c) statistically using overfitting theory.
2. Connect linear regression's Normal Equation (XᵀXw = Xᵀy) to the SVM's hard margin optimization. Both are solved as mathematical optimization problems — explain the fundamental differences in: (a) what they optimize, (b) the type of optimization problem, (c) what the solution represents geometrically.
3. A data scientist says: "I added 50 more features to my linear regression model and my R² went from 0.82 to 0.97. My model is now much better!" Write a complete response explaining every reason why this conclusion might be completely wrong, connecting concepts of overfitting, R² limitations, RSS, and the need for cross-validation.

### ✅ Answer

**Part 1:**
Why D=5, N=4 fails from three angles:

**(a) Algebraically — Normal Equations:**
The design matrix X has shape N×(D+1) = 4×6 (including bias column). The matrix XᵀX has shape (D+1)×(D+1) = 6×6. However, X has only 4 rows (N=4 observations), so rank(X) ≤ min(4, 6) = 4. Therefore XᵀX has rank ≤ 4, but it is a 6×6 matrix — rank is strictly less than 6, making it **singular (not invertible)**. The formula w* = (XᵀX)⁻¹Xᵀy cannot be computed because (XᵀX)⁻¹ does not exist.

**(b) Geometrically — Column Space:**
The column space of X is spanned by the 6 columns of X (bias + 5 features). But X has only 4 rows — in 4-dimensional observation space, at most 4 linearly independent directions exist. The column space Col(X) spans at most a 4-dimensional subspace of the 4-dimensional space ℝ⁴. In fact Col(X) ⊆ ℝ⁴ with dim(Col(X)) ≤ 4. With 6 columns in 4-dimensional space, the columns are necessarily linearly dependent (by dimension argument). The null space of X is non-trivial — there are weight vectors z ≠ 0 such that Xz = 0. Multiple weight vectors produce identical predictions. The orthogonal projection is onto a lower-dimensional subspace than the full parameter space — the solution is not unique.

**(c) Statistically — Overfitting:**
With D=5 features and N=4 observations, you have **more parameters (6 including bias) than data points (4)**. A system of 4 equations with 6 unknowns is underdetermined — infinitely many exact solutions exist. The model can achieve perfect fit (RSS = 0, R² = 1) on the training data simply by having enough free parameters — without learning any generalizable pattern. This is the most extreme overfitting: a model that perfectly interpolates 4 points with a 5-dimensional hyperplane memorizes the 4 training examples completely. Any new data point will be predicted based on extrapolation from this memorized surface, which in 5 dimensions can be wildly wrong. You have essentially zero degrees of freedom for error estimation.

**Part 2:**
Normal Equations vs. SVM Optimization:

**(a) What they optimize:**

**Linear Regression (Normal Equations):**
Minimizes **sum of squared errors** in the OUTPUT space:
min_w ||y - Xw||² = min_w Σₙ(yₙ - wᵀxₙ)²
Goal: Make predictions as close as possible to true labels. Every data point contributes equally to the loss.

**Hard Margin SVM:**
Maximizes **margin** in the INPUT space:
max_{w,b} 2/||w|| subject to tₙ(wᵀxₙ+b) ≥ 1 ∀n
Equivalently: min_{w,b} ||w||²/2 subject to constraints
Goal: Find the widest possible "road" between classes. Only support vectors determine the solution.

**(b) Type of optimization problem:**

**Linear Regression:** Unconstrained quadratic minimization — a smooth quadratic (parabola in multi-D) with no constraints. Solved by setting gradient to zero → linear system → closed-form solution in O(D³). No constraints to satisfy.

**Hard Margin SVM:** Constrained quadratic programming — minimize a quadratic objective (||w||²/2) subject to N linear constraints (one per training point). Requires a quadratic programming solver. The constraints make it harder — you need every training point on the correct side of the margin. Solved using Lagrange multipliers and the dual formulation.

**(c) Geometric representation of the solution:**

**Linear Regression solution:**
Geometrically: the hyperplane ŷ = wᵀx is the **orthogonal projection of y onto Col(X)**. The solution w* minimizes the distance between the target vector y and the prediction vector Xw in observation space. The residual is perpendicular to the feature subspace. Every training point influences the solution through its contribution to XᵀX and Xᵀy.

**Hard Margin SVM solution:**
Geometrically: the hyperplane wᵀx + b = 0 is the **maximum margin separator** — equidistant from the nearest positive and negative examples. The solution w* is determined ONLY by the support vectors (points on the margin). Non-support vectors have zero influence. The geometry is about the decision boundary in feature space, not about projections.

**Part 3:**
Complete response to "R² went from 0.82 to 0.97":

"Your conclusion that the model is 'much better' might be completely wrong. Here is a thorough explanation of every reason to be skeptical:

**1. R² always increases with more features — by mathematical necessity:**
RSS can never increase when adding features to a linear regression model. The worst case is that the new features' weights are set to zero (replicating the old model), and the best case is that they reduce RSS. Therefore R² = 1 - RSS/TSS can only stay the same or increase. Your improvement from 0.82 to 0.97 tells you nothing about whether the 50 features are genuinely useful — even 50 columns of pure random noise would increase R².

**2. Training R² is measuring training fit, not generalization:**
R² of 0.97 is measured on the same data the model was trained on. It tells you the model fits training data well — but we already knew that, because more parameters always fit training data better. What matters for a 'better model' is how it performs on new, unseen data. Training R² says nothing about this.

**3. Overfitting risk is extremely high:**
Adding 50 features to a model dramatically increases its parameter count. If your dataset is not enormous, this means the model now has many more parameters relative to data points — the classic recipe for overfitting. The model may be memorizing the training data (including noise) rather than learning genuine patterns. Evidence: the suspiciously high R² of 0.97 is consistent with a model that has memorized training data.

**4. RSS tells the same misleading story:**
Your RSS certainly decreased dramatically with 50 more features. But RSS is not a valid model comparison criterion between models of different complexity — it is biased toward more complex models regardless of their true predictive value.

**5. What you should have done instead:**
The only valid way to assess whether the 50 features genuinely improved the model is:
- **Cross-validation:** Evaluate both models (5 features vs. 55 features) using K-fold cross-validation. If the 55-feature model has lower cross-validated error, it genuinely generalizes better. If cross-validated error is worse or similar, the R² improvement was pure overfitting.
- **Adjusted R²:** A version that penalizes adding parameters — it only increases if the new feature's F-statistic exceeds 1.0 (i.e., genuinely informative beyond chance)
- **Held-out test set:** If you have a completely held-out test set that was never used in model development, comparing test MSE for both models would be definitive

**6. Summary:**
'Higher R² after adding features' is one of the most common and misleading statistical fallacies in data science. It exploits the fundamental flaw of using training-set metrics to make model selection decisions. Always validate on held-out data before claiming a model is 'better.'"

---

## Complete Answer Reference Table

| Question | Core Concepts | Difficulty | Exam Priority |
|----------|--------------|------------|---------------|
| Q1 | Problem formulation, simple linear model, geometric interpretation | Easy-Medium | Very High |
| Q2 | Least squares, MSE formula, outlier sensitivity | Medium | Very High |
| Q3 | 1D derivation, slope formula, statistical interpretation | Hard | High |
| Q4 | Matrix formulation, Normal Equations derivation | Hard | Very High |
| Q5 | Geometric interpretation, orthogonal projection, "Normal" | Hard | High |
| Q6 | Singular XᵀX, four conditions, null space | Hard | Very High |
| Q7 | Pseudoinverse, minimum norm solution, geometry | Very Hard | High |
| Q8 | RSS, R-squared, why more features is not better | Medium | Very High |
| Q9 | Bias term role, centroid property, bias-free problems | Medium | High |
| Q10 | Closed-form vs gradient descent, polynomial-kernel connection, full diagnosis | Very Hard | High |
| Q11 | Cross-topic synthesis | Very Hard | Medium |

---

## Top 6 Most Likely Exam Questions From This Topic

1. **Bias-Variance diagnosis** — given train/test MSE gap, identify overfitting/underfitting (mirrors Q2 from original exam exactly)
2. **L1 vs L2 Regularization** — what each does to parameters, when to use each (mirrors Q2 parts 2&3 from original exam)
3. **Normal Equations** — derive, explain XᵀX⁻¹Xᵀy, explain when it fails
4. **Geometric interpretation** — orthogonal projection, why residual ⊥ Col(X)
5. **MSE properties** — formula, advantages, disadvantage with outliers
6. **R-squared limitation** — why more features always increases R² and why this is misleading

**Send the next slides and I will build the complete exam for those topics too!**