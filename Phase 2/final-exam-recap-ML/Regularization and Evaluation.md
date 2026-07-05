- [[#Comprehensive Practice Exam WITH ANSWERS|Comprehensive Practice Exam WITH ANSWERS]]
	- [[#Comprehensive Practice Exam WITH ANSWERS#Regularization and Evaluation — Complete Question Bank|Regularization and Evaluation — Complete Question Bank]]
- [[#SECTION 1: Regularization Foundations|SECTION 1: Regularization Foundations]]
		- [[#Regularization and Evaluation — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Regularization and Evaluation — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Regularization and Evaluation — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 2: ElasticNet & Comparing Regularization Methods|SECTION 2: ElasticNet & Comparing Regularization Methods]]
		- [[#Regularization and Evaluation — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 3: Bias-Variance Decomposition|SECTION 3: Bias-Variance Decomposition]]
		- [[#Regularization and Evaluation — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Regularization and Evaluation — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 4: Regression Evaluation Metrics|SECTION 4: Regression Evaluation Metrics]]
		- [[#Regularization and Evaluation — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Regularization and Evaluation — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 5: Classification Evaluation Metrics|SECTION 5: Classification Evaluation Metrics]]
		- [[#Regularization and Evaluation — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Regularization and Evaluation — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Regularization and Evaluation — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 6: R² vs. Adjusted R²|SECTION 6: R² vs. Adjusted R²]]
		- [[#Regularization and Evaluation — Complete Question Bank#✅ Answer|✅ Answer]]
	- [[#SECTION 6: R² vs. Adjusted R²#BONUS CHALLENGE QUESTIONS|BONUS CHALLENGE QUESTIONS]]
		- [[#BONUS CHALLENGE QUESTIONS#✅ Answer|✅ Answer]]
	- [[#SECTION 6: R² vs. Adjusted R²#Complete Answer Reference Table|Complete Answer Reference Table]]
	- [[#SECTION 6: R² vs. Adjusted R²#Top 7 Most Likely Exam Questions From This Topic|Top 7 Most Likely Exam Questions From This Topic]]

# Comprehensive Practice Exam WITH ANSWERS
## Regularization and Evaluation — Complete Question Bank

---

# SECTION 1: Regularization Foundations

---

**Question 1**
Ordinary Least Squares fails in several important situations requiring regularization.

1. List and explain four specific situations where OLS fails. For each situation, explain the precise mechanism by which it leads to poor model performance.
2. Explain what "extremely large coefficients with large positive and negative values canceling each other out" means in the context of correlated predictors. Why does this happen mathematically and why is it dangerous?
3. Explain the general regularization framework: write the total error function E(w) = ED(w) + λEW(w). What does each term represent and what is the role of λ as a tradeoff parameter?

### ✅ Answer

**Part 1:**
Four situations where OLS fails:

**1. High-dimensional data (many features, D > N):**
When you have more features than observations, the design matrix X has more columns than rows. The system Xw = y is underdetermined — there are infinitely many weight vectors that fit the training data perfectly (RSS = 0). The OLS solution w* = (XᵀX)⁻¹Xᵀy doesn't exist because XᵀX is singular. Even if D is slightly less than N, with many features relative to N, the model overfits catastrophically — it memorizes training data but fails on new data.

**2. Strongly correlated predictors (multicollinearity):**
When features are highly correlated (e.g., income measured in dollars and income measured in thousands of dollars, or x and x²), the columns of X are nearly linearly dependent. XᵀX becomes nearly singular → (XᵀX)⁻¹ is numerically unstable → tiny changes in the data produce enormous changes in the estimated coefficients. The model is highly unstable and coefficients are essentially meaningless.

**3. Small datasets:**
With very few observations, OLS has high variance — the estimated coefficients are sensitive to the specific training examples used. A different small sample would give dramatically different coefficients. The model cannot reliably distinguish genuine signal from noise because there is insufficient data to estimate all parameters accurately. OLS uses all available degrees of freedom to fit the data, leaving nothing for uncertainty estimation.

**4. Ill-conditioned design matrices:**
Even with sufficient data, if the design matrix is ill-conditioned (condition number κ >> 1, common in polynomial regression or when features have very different scales), the matrix inversion is numerically unstable. Small rounding errors or perturbations in the data get amplified by the condition number, producing wildly incorrect coefficient estimates even when the mathematical solution exists in principle.

**Part 2:**
**Large canceling coefficients with correlated predictors:**

Suppose two features x₁ and x₂ are nearly identical (high correlation): x₂ ≈ x₁.

The model: ŷ = w₁x₁ + w₂x₂ ≈ (w₁ + w₂)x₁

The only thing that matters for predictions is the SUM (w₁ + w₂), not the individual values. OLS can achieve the same predictions with:
- w₁ = 1000, w₂ = -999 (sum = 1) ✓
- w₁ = -5000, w₂ = 5001 (sum = 1) ✓
- w₁ = 0.5, w₂ = 0.5 (sum = 1) ✓

All these produce identical predictions. OLS has no preference — it can assign any pair of enormous, opposite-sign values. In practice, small numerical differences between x₁ and x₂ determine which extreme solution OLS finds.

**Why this is dangerous:**
1. **Instability:** A tiny change in one training point shifts the balance between x₁ and x₂ slightly → the optimizer suddenly assigns w₁ = -5000, w₂ = 5001 instead of w₁ = 1000, w₂ = -999. The model changes dramatically from a tiny data perturbation.
2. **Uninterpretability:** The individual coefficients w₁ = 1000 and w₂ = -999 are meaningless — they tell you nothing about the actual effect of either feature.
3. **Generalization failure:** If x₁ and x₂ are slightly less correlated in new data, the enormous canceling coefficients will produce wildly wrong predictions instead of the stable (w₁+w₂)=1 relationship.

**Part 3:**
**General regularization framework:**

E(w) = ED(w) + λEW(w)

- **ED(w) = (1/2N)Σₙ(tₙ - y(xₙ,w))²:** The **data-dependent error** — measures how well the model fits training data (sum of squared residuals, MSE). We want this small to fit the data well.

- **EW(w) = (1/2)||w||²:** The **regularization term** — measures the complexity/magnitude of the weight vector. We want this small to keep the model simple and stable.

- **λ ≥ 0:** The **regularization coefficient** — the tradeoff dial:
  - λ = 0: Pure OLS — minimize only data fit, no complexity penalty. Can overfit.
  - Small λ: Data fit dominates. Model is flexible, may overfit.
  - Large λ: Regularization dominates. Model is simple, may underfit.
  - λ → ∞: Forces all weights to zero (except bias). Extreme underfitting.

The key insight: λ controls the **Bias-Variance tradeoff**:
- Increasing λ → increases bias (simpler model) → decreases variance (more stable)
- Optimal λ balances these opposing forces and is found by cross-validation

---

**Question 2**
Ridge Regression is the most common L2 regularization approach.

1. Write the Ridge Regression objective function in both the penalized form and the constrained form. Explain why these two formulations are equivalent and what t (the constraint bound) represents.
2. Derive the Ridge Regression closed-form solution. Show how adding λI modifies the Normal Equations and explain the eigenvalue argument for why this always produces an invertible matrix.
3. Explain why the intercept β₀ is excluded from the Ridge penalty. What specific problem would occur if you penalized the intercept?

### ✅ Answer

**Part 1:**
**Two equivalent formulations of Ridge Regression:**

**Penalized (Lagrangian) form:**
min_w RSS(w) + λΣⱼwⱼ²
= min_w Σₙ(yₙ - wᵀxₙ)² + λΣⱼ₌₁ᵖ wⱼ²

**Constrained form:**
min_w Σₙ(yₙ - wᵀxₙ)² subject to Σⱼ₌₁ᵖ wⱼ² ≤ t

**Why equivalent:**
By the Lagrangian duality of constrained optimization, for every value of the constraint bound t > 0, there exists a corresponding λ ≥ 0 such that the solution to the constrained problem equals the solution to the penalized problem. As t decreases (tighter constraint), the corresponding λ increases (stronger penalty). The two parameterizations are in one-to-one correspondence — they trace the same path of solutions as their parameters vary.

**What t represents:**
t is the maximum allowed squared magnitude of the coefficient vector ||w||². The constraint says: "find the best fit, but your coefficients cannot be too large — they must lie within a sphere of radius √t in weight space." Smaller t → smaller sphere → simpler model.

**Part 2:**
**Derivation of Ridge solution:**

Ridge objective in matrix form:
E(w) = (y - Xw)ᵀ(y - Xw) + λwᵀw

Take gradient with respect to w:
∂E/∂w = -2Xᵀ(y - Xw) + 2λw
= -2Xᵀy + 2XᵀXw + 2λw

Set to zero:
XᵀXw + λw = Xᵀy
**(XᵀX + λI)w = Xᵀy**

Solving:
**w_ridge = (XᵀX + λI)⁻¹Xᵀy**

**Eigenvalue argument for invertibility:**
Let XᵀX = QΛQᵀ (eigendecomposition, where Λ = diag(λ₁, λ₂, ..., λₚ) has eigenvalues ≥ 0).

Then: XᵀX + λI = Q(Λ + λI)Qᵀ = Q·diag(λ₁+λ, λ₂+λ, ..., λₚ+λ)·Qᵀ

The eigenvalues of XᵀX + λI are λᵢ + λ for each i.

Since λᵢ ≥ 0 (XᵀX is positive semidefinite) and λ > 0:
**λᵢ + λ ≥ λ > 0 for ALL i**

No eigenvalue is zero or negative → XᵀX + λI is **positive definite** → **always invertible** for any λ > 0.

This is true even when XᵀX is singular (has zero eigenvalues) — adding λ shifts every eigenvalue up by λ, guaranteeing strict positivity.

**Part 3:**
**Why the intercept is excluded from Ridge penalty:**

If we penalized the intercept β₀, the Ridge objective would be:
min_{β₀,w} Σₙ(yₙ - β₀ - wᵀxₙ)² + λ(β₀² + Σⱼwⱼ²)

**The specific problem:**
Suppose we shift all targets by a constant c: yₙ' = yₙ + c (adding the same constant to all observations). Intuitively, this should simply shift all predictions by c: the new intercept β₀' = β₀ + c, with all other coefficients unchanged.

But with β₀ penalized, the penalty λβ₀² increases when β₀ = β₀ + c is large. The optimizer is now penalized for having a large intercept, so it tries to shrink β₀ toward zero — which means it can no longer simply shift predictions by c. The fitted values are distorted to satisfy both the data fit and the (now large) intercept penalty.

**Result:** The procedure is no longer **equivariant under shifts of the target variable** — adding a constant c to all observations does not simply result in a shift of all predictions by c. The predictions depend on the arbitrary choice of origin for y, which is undesirable.

**The correct approach:**
Exclude β₀ from the penalty. Estimate β₀ = ȳ (the mean of targets) after centering, then apply Ridge only to the remaining slope coefficients. This ensures the procedure is invariant to shifts in y's origin.

---

**Question 3**
Lasso Regression introduces L1 regularization with fundamentally different properties.

1. Write the Lasso objective function. Explain why Lasso has no closed-form solution unlike Ridge. What type of computational problem must be solved instead?
2. Explain the concept of "soft thresholding" for Lasso. How does Lasso handle small coefficients differently from Ridge?
3. The slides describe Lasso as "continuous subset selection." Explain precisely what this means and why it is valuable.

### ✅ Answer

**Part 1:**
**Lasso objective function:**

min_w Σₙ(yₙ - wᵀxₙ)² + λΣⱼ|wⱼ|

Or in constrained form:
min_w Σₙ(yₙ - wᵀxₙ)² subject to Σⱼ|wⱼ| ≤ t

**Why no closed-form solution:**
The Lasso uses the **L1 norm** (sum of absolute values), which is **not differentiable at wⱼ = 0**. The absolute value function |wⱼ| has a sharp corner at 0 — the left derivative is -1 and the right derivative is +1, but the derivative does not exist at exactly 0.

Setting ∂E/∂wⱼ = 0 requires computing this derivative, which is undefined at the point of most interest (where sparsity occurs). We cannot write a simple linear system like Ridge's (XᵀX + λI)w = Xᵀy.

**Type of problem:**
Lasso is a **Quadratic Programming (QP) problem** — quadratic objective (squared residuals) with linear constraints (the L1 bound). Standard QP solvers can handle it. More practically, the **LARS algorithm (Least Angle Regression)** and **coordinate descent** methods compute the entire path of solutions as λ varies, with computational cost comparable to Ridge.

**Part 2:**
**Soft Thresholding:**

For orthonormal inputs (features uncorrelated with each other), the Lasso solution for each coefficient wⱼ is:

w_lasso_j = sign(w_OLS_j) × max(|w_OLS_j| - λ, 0)

This is called **soft thresholding**:
- If |w_OLS_j| > λ: Reduce the coefficient by λ toward zero (shrink by a constant amount)
- If |w_OLS_j| ≤ λ: Set the coefficient to **exactly zero** (eliminate the feature)

**Contrast with Ridge (scaling):**
w_ridge_j = w_OLS_j / (1 + λ)

Ridge scales each coefficient by the same factor 1/(1+λ) — every coefficient gets proportionally smaller but none reach exactly zero.

**Visual comparison:**

| w_OLS | Ridge (λ=1) | Lasso (λ=1) |
|-------|-------------|-------------|
| 3.0 | 1.5 | 2.0 |
| 1.5 | 0.75 | 0.5 |
| 0.8 | 0.4 | 0.0 |
| 0.3 | 0.15 | 0.0 |
| 0.1 | 0.05 | 0.0 |

Lasso eliminates small coefficients entirely; Ridge keeps all of them.

**Part 3:**
**Lasso as "continuous subset selection":**

**Traditional hard subset selection (Best Subset):**
Choose a subset of exactly k features and fit OLS using only those features. This is a discrete, combinatorial operation — a coefficient is either fully included (its OLS value) or completely excluded (set to zero). With p features, there are 2^p possible subsets to search.

**Lasso as "continuous" version:**
Lasso smoothly interpolates between these extremes:
- At λ = 0: All features included (OLS solution)
- As λ increases: Features are progressively eliminated one by one as their soft-thresholded values reach zero
- At λ = λ_max: All features excluded (all coefficients zero)

The path from all features to no features is **continuous** — controlled by a single smooth parameter λ, not a discrete subset choice. This is "continuous" subset selection.

**Why it is valuable:**

1. **Automatic feature selection:** Lasso simultaneously fits the model AND selects which features to include, eliminating irrelevant features automatically. No separate feature selection step needed.

2. **Computational tractability:** Finding the best subset of k features requires searching 2^p combinations (NP-hard for large p). Lasso finds a sparse solution in polynomial time through efficient algorithms.

3. **Regularization + selection together:** Unlike hard subset selection (which can overfit to the selected features), Lasso's soft thresholding also shrinks the selected features' coefficients — combining regularization with selection in one operation.

4. **High-dimensional applicability:** For problems with thousands of features (genomics, text data, image processing), hard subset selection is infeasible. Lasso can handle p >> N and produces sparse, interpretable models.

---

# SECTION 2: ElasticNet & Comparing Regularization Methods

---

**Question 4**
ElasticNet combines Ridge and Lasso.

1. Write the ElasticNet objective function. Explain the two penalty terms and what each contributes to the solution.
2. Explain the specific behavior of Lasso on highly correlated predictors that ElasticNet was designed to fix. Why does Lasso struggle with correlated features?
3. Create a comprehensive comparison table of OLS, Ridge, Lasso, and ElasticNet covering: penalty type, sparsity, closed-form solution, behavior with correlated features, and when to use each.

### ✅ Answer

**Part 1:**
**ElasticNet objective:**

min_w Σₙ(yₙ - wᵀxₙ)² + λ₁Σⱼ|wⱼ| + λ₂Σⱼwⱼ²

Or written with a single λ and mixing parameter α ∈ [0,1]:
min_w Σₙ(yₙ - wᵀxₙ)² + λ[α Σⱼ|wⱼ| + (1-α) Σⱼwⱼ²]

Where:
- **α = 1:** Pure Lasso (L1 only)
- **α = 0:** Pure Ridge (L2 only)
- **0 < α < 1:** ElasticNet — a weighted combination

**What each term contributes:**

**L1 term (λ₁Σ|wⱼ|):**
Promotes **sparsity** — drives some coefficients to exactly zero. Performs feature selection, eliminates truly irrelevant features. The "selection" component.

**L2 term (λ₂Σwⱼ²):**
Promotes **stability** — handles correlated features gracefully by distributing weight across a group rather than arbitrarily selecting one. Shrinks all coefficients, regularizes the numerical problem. The "regularization" component.

**Combined effect:**
ElasticNet has the selection property of Lasso (sparse solution, irrelevant features eliminated) AND the stability property of Ridge (correlated features handled gracefully, group selection instead of arbitrary single selection).

**Part 2:**
**Lasso's problem with correlated features:**

When two features x₁ and x₂ are highly correlated (ρ → 1), Lasso's behavior becomes arbitrary and unstable:

**The problem — random selection:**
Lasso tends to **arbitrarily select one feature from a correlated group and set all others to zero**. If x₁ and x₂ are perfectly correlated (x₂ = x₁), they are interchangeable — Lasso has no principled way to choose between them. Depending on tiny numerical differences in the data, it might select x₁ or x₂ — different runs with slightly different data give different feature selections.

**Why this is problematic:**
1. **Instability:** Small data perturbations flip which correlated feature is selected
2. **Missed information:** If x₁ and x₂ together capture a feature group (e.g., multiple correlated gene expressions all related to the same biological process), selecting only one misses the collective signal
3. **Uninterpretability:** You want to know "this group of related features matters" — not "feature 47 matters but feature 46 (nearly identical) doesn't"

**How ElasticNet fixes this:**
The L2 component of ElasticNet pushes correlated features to have similar coefficients — if x₁ and x₂ are highly correlated and both relevant, ElasticNet gives both non-zero (equal) coefficients instead of selecting one arbitrarily. The L1 component still eliminates truly irrelevant features. This is called the "grouping effect."

**Part 3:**
**Comprehensive comparison table:**

| Property | OLS | Ridge (L2) | Lasso (L1) | ElasticNet |
|----------|-----|-----------|-----------|-----------|
| Penalty | None | λΣwⱼ² | λΣ\|wⱼ\| | λ₁Σ\|wⱼ\|+λ₂Σwⱼ² |
| Constraint region | Unconstrained | Sphere (smooth) | Diamond (corners) | Rounded diamond |
| Sparsity | No | No | Yes (exact zeros) | Yes (exact zeros) |
| Closed-form | Yes | Yes | No (QP/coordinate descent) | No |
| Singular XᵀX | Fails | Always works | Works | Works |
| Correlated features | Unstable | Distributes weight equally | Arbitrarily picks one | Groups together |
| Feature selection | No | No | Yes (built-in) | Yes (built-in) |
| Solution type | w_OLS | Scaled OLS: w/(1+λ) | Soft-thresholded OLS | Both effects |
| Max features selected | p | p (all non-zero) | Min(N,p) | p |

**When to use each:**
- **OLS:** Clean data, N >> p, no multicollinearity, need exact solutions
- **Ridge:** Many correlated features, all features believed relevant, numerical stability needed
- **Lasso:** Many features with only a few truly relevant ones, need interpretable sparse model, p > N
- **ElasticNet:** Many correlated features AND you want sparsity, p >> N with correlated feature groups, when Lasso gives unstable results

---

# SECTION 3: Bias-Variance Decomposition

---

**Question 5**
The Bias-Variance decomposition provides the theoretical foundation for understanding model performance.

1. Derive the Bias-Variance decomposition from first principles. Starting from E[(y(x;D) - h(x))²], show how adding and subtracting ED[y(x;D)] leads to the separation into bias² and variance terms.
2. Define each term in the final decomposition: expected loss = bias² + variance + noise. Give an intuitive interpretation of each term and explain what determines its magnitude.
3. Explain why noise is called the "irreducible" component. What does this mean practically for a data scientist trying to minimize prediction error?

### ✅ Answer

**Part 1:**
**Derivation of Bias-Variance decomposition:**

**Setup:** 
- h(x) = E[t|x]: The true regression function (conditional expectation)
- y(x;D): Our model trained on dataset D
- We consider many datasets D of size N drawn from the same distribution

**Starting point — expected loss for a specific x:**

EL = E_D[(y(x;D) - h(x))²]

**Add and subtract the average prediction ED[y(x;D)]:**

Let ȳ(x) = E_D[y(x;D)] (average prediction across all possible training datasets)

EL = E_D[(y(x;D) - ȳ(x) + ȳ(x) - h(x))²]

**Expand the square:**
= E_D[(y(x;D) - ȳ(x))² + 2(y(x;D) - ȳ(x))(ȳ(x) - h(x)) + (ȳ(x) - h(x))²]

**The cross term vanishes:**
E_D[2(y(x;D) - ȳ(x))(ȳ(x) - h(x))] = 2(ȳ(x) - h(x))E_D[y(x;D) - ȳ(x)]
= 2(ȳ(x) - h(x)) × 0 = 0

(Because E_D[y(x;D) - ȳ(x)] = ȳ(x) - ȳ(x) = 0 by definition of ȳ(x))

**Remaining terms:**
EL = E_D[(y(x;D) - ȳ(x))²] + (ȳ(x) - h(x))²

= **Variance** + **Bias²**

**Adding noise** (from the irreducible variation in t around h(x)):

Full expected loss = **Bias²(x) + Variance(x) + Noise(x)**

Where:
- Bias²(x) = (ȳ(x) - h(x))² = (E_D[y(x;D)] - h(x))²
- Variance(x) = E_D[(y(x;D) - ȳ(x))²]
- Noise(x) = E[(h(x) - t)²] = Var(t|x)

**Part 2:**
**Interpretation of each term:**

**Bias²:**
Measures how far the **average** prediction (averaged over all possible training datasets) is from the true function h(x).

- High bias: The model is systematically wrong in the same direction, regardless of which specific training data was used. A linear model fitting truly nonlinear data has high bias — it can never capture the curve.
- What determines it: Model architecture and assumptions. Linear models have high bias for nonlinear data. Simple models have higher bias.
- Analogy: A systematic measurement instrument error — the thermometer always reads 5° too high regardless of what data it was trained on.

**Variance:**
Measures how much the predictions **vary** across different training datasets of the same size.

- High variance: The model learns different functions from different training samples — it is unstable, sensitive to the specific training data used. An overfit model has high variance — small changes in training data cause large changes in predictions.
- What determines it: Model complexity and training set size. Complex models on small datasets have high variance.
- Analogy: A noisy measurement — each measurement gives a different result, but on average they might be correct.

**Noise (Irreducible error):**
The inherent randomness in the data that **no model can eliminate** — the variance of t around its conditional mean h(x).

- This exists because the target t = h(x) + ε, where ε is random variation from unmodeled factors, measurement error, fundamental randomness
- No matter how perfect the model, it cannot predict better than h(x)
- The noise term is Var(t|x) — how much t varies around the true regression function

**Part 3:**
**Why noise is "irreducible":**

Noise = Var(t|x) = E[(h(x) - t)²]

This represents variation in the target variable t that is **not determined by the input x** — it comes from:
1. Measurement error in recording t
2. Factors affecting t that are not included as features
3. Fundamental randomness in the process generating t

No model, no matter how complex or clever, can predict what is fundamentally random. Even a perfect oracle that knew the exact true function h(x) would still make errors equal to the noise level.

**Practical implications for a data scientist:**

1. **Sets the performance floor:** If noise = 0.5 MSE, the best any model can ever achieve is MSE = 0.5 on this data. More data, better algorithms, more features — nothing can beat this floor.

2. **Diagnostic tool:** If your model achieves MSE = 2.0 on test data and you know noise ≈ 1.8, your model is nearly optimal — most of the error is irreducible. Don't waste time trying to improve it further.

3. **Feature engineering insight:** Noise can be REDUCED by adding better features. If unmodeled factors (currently contributing to "noise") can be measured and included as features, they move from irreducible noise to explainable signal. Example: adding weather data reduces the "noise" in ice cream sales prediction.

4. **When to stop:** If test error ≈ noise level, the model has essentially solved the problem. Further complexity only risks overfitting without reducing the irreducible floor.

---

**Question 6**
Regularization directly controls the Bias-Variance tradeoff.

1. Explain precisely how increasing λ in Ridge regression affects Bias and Variance. Walk through the mechanism for each direction of change.
2. The slides state: "Very flexible models have low bias and high variance. Relatively rigid models have high bias and low variance." Map specific λ values in Ridge to positions on this spectrum.
3. Connect the theoretical Bias-Variance tradeoff to the practical observation that training error decreases while test error first decreases then increases as model complexity increases.

### ✅ Answer

**Part 1:**
**Effect of increasing λ on Bias and Variance:**

**Effect on Variance (decreases with increasing λ):**

Mechanism: Larger λ → stronger penalty on weight magnitude → weights are shrunk closer to zero → the model becomes less sensitive to individual training points.

Specifically: w_ridge = (XᵀX + λI)⁻¹Xᵀy. As λ increases, (XᵀX + λI)⁻¹ becomes dominated by λI rather than XᵀX → weights are increasingly determined by the regularization, not the data.

Result: If you train on a different dataset D', the weights change less (the regularization is a constant force pulling toward zero, overriding data-specific variation). The variance of predictions across different training datasets decreases.

**Effect on Bias (increases with increasing λ):**

Mechanism: Larger λ → weights shrunk more aggressively toward zero → the model is forced to be simpler than the data might support → systematic underfitting.

Specifically: If the true optimal weight for feature j is w*ⱼ = 5, Ridge shrinks it toward: w_ridge_j ≈ w_OLS_j/(1+λ). For large λ: w_ridge_j ≈ w_OLS_j/λ → 0. The model is forced to underestimate all effects.

Result: The average prediction across different datasets deviates more from the true function h(x). The model is systematically wrong — biased toward predicting values closer to zero/mean.

**Summary:**
- λ ↑ → Variance ↓ (more stable, less sensitive to data)
- λ ↑ → Bias² ↑ (systematically wrong, too simple)
- Optimal λ* minimizes Bias² + Variance (total expected loss)

**Part 2:**
**Mapping λ to Bias-Variance spectrum:**

**λ = 0 (pure OLS):**
- Bias: Lowest possible (for linear models) — no constraint forces the model away from the optimal fit
- Variance: Highest — coefficients can be anything the data supports, including memorizing noise
- Position: Low Bias, High Variance → overfitting regime for complex data

**Small λ (e.g., 0.001):**
- Bias: Slightly above OLS (negligible constraint)
- Variance: Slightly below OLS (small stability improvement)
- Position: Still close to low bias, high variance end

**Optimal λ* (cross-validated):**
- Bias: Moderate — some systematic error accepted
- Variance: Moderate — stability significantly improved
- Position: At the minimum of total expected loss — the sweet spot

**Large λ (e.g., 1000):**
- Bias: High — model is forced to be very simple, most coefficients nearly zero
- Variance: Low — predictions barely change across different training datasets
- Position: High Bias, Low Variance → underfitting regime

**λ → ∞:**
- Bias: Maximum — all coefficients forced to zero, model predicts only intercept (ȳ)
- Variance: Minimum — predictions are constant regardless of training data
- Position: Maximum bias, zero variance — completely useless model

**Part 3:**
**Connecting theory to practice:**

**Training error decreases monotonically with complexity (λ decreasing):**
As the model becomes more flexible (λ decreases):
- More complex models can fit any training pattern
- The model minimizes training RSS more effectively
- At λ=0 with enough parameters, training error approaches zero

This corresponds to decreasing Bias (the model can represent more functions) dominating the training loss. Training error measures only how well the model memorizes training data — it does not penalize variance.

**Test error first decreases then increases:**

**Initial decrease (as λ decreases from large values):**
Starting from high λ (highly regularized, underfitting), decreasing λ reduces Bias — the model captures more genuine signal in the data. The reduction in Bias² dominates the increase in Variance at this stage. Test error decreases as the model becomes more accurate on average.

**Eventually increases (as λ continues to decrease):**
As λ decreases further past the optimal point:
- Bias² continues to decrease (but is already small — approaching the noise floor)
- Variance increases rapidly — the model starts memorizing training noise
- The increase in Variance now dominates the (already small) decrease in Bias²
- Test error increases: predictions change dramatically for new data because the model learned training-specific noise patterns

**The optimal λ (minimum test error):**
Occurs where the decrease in Bias² from reducing λ exactly balances the increase in Variance. This is the Bias-Variance tradeoff in action — the theoretical minimum of Bias² + Variance + Noise corresponds exactly to the minimum of the empirical test error curve.

---

# SECTION 4: Regression Evaluation Metrics

---

**Question 7**
Multiple regression evaluation metrics exist with different properties.

1. Write the formulas for MAE, MSE, and RMSE. For each, explain: (a) what it measures, (b) its units, (c) its key advantage, (d) its key disadvantage.
2. Explain why MAE is "most robust to outliers" while MSE is not. Give a concrete numerical example demonstrating the difference in how each handles an extreme outlier.
3. MSE is differentiable but MAE is not. Explain why differentiability matters critically for optimization-based training. What specific problem does non-differentiability at zero cause for gradient-based methods?

### ✅ Answer

**Part 1:**
**Formulas and properties:**

**MAE (Mean Absolute Error):**
MAE = (1/N) Σₙ |yₙ - ŷₙ|

| Property | Detail |
|----------|--------|
| Measures | Average absolute prediction error |
| Units | Same as output variable y |
| Key advantage | Robust to outliers; interpretable (same units as y) |
| Key disadvantage | Not differentiable at 0 — cannot use standard gradient descent directly |

**MSE (Mean Squared Error):**
MSE = (1/N) Σₙ (yₙ - ŷₙ)²

| Property | Detail |
|----------|--------|
| Measures | Average squared prediction error |
| Units | Squared units of output (y²) — not directly interpretable |
| Key advantage | Differentiable everywhere — ideal as a loss function for gradient-based optimization |
| Key disadvantage | Sensitive to outliers (squares large errors, amplifying them); units are y² |

**RMSE (Root Mean Squared Error):**
RMSE = √(MSE) = √((1/N) Σₙ (yₙ - ŷₙ)²)

| Property | Detail |
|----------|--------|
| Measures | Square root of average squared error |
| Units | Same as output variable y (interpretable) |
| Key advantage | Interpretable units (like MAE) while retaining MSE's differentiability advantage |
| Key disadvantage | Still sensitive to outliers (inherited from MSE); penalizes large errors more than MAE |

**Part 2:**
**Outlier robustness — MAE vs. MSE:**

**Dataset without outlier:** True values [10, 12, 11, 13], Predictions [10, 10, 10, 10]
- Errors: [0, 2, 1, 3]
- MAE = (0 + 2 + 1 + 3)/4 = **1.5**
- MSE = (0 + 4 + 1 + 9)/4 = **3.5**

**Same dataset WITH extreme outlier:** Replace last true value with 100
True values [10, 12, 11, 100], Predictions [10, 10, 10, 10]
- Errors: [0, 2, 1, 90]
- MAE = (0 + 2 + 1 + 90)/4 = **23.25** (15× increase)
- MSE = (0 + 4 + 1 + 8100)/4 = **2026.25** (579× increase!)

**Why the enormous difference:**
- MAE treats the outlier error of 90 as just 90 — linearly proportional
- MSE squares the outlier error: 90² = 8100 — the outlier's contribution is 8100/4 = 2025, dwarfing all other contributions

The squaring in MSE causes the outlier to contribute quadratically to the total error. The same principle that makes large errors "more important" in fitting (which is sometimes desirable) makes the metric extremely sensitive to outliers in evaluation.

**MAE's robustness:**
MAE treats a 90-unit error as 45× more important than a 2-unit error. MSE treats the 90-unit error as 90²/2² = 2025× more important. MAE's linear treatment means outliers don't dominate the metric — they contribute proportionally to their magnitude.

**Part 3:**
**Why differentiability matters for optimization:**

Gradient-based optimization (gradient descent and its variants) requires computing the **gradient** (derivative) of the loss function with respect to the model parameters:

w ← w - η × ∂L/∂w

For MSE:
∂MSE/∂w = -(2/N) Xᵀ(y - Xw) — smooth, defined everywhere

For MAE:
∂MAE/∂wⱼ = -(1/N) Σₙ xₙⱼ × sign(yₙ - ŷₙ)

**The problem at exactly zero error:**
sign(0) is undefined — the subgradient is the interval [-1, +1]. At exactly zero residual, the "gradient" could point in any direction between -1 and +1. Standard gradient descent doesn't know which direction to step.

**Practical consequences:**
1. **Cannot use standard gradient descent:** Algorithms require a single well-defined gradient direction at each step. The non-differentiability of |x| at 0 means standard gradient descent can oscillate around a minimum without converging.
2. **Requires specialized algorithms:** Subgradient methods, proximal algorithms, or smooth approximations to |x| (like Huber loss) must be used.
3. **Slower convergence:** Even with specialized algorithms, non-smooth objectives converge more slowly than smooth ones.
4. **Computational overhead:** The handling of non-differentiable points requires special-case logic that adds implementation complexity.

This is why MSE is the standard training loss for regression despite MAE being more robust — MSE's smooth quadratic landscape enables simple, fast, reliable gradient-based optimization with guaranteed convergence.

---

**Question 8**
RMSLE and MAPE are specialized metrics for specific situations.

1. Write the RMSLE formula. Explain when you would choose RMSLE over RMSE and what the "larger penalty for underestimation" property means practically.
2. Explain the MAPE metric. What does it measure that MSE cannot, and what are the two critical failure modes of MAPE?
3. You are predicting house prices ranging from $50,000 to $5,000,000. A model predicts $95,000 for a $100,000 house and $950,000 for a $1,000,000 house. Calculate the MSE, MAE, and RMSLE for these two predictions. Which metric best captures that both errors are "equally good" proportionally?

### ✅ Answer

**Part 1:**
**RMSLE Formula:**
RMSLE = √((1/N) Σₙ (log(ŷₙ + 1) - log(yₙ + 1))²)

The +1 offset prevents errors when predictions or true values are zero.

**When to choose RMSLE over RMSE:**

1. **Skewed distributions:** When target values span many orders of magnitude (e.g., house prices: $50K to $50M, population sizes, sales quantities). For such data, RMSE is dominated by large values — a $1M error on a $50M property is similar to a $1M error on a $100K property in RMSE, even though the second is catastrophically proportionally worse.

2. **Relative error matters more than absolute:** When a 10% prediction error is equally serious whether the true value is $100K or $10M. RMSLE captures proportional errors rather than absolute errors.

3. **Exponential growth data:** Time series with exponential trends, where errors naturally grow with the magnitude of the target.

**Larger penalty for underestimation — practical meaning:**

For RMSLE: log(ŷ+1) - log(y+1) = log((ŷ+1)/(y+1))

If ŷ < y (underestimation): log(ŷ/y) < 0, the ratio is <1
If ŷ > y (overestimation): log(ŷ/y) > 0, the ratio is >1

For a fixed proportional error magnitude |ŷ/y - 1| = e:
- Underestimation by e: |log(1-e)| > |log(1+e)| (more negative)
- Overestimation by e: |log(1+e)| < |log(1-e)|

So RMSLE penalizes under-prediction more than over-prediction for the same proportional error.

**Practical example:**
True value = 100. Predict 50 (underestimate 50%) vs predict 150 (overestimate 50%):
- Underestimation: log(51/101) = log(0.505) ≈ -0.684
- Overestimation: log(151/101) = log(1.495) ≈ 0.402

RMSLE penalizes the underestimation more severely.

**When this asymmetry is appropriate:** In supply chain (predicting demand): underestimating demand means running out of stock (lost sales, customer dissatisfaction) — often worse than overestimating (excess inventory). In medical dosing: underdosing is often more dangerous than overdosing.

**Part 2:**
**MAPE (Mean Absolute Percentage Error):**
MAPE = (100/N) Σₙ |yₙ - ŷₙ| / |yₙ|

**What it measures that MSE cannot:**
MAPE expresses error as a **percentage of the true value** — it is scale-independent. A 10% error means the same thing whether the true value is $100 or $100,000. This makes MAPE directly interpretable to business stakeholders ("we're wrong by 8% on average") and allows comparison of model performance across datasets with different scales or units.

**Two critical failure modes:**

1. **Undefined when yₙ = 0:**
The formula divides by |yₙ|. If any true value is exactly zero, MAPE → ∞. This makes MAPE unusable for count data (where zeros are common), demand forecasting (zero-demand periods), or any dataset with zero values in the target.

2. **Extreme sensitivity to small true values:**
When yₙ is very small (close to but not zero), even a small absolute error produces an enormous percentage error. Example: yₙ = 0.001, ŷₙ = 0.002 → MAPE contribution = |0.001|/0.001 = 100% — a tiny absolute error inflates MAPE catastrophically. One or two small true values can make MAPE meaningless as an aggregate metric.

**Part 3:**
**Calculations for house price predictions:**

Prediction 1: ŷ₁ = $95,000, y₁ = $100,000
- Error: $5,000 (5% underestimate)

Prediction 2: ŷ₂ = $950,000, y₂ = $1,000,000
- Error: $50,000 (5% underestimate)

**MSE:**
= (5000² + 50000²)/2 = (25,000,000 + 2,500,000,000)/2 = **1,262,500,000 $/²**

The second prediction contributes 100× more to MSE despite being the same 5% error! MSE treats proportionally identical errors very differently based on absolute magnitude.

**MAE:**
= (5000 + 50000)/2 = **$27,500**

The second prediction contributes 10× more to MAE. Still not capturing "equal proportional error."

**RMSLE:**
- Prediction 1: log(95001) - log(100001) = log(95001/100001) = log(0.95) ≈ -0.0513
- Prediction 2: log(950001) - log(1000001) = log(950001/1000001) = log(0.95) ≈ -0.0513

Squared: 0.00263 each
RMSLE = √((0.00263 + 0.00263)/2) = √0.00263 ≈ **0.0513 (about 5%)**

**RMSLE best captures proportional equality:**
Both predictions are 5% underestimates. RMSLE gives them identical contributions — exactly 0.0513 each — correctly recognizing that both are "equally good" relative to their respective scales. MSE and MAE both treat the larger-scale error as much more important, despite both being identical 5% errors.

---

# SECTION 5: Classification Evaluation Metrics

---

**Question 9**
The confusion matrix underlies all classification metrics.

1. Draw and label the full confusion matrix for binary classification. Define TP, FP, TN, FN precisely. Give a concrete medical diagnosis example for each cell.
2. Derive the formulas for Accuracy, Precision, Recall, and Specificity from the confusion matrix. Explain what each measures in plain English.
3. Explain why Accuracy is a misleading metric for imbalanced datasets. Give a concrete example where a model achieves 99% accuracy but is completely useless.

### ✅ Answer

**Part 1:**
**Confusion Matrix:**

```
                    PREDICTED
                  Positive  Negative
ACTUAL  Positive |   TP    |   FN   |
        Negative |   FP    |   TN   |
```

**Definitions with medical diagnosis example (disease detection):**

- **TP (True Positive):** Model predicts POSITIVE, truth is POSITIVE. ✓ Patient has disease and model correctly flags them. "Correctly diagnosed sick patient."

- **FP (False Positive):** Model predicts POSITIVE, truth is NEGATIVE. ✗ Patient is healthy but model wrongly flags them. "Healthy patient falsely told they have disease" — causes unnecessary anxiety and follow-up tests.

- **TN (True Negative):** Model predicts NEGATIVE, truth is NEGATIVE. ✓ Patient is healthy and model correctly clears them. "Correctly cleared healthy patient."

- **FN (False Negative):** Model predicts NEGATIVE, truth is POSITIVE. ✗ Patient has disease but model misses it. "Sick patient wrongly told they're healthy" — disease goes untreated.

**Part 2:**
**Formulas and plain English:**

**Accuracy:**
= (TP + TN) / (TP + FP + TN + FN)
= "Of all patients examined, what fraction did we classify correctly (either correctly diagnosed as sick OR correctly cleared as healthy)?"

**Precision (Positive Predictive Value):**
= TP / (TP + FP)
= "Of all patients I PREDICTED as sick, what fraction actually HAD the disease?"
= "How trustworthy is a positive prediction?"

**Recall (Sensitivity / True Positive Rate):**
= TP / (TP + FN)
= "Of all patients who ACTUALLY had the disease, what fraction did I correctly identify?"
= "What fraction of actual sick patients did I catch?"

**Specificity (True Negative Rate):**
= TN / (TN + FP)
= "Of all patients who were actually HEALTHY, what fraction did I correctly identify as healthy?"
= "How good am I at clearing innocent (healthy) patients?"

**Part 3:**
**Why Accuracy fails for imbalanced datasets:**

**Concrete example — rare disease detection (1% prevalence):**

Suppose 10,000 patients are tested: 100 have the disease, 9,900 are healthy.

**Strategy: Predict "Healthy" for EVERYONE:**
- TP = 0 (caught zero sick patients)
- FP = 0 (never incorrectly flagged healthy patients as sick)
- TN = 9,900 (correctly identified all healthy patients)
- FN = 100 (missed all 100 sick patients)

**Accuracy = (0 + 9,900) / 10,000 = 99%**

The model achieves 99% accuracy while being **completely useless** — it catches exactly zero sick patients. Every single sick person is sent home untreated.

**Why accuracy lies:**
Accuracy rewards correct predictions equally regardless of class. In imbalanced datasets, simply predicting the majority class always earns high accuracy. A model that always says "not spam" achieves 99% accuracy on a dataset with 1% spam — while being a completely failed spam filter.

**What you should use instead:**
- **Precision and Recall** separately (and their F1 combination)
- **AUC-ROC** which evaluates across all thresholds
- **Balanced accuracy** = (Recall + Specificity) / 2

---

**Question 10**
Precision, Recall, F1 Score, and the Precision-Recall tradeoff.

1. Explain the fundamental tradeoff between Precision and Recall. Why is it generally impossible to maximize both simultaneously? Describe what happens to each as you lower the classification threshold.
2. The slides give specific domain examples: cancer detection (high Recall) vs. wine quality (high Precision). Explain the reasoning for each choice using the concepts of false positives and false negatives.
3. Explain the F1 Score. Why is it the harmonic mean rather than the arithmetic mean of Precision and Recall? Give a concrete example showing why arithmetic mean would be misleading.

### ✅ Answer

**Part 1:**
**The Precision-Recall tradeoff:**

A classification model produces a **probability score** (e.g., 0.0 to 1.0) for each example. A threshold τ converts this to a binary prediction: if score > τ → Positive, else → Negative.

**As threshold τ decreases (predict Positive more aggressively):**
- More examples are classified as Positive
- **Recall increases:** We catch more actual positives (fewer false negatives) — it's harder to miss a positive when you're willing to call almost everything positive
- **Precision decreases:** Among all predicted positives, more are actually negative (more false positives) — you're being less selective

**As threshold τ increases (predict Positive more conservatively):**
- Fewer examples are classified as Positive
- **Precision increases:** The few predictions you make as Positive are more likely correct (fewer false positives) — being selective improves accuracy of positive predictions
- **Recall decreases:** You miss more actual positives (more false negatives) — being too conservative means sick patients are missed

**Why impossible to maximize both simultaneously:**
At threshold τ = 0 (everything Positive): Recall = 1.0, Precision ≈ class prevalence (low for rare classes)
At threshold τ = 1 (nothing Positive): Precision = undefined (0/0), Recall = 0.0

Moving τ through [0,1] traces a Precision-Recall curve. The curve typically rises then falls — there is no single τ that achieves both Precision=1 and Recall=1 (unless the model is perfect). Every increase in one comes at the cost of the other.

**Part 2:**
**Domain-specific metric choices:**

**Cancer detection → Prioritize HIGH RECALL:**

The critical error is a **False Negative** — a sick patient told they are healthy. Consequences:
- Patient goes untreated while cancer progresses
- By the time the cancer is detected (at next screening or symptom onset), it may be inoperable
- Patient may die from a treatable condition
- This is irreversible harm

The False Positive cost (healthy patient told they might have cancer):
- Causes anxiety and stress — serious but manageable
- Leads to follow-up tests (biopsy, etc.) — inconvenient and costly
- Eventually resolved: "Good news, the follow-up test showed no cancer"
- Reversible harm

**Decision:** The cost of FN >> cost of FP. Prioritize Recall — catch as many true positives as possible, even at the cost of many false alarms.

**Wine quality classification → Prioritize HIGH PRECISION:**

Scenario: A winery uses a model to award a "Premium" label (used for marketing and higher price).

The critical error is a **False Positive** — mediocre wine labeled Premium.
- Customers pay a premium price for inferior wine
- Brand reputation is damaged when customers realize it's not actually premium quality
- Long-term business damage

The False Negative cost (good wine not labeled Premium):
- Good wine is sold as standard wine at lower price
- Lost revenue opportunity
- But no active harm to customers

**Decision:** The cost of FP >> cost of FN. Prioritize Precision — when you say something is Premium, it should genuinely be Premium, even if you miss some good wines.

**Part 3:**
**F1 Score and the harmonic mean:**

**Formula:**
F1 = 2 × (Precision × Recall) / (Precision + Recall)
= 2TP / (2TP + FP + FN)

This is the **harmonic mean** of Precision and Recall.

**Why harmonic mean and not arithmetic mean:**

The harmonic mean is always ≤ arithmetic mean, and it is especially sensitive to low values in either term. A single very low value drags the harmonic mean down dramatically, even if the other value is high.

**Concrete example showing arithmetic mean would be misleading:**

**Model A:** Precision = 0.90, Recall = 0.90
- F1 (harmonic) = 2×(0.9×0.9)/(0.9+0.9) = **0.90**
- Arithmetic mean = (0.9+0.9)/2 = **0.90**

**Model B:** Precision = 0.99, Recall = 0.10
- F1 (harmonic) = 2×(0.99×0.10)/(0.99+0.10) = 0.198/1.09 = **0.182**
- Arithmetic mean = (0.99+0.10)/2 = **0.545**

Model B predicts Positive for almost nothing (Recall=0.10 — misses 90% of actual positives). This model is essentially useless for finding true positives. Yet its arithmetic mean of 0.545 makes it appear decent.

The **harmonic mean correctly gives Model B an F1 of 0.182** — reflecting that a model with 10% recall is failing badly at the core task of finding positives, regardless of how precise it is on the rare cases it does predict positive.

**The harmonic mean's logic:** To get a high F1, you need BOTH Precision and Recall to be reasonably high. A model can't "cheat" F1 by excelling on one metric while failing on the other — the harmonic mean ensures the weaker metric dominates.

---

**Question 11**
ROC Curves and AUC provide threshold-independent evaluation.

1. Explain what a ROC Curve is. Define TPR (True Positive Rate) and FPR (False Positive Rate). What does each axis of the ROC Curve represent?
2. Explain what AUC measures intuitively. What do AUC = 1.0, AUC = 0.5, and AUC < 0.5 each indicate about the classifier?
3. Explain why ROC curves are "particularly useful when datasets are imbalanced" and "when classification thresholds must be tuned." Contrast this with why Accuracy fails in these same situations.

### ✅ Answer

**Part 1:**
**ROC Curve definition:**

A ROC (Receiver Operating Characteristic) Curve is a graphical plot that shows classifier performance across **all possible classification thresholds simultaneously**.

**TPR (True Positive Rate = Recall = Sensitivity):**
TPR = TP / (TP + FN)
"Of all actual positives, what fraction did I correctly identify?"
→ **Y-axis of ROC Curve** — we want this high

**FPR (False Positive Rate):**
FPR = FP / (FP + TN)
"Of all actual negatives, what fraction did I incorrectly call positive?"
→ **X-axis of ROC Curve** — we want this low

**What each axis represents:**
- **X-axis (FPR):** How often the model raises a false alarm — incorrectly classifying a negative as positive. Values from 0 (never false alarm) to 1 (always false alarm for negatives).
- **Y-axis (TPR):** How often the model correctly catches a positive — sensitivity. Values from 0 (catches nothing) to 1 (catches everything).

**How the curve is drawn:**
As the classification threshold τ sweeps from 1.0 (most conservative) to 0.0 (most liberal):
- At τ = 1: No predictions positive → TPR = 0, FPR = 0 (bottom-left corner)
- As τ decreases: More predictions become positive → both TPR and FPR increase
- At τ = 0: All predictions positive → TPR = 1, FPR = 1 (top-right corner)

The resulting curve traces the Sensitivity-Specificity tradeoff across all thresholds.

**Part 2:**
**AUC (Area Under the Curve) interpretation:**

AUC = Area under the ROC Curve = integral from FPR=0 to FPR=1

**Probabilistic interpretation:** AUC = the probability that, given a randomly chosen positive example and a randomly chosen negative example, the model assigns a higher score to the positive than the negative.

**AUC = 1.0 (Perfect classifier):**
The ROC curve hugs the top-left corner — TPR = 1 is achieved at FPR = 0. The model perfectly separates all positives from all negatives at some threshold τ. Every positive example gets a higher score than every negative example.

**AUC = 0.5 (Random guessing):**
The ROC curve is the diagonal line from (0,0) to (1,1). The model has no discriminative ability — for any threshold, TPR ≈ FPR. The model randomly assigns scores with no relationship to true class. Equivalent to flipping a fair coin.

**AUC < 0.5 (Worse than random):**
The ROC curve falls below the diagonal. The model is "inverted" — it assigns higher scores to actual negatives than positives. Paradoxically, you can improve this model by simply inverting all its predictions (and you'd get AUC > 0.5). This often indicates a labeling error or a fundamental modeling mistake.

**Part 3:**
**Why ROC/AUC excels where Accuracy fails:**

**For imbalanced datasets:**

Recall: With 1% positive prevalence, predicting "all negative" gives 99% Accuracy but is useless.

**Accuracy fails because:** It weights all correct predictions equally — 9,900 correct negatives count the same as 100 correct positives. The majority class dominates.

**ROC/AUC succeeds because:** TPR = TP/(TP+FN) measures performance relative to the positive class size. FPR = FP/(FP+TN) measures relative to the negative class size. Each rate is computed within its own class — a model that gets 0/100 positives correct has TPR = 0 regardless of how many negatives it correctly classifies. Class imbalance doesn't distort these rates.

**For threshold tuning:**

**Accuracy at a fixed threshold fails because:** The "natural" threshold of 0.5 is arbitrary and often wrong for imbalanced data or asymmetric cost situations.

**ROC/AUC succeeds because:** The ROC curve shows performance at ALL thresholds simultaneously. A practitioner can:
1. Look at the entire curve to understand the TPR-FPR tradeoff
2. Select the threshold that achieves a required TPR (e.g., "catch at least 95% of positives") and read off the corresponding FPR
3. Choose different thresholds for different deployment contexts (high-stakes screening vs. low-stakes filtering)

AUC additionally provides a single threshold-independent number summarizing overall discriminative ability — useful for comparing models without committing to a specific threshold.

---

# SECTION 6: R² vs. Adjusted R²

---

**Question 12**
R² and Adjusted R² are important regression evaluation metrics with critical differences.

1. Write both formulas. Explain every component of each formula and the conceptual difference between them.
2. Explain the fundamental flaw of R² that Adjusted R² corrects. Show algebraically why R² can only increase when adding features.
3. When would R² decrease as you add a feature? When would Adjusted R² decrease? Explain the conditions for each.

### ✅ Answer

**Part 1:**
**R² formula:**
R² = 1 - RSS/TSS = 1 - Σ(yₙ-ŷₙ)² / Σ(yₙ-ȳ)²

Components:
- **RSS (Residual Sum of Squares):** Σ(yₙ-ŷₙ)² — unexplained variance remaining after the model
- **TSS (Total Sum of Squares):** Σ(yₙ-ȳ)² — total variance in y (how much y varies around its mean)
- **RSS/TSS:** Fraction of variance the model FAILED to explain
- **1 - RSS/TSS:** Fraction of variance the model DID explain

**Adjusted R² formula:**
R²_adj = 1 - (RSS/(N-p-1)) / (TSS/(N-1))
= 1 - [(1-R²)(N-1)] / (N-p-1)

Additional components:
- **N:** Number of observations
- **p:** Number of predictor variables (features)
- **N-p-1:** Degrees of freedom of the residuals — accounts for the number of parameters estimated
- **N-1:** Total degrees of freedom

**Conceptual difference:**
R² measures proportion of variance explained — but uses raw RSS/TSS, which can only improve with more features. Adjusted R² uses **variance** (divides by degrees of freedom) rather than raw sums of squares. Each additional feature "costs" one degree of freedom. If the feature doesn't improve RSS by enough to offset this cost, Adjusted R² decreases.

**Part 2:**
**Fundamental flaw of R² and algebraic proof:**

R² = 1 - RSS/TSS

TSS is a fixed constant (depends only on y, not the model). Therefore R² changes only through RSS.

When you add a new feature to a linear regression model:
- The new model has one additional parameter to optimize
- In the worst case, the optimizer sets this parameter to zero → RSS stays the same → R² stays the same
- In any other case, the optimizer finds a non-zero value that reduces RSS → R² increases

**Therefore: R² can never decrease when adding features.**

Even random noise features improve R² slightly — the optimizer always finds the weight that minimizes training RSS, which will be non-zero for any feature with any correlation to y, however small.

**This makes R² useless for model comparison with different feature counts:** A model with 50 random noise features has higher R² than a model with 5 genuinely informative features — not because it's better, but because it has more parameters to exploit.

**Part 3:**
**When R² decreases:**
R² can only decrease if RSS increases. This can happen:
- **Never with standard OLS:** The optimizer minimizes RSS — adding a feature can only decrease or maintain RSS, never increase it. Standard OLS R² is mathematically guaranteed to not decrease with added features.
- **Possible with regularized models:** If you add a feature AND apply strong regularization, the regularization might distort all coefficients enough that the overall fit worsens. But this is not standard OLS R².
- **With constrained models or different data:** If the new feature is perfectly negatively correlated with existing features in a way that increases total prediction error (very unusual).

**When Adjusted R² decreases:**
Adjusted R² decreases when adding a feature DOES NOT improve the model fit sufficiently to justify the loss of one degree of freedom.

Specifically: Adjusted R² decreases when the new feature's F-statistic < 1, which means:
- The proportional improvement in RSS from adding the feature
- Is less than 1/(N-p-1) — the proportional improvement expected from pure chance

**Algebraically:**
R²_adj = 1 - [(1-R²)(N-1)] / (N-p-1)

When we add feature p+1:
- N stays the same
- p increases by 1 → (N-p-1) decreases by 1
- R² increases by some small amount Δ

R²_adj(new) - R²_adj(old) > 0 only if the increase in R² outweighs the decrease in degrees of freedom.

If the new feature is pure noise: R² barely increases (by a random small amount), but (N-p-1) decreases by 1, making (1-R²)(N-1)/(N-p-1) larger → R²_adj decreases.

---

## BONUS CHALLENGE QUESTIONS

---

**Question 13**
Cross-topic synthesis.

1. You are evaluating three models: (A) Linear Regression, (B) Ridge with λ=0.01, (C) Ridge with λ=100, all on the same dataset (N=50, p=20). Rank them from highest to lowest on: training RSS, test RSS, coefficient magnitudes, and Adjusted R². Justify every ranking.
2. A medical diagnostic model achieves: Precision=0.95, Recall=0.40, Accuracy=0.98 on a rare disease dataset (1% prevalence). Interpret each metric in plain English. Is this a good model? What does each number actually tell you about clinical utility?
3. Connect the entire regularization framework to the Bias-Variance decomposition. Explain how λ controls Bias² and Variance, how this manifests in training vs. test error, why we use cross-validation to find optimal λ, and what metric you would use for model selection at the end.

### ✅ Answer

**Part 1:**
**Rankings for Models A, B, C:**

**Training RSS (lowest to highest): A < B << C**

- Model A (OLS, λ=0): Minimizes training RSS with no constraints. Lowest possible training RSS.
- Model B (λ=0.01): Very slight regularization, barely deviates from OLS. Training RSS slightly higher than A.
- Model C (λ=100): Strong regularization forces coefficients toward zero. Predictions move toward the mean, fitting individual training points poorly. Highest training RSS.

**Test RSS (likely ordering): B < A, B < C** (B is likely best)

- Model A: With N=50 and p=20, OLS has 20/50 = 40% as many parameters as observations — moderate overfitting risk. Test RSS > training RSS with a significant gap.
- Model B: Small regularization slightly reduces variance without much bias — likely best test RSS.
- Model C: Strong regularization produces high bias — predictions near mean regardless of input. Test RSS is high due to systematic underfitting.

The exact ordering of A vs. C on test RSS depends on data characteristics, but B is likely best.

**Coefficient magnitudes (largest to smallest): A > B >> C**

- A: Unconstrained OLS. With 20 correlated features (likely in a p=20 setting), coefficients may be large and canceling.
- B: Slightly shrunk toward zero by λ=0.01 penalty. Marginally smaller than A.
- C: Strongly shrunk toward zero by λ=100. Coefficients close to (but not exactly) zero for all features.

**Adjusted R² (highest to lowest): B > A > C (likely)**

- Model B: Best generalization (test RSS) translates to best explained variance on held-out variance. Highest Adjusted R² if we use a proper estimate.
- On TRAINING data: A > B > C (same order as training RSS)
- For model selection (what Adjusted R² is designed for): B > A because Adjusted R² penalizes complexity but rewards genuine fit. The small regularization in B achieves better generalization.
- Model C: High training RSS → low training R² → low Adjusted R²

**Part 2:**
**Interpretation of medical model metrics:**

**Accuracy = 0.98:**
"98% of patients were correctly classified (as sick OR healthy)." 
This sounds excellent. But with 1% disease prevalence: predicting "healthy" for everyone gives 99% accuracy. Our 98% accuracy is actually WORSE than always saying "healthy" — a meaningless or possibly harmful model can outscore this model by accuracy alone. **Accuracy is completely misleading here.**

**Precision = 0.95:**
"Of the patients my model flagged as having the disease, 95% actually had it." This is excellent — the model's positive predictions are highly trustworthy. When the model raises an alarm, it is right 95% of the time. Low false alarm rate — patients identified as sick are almost certainly sick.

**Recall = 0.40:**
"Of all patients who actually had the disease, my model only identified 40%." This is catastrophic for a contagious dangerous disease. 60% of sick patients are told they're healthy and sent home untreated. They continue spreading disease, deteriorating, potentially dying.

**Clinical utility assessment:**
This model is NOT clinically useful for screening. Despite high precision and accuracy:
- It **misses 60% of actual cases** — fundamentally failing at the primary clinical goal
- The high accuracy is driven by correctly identifying healthy patients (trivial with 99% healthy population)
- The high precision just means "when it bothers to say 'sick,' it's usually right" — but it rarely says sick

**What would be clinically useful:**
A screening test should prioritize Recall ≥ 0.95 (catch at least 95% of cases), accepting lower Precision (more false alarms that follow-up tests resolve). This model is calibrated for the wrong objective — it's too conservative, missing the majority of cases to avoid false alarms.

**Part 3:**
**Complete connection: Regularization → Bias-Variance → Training/Test Error → Cross-Validation → Model Selection:**

**How λ controls Bias² and Variance:**

The Ridge solution: w(λ) = (XᵀX + λI)⁻¹Xᵀy

**λ controls Variance:**
Var(ŷ(x)) = σ²xᵀ(XᵀX + λI)⁻²XᵀXx

As λ increases: (XᵀX + λI)⁻¹ becomes dominated by λ⁻¹I → predictions become more similar across different training datasets → Variance decreases.

**λ controls Bias²:**
Bias(ŷ(x)) = -λxᵀ(XᵀX + λI)⁻¹w*

As λ increases: The bias term grows — the model is systematically pushed away from the true optimal weights w* toward zero.

**How this manifests in training vs. test error:**

Training error only reflects ED(w) on training data:
- λ ↑ → training RSS increases (model is constrained from fitting perfectly)
- Training error monotonically increases with λ

Test error reflects ED(w) + Bias²(w) + Var effects:
- Small λ: Low Bias, High Variance → test error dominated by variance (overfitting)
- Optimal λ*: Minimum of Bias² + Variance → minimum test error
- Large λ: High Bias, Low Variance → test error dominated by bias (underfitting)

Training error always decreases as λ → 0 (more flexibility = better training fit). Test error forms a U-curve with minimum at optimal λ*.

**Why cross-validation to find optimal λ:**
The optimal λ* minimizes test error (Bias² + Variance), NOT training error. Training error always improves with smaller λ — using training error to select λ would always choose λ = 0 (OLS). We need an unbiased estimate of test/generalization error.

K-fold cross-validation:
1. For each candidate λᵢ in {0.001, 0.01, 0.1, 1, 10, 100, ...}
2. Train on K-1 folds, validate on remaining fold
3. Repeat K times, average validation errors
4. This gives an estimate of test error for each λᵢ
5. Select λ* = argmin CV_error(λ)

CV error estimates the true test error because each validation fold was held out during training — exactly mimicking the train/test generalization scenario.

**Final model selection metric:**
After selecting optimal λ* by cross-validation, for final model reporting:
- **Test MSE or RMSE:** For interpretable absolute error in original units
- **Adjusted R²:** For proportion of variance explained, accounting for model complexity
- **Cross-validated MSE at λ*:** The primary selection criterion itself

Do NOT use training R² or training RSS for final selection — these are biased toward complexity and will always favor λ → 0.

---

## Complete Answer Reference Table

| Question | Core Concepts | Difficulty | Exam Priority |
|----------|--------------|------------|---------------|
| Q1 | When OLS fails, regularization framework | Medium | Very High |
| Q2 | Ridge regression, eigenvalue argument, intercept exclusion | Hard | Very High |
| Q3 | Lasso, soft thresholding, continuous subset selection | Hard | Very High |
| Q4 | ElasticNet, correlated features, full comparison | Hard | High |
| Q5 | Bias-Variance derivation, term interpretation | Very Hard | Very High |
| Q6 | λ controls Bias-Variance, practical manifestation | Hard | Very High |
| Q7 | MAE/MSE/RMSE formulas, outlier robustness, differentiability | Medium | Very High |
| Q8 | RMSLE, MAPE, metric selection for scale invariance | Medium-Hard | High |
| Q9 | Confusion matrix, all classification metrics derived | Medium | Very High |
| Q10 | Precision-Recall tradeoff, F1 harmonic mean | Hard | Very High |
| Q11 | ROC/AUC, threshold independence, imbalanced data | Hard | Very High |
| Q12 | R² vs Adjusted R², when each changes | Medium | High |
| Q13 | Cross-topic synthesis | Very Hard | Medium |

---

## Top 7 Most Likely Exam Questions From This Topic

1. **Precision vs. Recall** — tradeoff, when to prioritize each, domain examples (mirrors Q3 from original exam exactly)
2. **Accuracy on imbalanced data** — why it fails, concrete example (mirrors Q3 part 1 from original exam exactly)
3. **L1 vs L2 regularization** — geometric argument, sparsity, when to use each (mirrors Q2 parts 2&3 from original exam)
4. **Bias-Variance decomposition** — expected loss = bias² + variance + noise, what each means
5. **F1 Score** — why harmonic mean, when to use vs. Precision/Recall separately
6. **Ridge solution** — write (XᵀX+λI)⁻¹Xᵀy, explain eigenvalue argument, role of λ
7. **ROC/AUC** — what the curve shows, what AUC measures, why better than Accuracy for imbalanced data

**Send the next slides and I will build the complete exam for those topics too!**