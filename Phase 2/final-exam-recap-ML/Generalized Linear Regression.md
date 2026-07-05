- [[#Comprehensive Practice Exam WITH ANSWERS|Comprehensive Practice Exam WITH ANSWERS]]
	- [[#Comprehensive Practice Exam WITH ANSWERS#Generalized Linear Regression — Complete Question Bank|Generalized Linear Regression — Complete Question Bank]]
- [[#SECTION 1: Basis Functions & Extending Linear Models|SECTION 1: Basis Functions & Extending Linear Models]]
		- [[#Generalized Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 2: Maximum Likelihood & The Probabilistic View|SECTION 2: Maximum Likelihood & The Probabilistic View]]
		- [[#Generalized Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Generalized Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 3: The Design Matrix|SECTION 3: The Design Matrix]]
		- [[#Generalized Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 4: Ill-Conditioning & The Condition Number|SECTION 4: Ill-Conditioning & The Condition Number]]
		- [[#Generalized Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 5: The Runge Phenomenon|SECTION 5: The Runge Phenomenon]]
		- [[#Generalized Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 6: Polynomial Regression — Key Practical Considerations|SECTION 6: Polynomial Regression — Key Practical Considerations]]
		- [[#Generalized Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 7: Regularization|SECTION 7: Regularization]]
		- [[#Generalized Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Generalized Linear Regression — Complete Question Bank#✅ Answer|✅ Answer]]
	- [[#SECTION 7: Regularization#BONUS CHALLENGE QUESTIONS|BONUS CHALLENGE QUESTIONS]]
		- [[#BONUS CHALLENGE QUESTIONS#✅ Answer|✅ Answer]]
	- [[#SECTION 7: Regularization#Complete Answer Reference Table|Complete Answer Reference Table]]
	- [[#SECTION 7: Regularization#Top 6 Most Likely Exam Questions From This Topic|Top 6 Most Likely Exam Questions From This Topic]]

# Comprehensive Practice Exam WITH ANSWERS
## Generalized Linear Regression — Complete Question Bank

---

# SECTION 1: Basis Functions & Extending Linear Models

---

**Question 1**
Linear models have fundamental limitations that basis functions overcome.

1. Explain precisely why the standard linear model y(x,w) = w₀ + Σwⱼxⱼ is limited. What class of relationships can it NOT capture, and why is this a problem for real-world data?
2. Write the generalized linear model using basis functions φⱼ(x). Explain every component and what specifically changes compared to the standard linear model. Why is this still called "linear" despite using nonlinear functions?
3. List the three types of basis functions from the slides. For each, write its formula, describe its shape, and give a concrete real-world scenario where it would be the most appropriate choice.

### ✅ Answer

**Part 1:**
**Why the standard linear model is limited:**

The standard linear model y = w₀ + w₁x₁ + w₂x₂ + ... + wDxD is a linear function of both the parameters w AND the input variables x. This means it can only represent **flat hyperplanes** in the input space — it assumes the relationship between inputs and output is a straight line (in 1D), a flat plane (in 2D), or a hyperplane (in higher dimensions).

**What it cannot capture:**
- **Curved relationships:** If y grows as x² (e.g., kinetic energy = ½mv²), a linear model will always underfit — no matter how well you tune the weights, a straight line cannot represent a parabola
- **Oscillatory patterns:** Seasonal data (sales peaking in December, dipping in February) follows periodic patterns that straight lines cannot model
- **Threshold effects:** Many biological and economic phenomena have nonlinear responses (e.g., drug dose-response curves are sigmoidal — almost no effect at low doses, maximum effect at high doses, with a steep transition in between)
- **Interaction effects:** The combined effect of two features being simultaneously high may be much larger than the sum of their individual effects — this requires cross-product terms

**Why this is a real-world problem:**
The slides explicitly state: "In the real world it is extremely unlikely that the true function f(x) is actually linear in x." Almost every natural phenomenon has some degree of nonlinearity. Forcing a linear model on inherently nonlinear data produces high bias — systematic, irreducible errors regardless of how much data you have.

**Part 2:**
**Generalized linear model with basis functions:**

y(x,w) = Σⱼ₌₀ᴹ⁻¹ wⱼφⱼ(x) = wᵀφ(x)

Where:
- **φⱼ(x):** The j-th basis function — a fixed, nonlinear transformation of the input x. These are chosen by the designer, not learned.
- **φ(x) = (φ₀(x), φ₁(x), ..., φₘ₋₁(x))ᵀ:** The vector of all M basis function values
- **w = (w₀, w₁, ..., wₘ₋₁)ᵀ:** The weight vector — these ARE learned from data
- **φ₀(x) = 1:** A dummy basis function (always equals 1) introduced to incorporate the bias weight w₀
- **M:** The number of basis functions (controls model complexity)

**What changes vs. standard linear model:**
The inputs to the linear combination are no longer the raw features x₁, x₂, ..., xD — they are nonlinear transformations φ₁(x), φ₂(x), ..., φₘ(x) of those features. This allows the model to represent curved, nonlinear relationships in x while remaining linear in w.

**Why still called "linear":**
The model remains a linear function of the **parameters w** — if you double all weights, the output doubles; the weights appear only as first-degree terms with no w²ᵢ or wᵢwⱼ cross-products. Linearity in the parameters is what matters for the mathematical machinery (least squares, normal equations, convexity guarantees). Nonlinearity in x is introduced through the basis functions φⱼ(x), which are fixed and known — not learned.

This is the key insight: **nonlinearity in x, linearity in w** — the best of both worlds.

**Part 3:**
Three types of basis functions:

**1. Polynomial: φⱼ(x) = xʲ**
- **Formula:** φ₁(x) = x, φ₂(x) = x², φ₃(x) = x³, etc.
- **Shape:** Power functions — increasingly curved, starting flat near zero and growing faster for higher degrees
- **Real-world scenario:** Modeling the relationship between a car's speed and its braking distance. Braking distance grows approximately quadratically with speed (d ∝ v²) — a degree-2 polynomial basis captures this physical relationship naturally. Also: projectile motion, compound interest curves.

**2. Gaussian: φⱼ(x) = exp(-( x - μⱼ)²/2s²)**
- **Formula:** Bell-shaped curves centered at different locations μⱼ with width s
- **Shape:** Localized bumps — each basis function is only "active" (significantly non-zero) near its center μⱼ, decaying rapidly away from center
- **Real-world scenario:** Modeling a sensor response that varies across space — a temperature sensor reading that peaks near a heat source and decays with distance. Also: radial basis function networks, density estimation, any problem where the response is locally smooth but varies from region to region.

**3. Sigmoidal: φⱼ(x) = σ((x - μⱼ)/s) where σ(a) = 1/(1+e⁻ᵃ)**
- **Formula:** S-shaped curves transitioning from 0 to 1 at different thresholds μⱼ
- **Shape:** Smooth step functions — essentially zero below the threshold, transitions to one, essentially one above
- **Real-world scenario:** Modeling dose-response relationships in pharmacology — a drug has negligible effect at low concentrations, then rapidly becomes effective above a threshold dose, then saturates at maximum effect. Also: any categorical threshold in a continuous variable (e.g., income above $50K triggers a different credit behavior), activation functions in neural networks.

---

# SECTION 2: Maximum Likelihood & The Probabilistic View

---

**Question 2**
Maximum likelihood provides a probabilistic foundation for linear regression.

1. State the probabilistic assumption underlying maximum likelihood linear regression. Write the distribution of t given x and explain every component of the model. Why is this assumption called "Gaussian noise"?
2. Write the likelihood function for N independently drawn observations. Explain why we use the log-likelihood rather than the likelihood directly.
3. Show that maximizing the log-likelihood with respect to w is equivalent to minimizing the sum-of-squares error function. Explain why this equivalence is important.

### ✅ Answer

**Part 1:**
**Probabilistic assumption — Gaussian noise model:**

The assumption is that each observed target value tₙ is generated by the true regression function plus independent Gaussian noise:

**t = y(x,w) + ε where ε ~ N(0, β⁻¹)**

This gives the conditional distribution:
**p(t|x,w,β) = N(t | y(x,w), β⁻¹)**

Equivalently written as:
p(t|x,w,β) = (β/2π)^(1/2) exp(-β/2 × (t - y(x,w))²)

Every component:
- **t:** The observed target value (what we measure)
- **y(x,w):** The mean of the distribution — the "true" regression function evaluated at x with parameters w
- **β:** The **precision** parameter — the inverse of the noise variance (β = 1/σ²)
- **β⁻¹:** The **variance** of the noise — how much random scatter exists around the regression line
- **ε:** The noise term — random variation due to measurement error, unmodeled effects, inherent randomness
- **N(t | μ, σ²):** A Gaussian (normal) distribution with mean μ and variance σ²

**Why called "Gaussian noise":**
The error/noise ε is assumed to follow a Gaussian (normal) distribution centered at zero. This means:
- Most observations fall close to the regression line (peak of the bell curve)
- Observations further from the line are increasingly unlikely
- Deviations above and below the line are equally likely (symmetric)
- The noise has mean zero — on average, the observations equal the true regression value

**Why unimodal assumption matters:**
The slides note this assumes the conditional distribution p(t|x) is **unimodal** — there is one most likely value of t for each x (the regression function value). This fails when, for example, the data comes from a mixture of populations where the same x could yield very different t values through different mechanisms.

**Part 2:**
**Likelihood function:**

Assuming observations {(xₙ, tₙ)} are drawn **independently**, the joint probability (likelihood) is the product:

p(t|X,w,β) = Πₙ₌₁ᴺ N(tₙ | y(xₙ,w), β⁻¹)

= Πₙ₌₁ᴺ (β/2π)^(1/2) exp(-β/2 × (tₙ - y(xₙ,w))²)

**Why log-likelihood instead of likelihood:**

**Mathematical reason 1 — Product to sum:**
The likelihood is a product of N terms. Taking the log converts this to a sum:
ln p(t|X,w,β) = Σₙ ln N(tₙ|y(xₙ,w),β⁻¹)

Sums are much easier to differentiate than products. The log-likelihood gradient is a sum of terms, each depending on one data point.

**Mathematical reason 2 — Numerical stability:**
N probabilities multiplied together can be astronomically small — for N=1000 data points, each probability might be 0.1, giving a likelihood of 0.1^1000 = 10^{-1000}. This underflows to exactly zero in floating-point arithmetic, making numerical optimization impossible. Taking the log keeps values in a manageable range: log(10^{-1000}) = -1000.

**Mathematical reason 3 — Monotonic transformation:**
log is a monotonically increasing function — maximizing log(L) gives the same maximizer as maximizing L. The argmax is preserved, so we find identical parameters.

**Log-likelihood for Gaussian model:**
ln p = (N/2)ln(β/(2π)) - (β/2) Σₙ (tₙ - y(xₙ,w))²

**Part 3:**
**Equivalence of maximum likelihood and least squares:**

Maximizing the log-likelihood with respect to w:

ln p = (N/2)ln(β/(2π)) - (β/2) Σₙ (tₙ - y(xₙ,w))²

The only term involving w is -(β/2) Σₙ (tₙ - y(xₙ,w))²

Maximizing this with respect to w means:
- Maximizing -(β/2) × (something) is the same as minimizing (β/2) × (something)
- Since β > 0 is a constant, minimizing (β/2) × Σₙ(tₙ - y(xₙ,w))² is the same as minimizing Σₙ(tₙ - y(xₙ,w))²

This is exactly the **sum-of-squares error function ED(w) = (1/2)Σₙ(tₙ - y(xₙ,w))²**

Therefore: **Maximum likelihood under Gaussian noise ≡ Least squares minimization**

**Why this equivalence is important:**

1. **Theoretical foundation:** It shows that least squares is not just a convenient computational choice — it has a rigorous probabilistic justification. Least squares is optimal (maximum likelihood) when the noise is truly Gaussian.

2. **Assumptions revealed:** It makes the implicit assumption of least squares explicit — by using MSE/least squares, you are implicitly assuming that the errors follow a Gaussian distribution. If errors are not Gaussian (e.g., heavy-tailed, skewed), least squares is no longer the maximum likelihood estimator and better alternatives exist.

3. **Extension path:** Once you have the probabilistic framework, you can extend to non-Gaussian noise (robust regression for heavy-tailed noise, Poisson regression for count data, etc.) by changing the assumed distribution while keeping the maximum likelihood principle.

4. **Gradient formula:** The gradient of the log-likelihood:
∇ₓln p = β Σₙ (tₙ - y(xₙ,w)) φ(xₙ)

Setting to zero gives: Σₙ (tₙ - y(xₙ,w)) φ(xₙ) = 0 → ΦᵀΦw = Φᵀt → **Normal Equations**

---

**Question 3**
The noise precision parameter β has a specific maximum likelihood estimate.

1. Explain what the precision parameter β represents physically. How does it relate to the noise variance σ²?
2. Derive or explain the maximum likelihood estimate for β. What does the formula reveal about how to estimate noise from data?
3. Explain why knowing β (or σ²) is important beyond just fitting the regression line. What does it allow you to do that point prediction alone cannot?

### ✅ Answer

**Part 1:**
**Physical meaning of precision β:**

**β = 1/σ²** where σ² is the variance of the Gaussian noise.

**High β (high precision = low variance):**
- Observations cluster tightly around the regression line
- Small noise — measurements are very accurate
- The regression line explains almost all variation in t
- Predictions are highly reliable — narrow confidence intervals

**Low β (low precision = high variance):**
- Observations are widely scattered around the regression line
- Large noise — measurements are noisy or many unmodeled factors affect t
- The regression line explains only some of the variation
- Predictions are unreliable — wide confidence intervals

**Physical example:**
Measuring temperature with a precision thermometer (high β) vs. a cheap thermometer in a windy location (low β). The regression line (true temperature trend) is the same — but the scatter around it differs.

**Part 2:**
**Maximum likelihood estimate of β:**

After finding w_ML (by minimizing sum-of-squares), maximize the log-likelihood with respect to β:

∂(ln p)/∂β = N/(2β) - (1/2) Σₙ (tₙ - y(xₙ, w_ML))² = 0

Solving for β:
**1/β_ML = (1/N) Σₙ (tₙ - y(xₙ, w_ML))²**

Or equivalently:
**σ²_ML = (1/N) Σₙ rₙ²**

Where rₙ = tₙ - ŷₙ is the residual for observation n.

**What this reveals:**
The maximum likelihood estimate of the noise variance is simply the **average squared residual** — the mean squared error of the fitted model on training data. The noise variance equals how wrong the model is on average.

This makes intuitive sense: after fitting the best possible regression line, whatever scatter remains around that line must be due to noise. The variance of that remaining scatter is our best estimate of the noise variance.

**Important note:** σ²_ML = (1/N) Σrₙ² is a **biased** estimate of the true noise variance (divides by N instead of N-k-1, where k is the number of features). The unbiased estimate divides by N-k-1 to account for the degrees of freedom used in estimating w.

**Part 3:**
**Why knowing β matters beyond point prediction:**

Point prediction ŷ(x) = wᵀφ(x) gives only the most likely value. Knowing β additionally provides:

1. **Confidence intervals:** Since p(t|x,w,β) = N(t|ŷ(x), β⁻¹), we can compute:
   - 95% confidence interval: ŷ(x) ± 1.96/√β
   - This tells a user: "I predict 45.2, and I'm 95% confident the true value is between 41.8 and 48.6"

2. **Prediction uncertainty quantification:** Different regions of x may have different effective noise levels. Knowing β lets you communicate how much to trust any given prediction — essential in medical, financial, and engineering applications where decisions depend on prediction reliability.

3. **Model comparison:** Two models might have the same RSS but very different β_ML estimates — the model with higher β (lower residual variance) is doing a genuinely better job explaining the data.

4. **Detecting model misspecification:** If the true noise is not Gaussian (e.g., heavy-tailed), the estimated β will be systematically wrong, flagging that the Gaussian assumption may be violated.

5. **Bayesian extension:** β is a key parameter in Bayesian linear regression — it combines with a prior on w to produce a posterior distribution, enabling full probabilistic predictions that account for both noise and parameter uncertainty.

---

# SECTION 3: The Design Matrix

---

**Question 4**
The design matrix Φ is central to generalized linear regression.

1. Define the design matrix Φ for the basis function model. Explain its dimensions and what each entry Φₙⱼ represents.
2. Write out the Vandermonde matrix that results from choosing polynomial basis functions φⱼ(x) = xʲ. Explain the structure of this matrix and why each column has a specific interpretation.
3. Explain why the columns of the Vandermonde matrix become highly correlated when x values are restricted to a narrow range (e.g., x ∈ [0,1]). What mathematical consequence does this have?

### ✅ Answer

**Part 1:**
**Design matrix Φ:**

Φ is an **N × M matrix** where:
- N = number of training observations
- M = number of basis functions

**Entry definition:**
Φₙⱼ = φⱼ(xₙ)

The (n,j) entry is the value of the j-th basis function evaluated at the n-th training input xₙ.

**Full matrix structure:**

Φ = 
| φ₀(x₁)  φ₁(x₁)  ...  φₘ₋₁(x₁) |
| φ₀(x₂)  φ₁(x₂)  ...  φₘ₋₁(x₂) |
| ...                               |
| φ₀(xₙ)  φ₁(xₙ)  ...  φₘ₋₁(xₙ) |

- **Row n:** All M basis function values for observation n — the feature representation of the n-th data point in basis function space
- **Column j:** The j-th basis function evaluated at all N training inputs — how the j-th feature varies across observations
- **First column:** φ₀(xₙ) = 1 for all n → a column of all ones — corresponds to the bias weight w₀

**The model in matrix form:**
ŷ = Φw

All Normal Equations and geometry from standard linear regression apply exactly — just replace X with Φ.

**Part 2:**
**Vandermonde Matrix (polynomial basis φⱼ(x) = xʲ):**

Φ_Vandermonde =
| 1  x₁   x₁²  ...  x₁ᴹ⁻¹ |
| 1  x₂   x₂²  ...  x₂ᴹ⁻¹ |
| .                          |
| 1  xₙ   xₙ²  ...  xₙᴹ⁻¹ |

**Column interpretations:**
- **Column 0 (all 1s):** The constant term — corresponds to the intercept w₀
- **Column 1 (xₙ values):** The linear feature — corresponds to the degree-1 coefficient w₁
- **Column 2 (xₙ² values):** The quadratic feature — corresponds to the degree-2 coefficient w₂
- **Column j (xₙʲ values):** The degree-j feature — corresponds to the degree-j polynomial coefficient wⱼ

**Interpolation Theorem connection:**
If we have N distinct points {xᵢ}, there exists a polynomial of degree at most N-1 that passes exactly through all N points. This means the N×N Vandermonde matrix (with M=N) is invertible when all xᵢ are distinct — a unique degree-(N-1) polynomial interpolates any set of N distinct points. This is simultaneously a mathematical fact and a warning: such a polynomial always exists but is almost always severely overfit.

**Part 3:**
**Why columns become correlated for x ∈ [0,1]:**

For x ∈ [0,1], all values are between 0 and 1. Consider higher powers:
- x = 0.8: x² = 0.64, x³ = 0.51, x⁴ = 0.41, x⁵ = 0.33

For x close to 1: x ≈ x² ≈ x³ ≈ x⁴ ≈ x⁵
For x close to 0: all powers are close to 0

The **columns of the Vandermonde matrix are nearly identical** — x⁴ ≈ x⁵ for all values in [0,1]. This means column 4 ≈ column 5, column 3 ≈ column 4, etc.

**Mathematical consequence:**
The columns are nearly linearly dependent → the rank of Φ is effectively less than M despite having M columns → **ΦᵀΦ is nearly singular** (its smallest eigenvalues are very close to zero).

The design matrix Φ has **high multicollinearity** — the polynomial features convey nearly the same information. The optimizer cannot distinguish between "how much of this effect is due to x⁴ vs. x⁵" because they look almost identical in the data.

**Result:** ΦᵀΦ approaches singularity → its inverse (ΦᵀΦ)⁻¹ is **numerically unstable** — tiny errors in the data cause enormous changes in the estimated weights. This is ill-conditioning.

---

# SECTION 4: Ill-Conditioning & The Condition Number

---

**Question 5**
Ill-conditioning is a major numerical problem in polynomial regression.

1. Define the condition number κ(A) of a matrix. Write the formula and explain what σ_max and σ_min are. What does it mean for a matrix to be "well-conditioned" vs. "ill-conditioned"?
2. Explain precisely why a large condition number κ(ΦᵀΦ) causes catastrophic problems when solving for the regression weights. Show the chain of causation from large κ to unstable weights.
3. The slides state: "Ill-conditioning ≠ overfitting. It is a numerical stability issue, but high-degree polynomials often cause both." Explain the distinction between these two problems clearly. Can a model be ill-conditioned but not overfit? Can it overfit without ill-conditioning?

### ✅ Answer

**Part 1:**
**Condition Number κ(A):**

κ(A) = σ_max / σ_min

Where:
- **σ_max:** The largest singular value of matrix A — the maximum "stretch" the matrix applies to any unit vector
- **σ_min:** The smallest singular value — the minimum "stretch" in any direction
- **Singular values:** Non-negative square roots of eigenvalues of AᵀA (or AAᵀ) — they measure how much the matrix stretches/compresses vectors in different directions

**Interpretation:**
- **κ(A) ≈ 1:** Well-conditioned — the matrix stretches all directions approximately equally. A small change in input produces a proportionally small change in output. Numerically stable.
- **Large κ(A):** Ill-conditioned — the matrix stretches some directions enormously while nearly collapsing others. A small change in input can produce a disproportionately large change in output.
- **κ(A) → ∞:** Occurs when σ_min → 0 — the matrix is approaching singularity. At exact singularity (σ_min = 0), the matrix has no inverse.

**Practical example:**
κ(A) = 1000 means: a relative error of 10⁻³ in the data can cause a relative error of up to 10⁻³ × 1000 = 1 (100%) in the solution. The error is amplified by the condition number.

**Part 2:**
**Chain of causation from large κ to unstable weights:**

**Step 1:** We need to solve: ΦᵀΦw = Φᵀt → w = (ΦᵀΦ)⁻¹Φᵀt

**Step 2:** If κ(ΦᵀΦ) is large, then ΦᵀΦ has at least one very small singular value σ_min ≈ 0. Inverting ΦᵀΦ requires dividing by its singular values. When σ_min ≈ 0, we are dividing by a tiny number → 1/σ_min → ∞.

**Step 3:** Small perturbations enter the data through:
- Measurement noise in tₙ (small but always present)
- Floating-point rounding errors in computation
- Tiny perturbations in training data

**Step 4:** These small errors, when multiplied by (ΦᵀΦ)⁻¹ with its enormous entries (from dividing by σ_min ≈ 0), get **amplified by κ(ΦᵀΦ)** in the solution w.

**Result:** The estimated weights w become **enormous in magnitude, oscillating wildly in sign** — alternating large positive and negative values that nearly cancel each other. A tiny change in one data point completely changes all the polynomial coefficients. The model is numerically meaningless.

**Concrete illustration:**
κ = 10⁸ means: a rounding error of 10⁻¹⁶ (machine precision for 64-bit floating point) gets amplified to 10⁻¹⁶ × 10⁸ = 10⁻⁸ relative error in w — which might still be acceptable. But κ = 10¹⁶ would give errors of order 1 — completely wrong weights from floating-point rounding alone.

**Part 3:**
**Ill-conditioning vs. Overfitting — the distinction:**

**Overfitting** is a **statistical/generalization** problem:
- The model fits training data too well by capturing noise and specific training set idiosyncrasies
- Measured by the gap between training error and test error
- Exists even with perfect numerical computation
- Caused by model complexity exceeding what the data can support

**Ill-conditioning** is a **numerical/computational** problem:
- The algorithm cannot stably compute the model parameters due to near-singularity of ΦᵀΦ
- Exists even if the model class would generalize perfectly (if you could compute the correct weights)
- Measured by the condition number κ
- Caused by correlations in the feature matrix, not by model complexity relative to data

**Can a model be ill-conditioned but not overfit?**
Yes — consider a degree-3 polynomial (not overfit for many datasets) with x values in [0.99, 1.01] (extremely narrow range). The polynomial columns x, x², x³ are nearly identical in this tiny range → high condition number → ill-conditioned. But the polynomial might generalize well if x test values are also in this range and the true function is genuinely cubic. The numerical computation is unstable, but the statistical goal (fitting a cubic) is appropriate.

**Can a model overfit without ill-conditioning?**
Yes — consider a Decision Tree with depth 20 (severely overfit). Decision Trees don't involve matrix inversions — there is no condition number concept. The tree perfectly memorizes training data (overfitting) but the algorithm's numerical computations are perfectly stable.

**Why high-degree polynomials cause BOTH:**
High-degree polynomial → many correlated columns in Vandermonde matrix → ill-conditioned ΦᵀΦ (numerical problem) AND → many parameters relative to data → overfitting (statistical problem). The two problems co-occur for high-degree polynomials but are conceptually distinct.

---

# SECTION 5: The Runge Phenomenon

---

**Question 6**
The Runge Phenomenon reveals a fundamental limitation of polynomial interpolation.

1. Explain what the Runge Phenomenon is. What specific behavior does it exhibit, and where in the domain does it occur most severely?
2. Explain the geometric view: "As degree increases, Col(Φ) enlarges and projection fits data exactly. But between data points, function may fluctuate dramatically." What does this mean in terms of the column space interpretation?
3. Explain why polynomial basis functions are described as "global." How does this global property cause the oscillations seen in the Runge Phenomenon, and how does this contrast with Gaussian basis functions?

### ✅ Answer

**Part 1:**
**The Runge Phenomenon:**

When fitting a high-degree polynomial to data points (especially equally-spaced points), the polynomial passes exactly through all points — but between the points (particularly near the **boundaries of the interval**), the polynomial oscillates wildly with enormous amplitude.

**Specific behavior:**
- The polynomial fits all N training points perfectly (zero training error)
- Between training points, especially near x = ±1 for equally-spaced points in [-1,1], the polynomial makes violent up-and-down swings
- These oscillations grow more extreme as the degree increases
- Points near the interior of the interval are approximated well; boundary regions show the worst behavior

**Mechanical cause:**
When the polynomial is forced to pass through all N constraints (data points), the mathematical system resolves the tension between constraints by creating large alternating coefficients. The polynomial "whips around" violently between constrained points to satisfy all constraints simultaneously — this appears as oscillations.

**Classic example:**
Runge (1901) showed that fitting a polynomial to the function f(x) = 1/(1+25x²) on [-1,1] with equally-spaced points diverges as degree increases — the polynomial fits the function poorly near ±1 even as the degree (and number of fitting points) increases.

**Part 2:**
**Geometric view using column space:**

Recall: linear regression = orthogonal projection of the target vector t onto Col(Φ).

**As degree increases:**
- Each new degree adds a new column to Φ (xᴺ is added)
- The column space Col(Φ) gains one more dimension — it spans more of the N-dimensional observation space
- At degree N-1 (N columns for N observations), Col(Φ) spans all of ℝᴺ — the target vector t lies exactly in Col(Φ)
- The projection ŷ = t exactly — zero training error, perfect interpolation

**Why between-point oscillations occur:**
Col(Φ) is a mathematical subspace — it contains the projection ŷ that exactly matches training points. But the polynomial is a **continuous function** that must connect these fitted values smoothly. The polynomial basis functions (1, x, x², ...) are global — they interact across the entire domain. The only continuous polynomial of degree N-1 that passes through all N points may require wild oscillations between those points to satisfy all N constraints.

The column space view shows the issue: we can project onto Col(Φ) in observation space (N discrete points) perfectly, but the continuous function that implements this projection in input space x may oscillate arbitrarily between the observation points. The column space tells us nothing about what happens at unobserved x values.

**Part 3:**
**Global vs. local basis functions:**

**Polynomial basis functions are global:**
Each polynomial term xᵖ is non-zero everywhere across the entire domain. Changing one coefficient wₚ changes the model's output at EVERY point in the domain simultaneously. There is no locality — a coefficient meant to fit the data near x=0 also affects predictions at x=1, x=10, x=-5.

**Why this causes Runge oscillations:**
When the high-degree polynomial adjusts its coefficients to pass through a data point at x = 0.5, those same coefficient changes create ripple effects everywhere else — including the boundaries. The polynomial cannot make a local adjustment; every adjustment propagates globally. To satisfy all N constraints simultaneously with global basis functions requires the coefficients to engage in complex cancellation — alternating in sign and large in magnitude — producing the oscillatory behavior.

**Contrast with Gaussian basis functions:**
φⱼ(x) = exp(-(x-μⱼ)²/2s²) is a **localized** basis function. It is essentially zero everywhere except near its center μⱼ. Changing coefficient wⱼ primarily affects predictions near μⱼ — it has minimal effect far from μⱼ.

With Gaussian basis functions:
- Fitting a point at x = 0.5 (near μⱼ = 0.5) mainly adjusts wⱼ
- Points far from 0.5 are barely affected
- No global ripple effects → no Runge-like oscillations
- The model can make **local adjustments** without globally disrupting the fit

This is why radial basis functions (Gaussian) and splines (piecewise polynomials with locality) are often preferred over global high-degree polynomials for interpolation and regression.

---

# SECTION 6: Polynomial Regression — Key Practical Considerations

---

**Question 7**
Building polynomial regression models requires careful consideration of order, hierarchy, and extrapolation.

1. Explain the two strategies for selecting polynomial order: Forward Selection and Backward Elimination. For each, describe the starting point, the stopping criterion, and when each approach is preferable.
2. Explain the Hierarchy principle for polynomial models. What makes a model "hierarchical" and why must all polynomial models have this property? Use the transformation z = x + c to prove why non-hierarchical models fail.
3. Explain why extrapolation with polynomial models is "extremely hazardous." What specifically can go wrong that does not happen with simpler models?

### ✅ Answer

**Part 1:**
**Two strategies for polynomial order selection:**

**Forward Selection:**
- **Starting point:** Begin with the simplest model — degree 1 (linear regression)
- **Process:** Add polynomial terms one degree at a time: try degree 2, then degree 3, etc.
- **At each step:** Fit the new higher-degree model and evaluate whether the loss (RSS, cross-validated error) improves significantly
- **Stopping criterion:** Stop when adding the next degree does not produce a significant improvement — the additional complexity is not justified by the marginal gain in fit
- **When preferable:** When you expect a relatively low-degree relationship and want to start simple. Computationally efficient — stops early if a simple model suffices. Less risk of accidentally fitting a very high-degree model.

**Backward Elimination:**
- **Starting point:** Begin with the most complex model — degree N-1 (perfect interpolation of N points)
- **Process:** Remove polynomial terms one at a time, from highest degree downward
- **At each step:** Test whether the highest-remaining-degree term's coefficient has a significant t-statistic (is the coefficient significantly different from zero?)
- **Stopping criterion:** Stop removing terms when all remaining terms have t-statistics indicating their coefficients are significantly non-zero (they contribute meaningfully)
- **When preferable:** When you have theoretical reasons to expect a high-degree relationship and want to identify which terms are actually important. Can discover unexpected important higher-order terms that forward selection might miss.

**Key difference in practice:**
Forward selection is greedy in the "building up" direction — it might stop too early if a higher-degree term only becomes significant after controlling for an intermediate term. Backward elimination is greedy in the "tearing down" direction — it might retain too many terms if early terms create statistical interference. Cross-validation is more reliable than either for final model selection.

**Part 2:**
**Hierarchy Principle:**

A polynomial model is **hierarchical** if, whenever a degree-p term is included, ALL lower-degree terms (1, x, x², ..., x^(p-1)) are also included.

**Valid hierarchical model:** y = w₀ + w₁x + w₂x² + w₃x³
(includes all degrees 0 through 3)

**Non-hierarchical model:** y = w₀ + w₃x³
(missing degrees 1 and 2)

**Why non-hierarchical models fail — the z = x + c proof:**

Suppose we fit: y = w₀ + w₃x³ (non-hierarchical, missing x and x² terms)

Now suppose we shift the input by a constant: z = x + c (centering or unit change)

Then x = z - c, and:
x³ = (z - c)³ = z³ - 3cz² + 3c²z - c³

So: y = w₀ + w₃x³ = w₀ + w₃(z³ - 3cz² + 3c²z - c³)
= (w₀ - w₃c³) + (3w₃c²)z + (-3w₃c)z² + w₃z³

**After shifting by c, the model in terms of z requires ALL three terms: z, z², z³**

But the original model in x only had x³ — missing x and x² terms. After shifting, the model CANNOT represent the same function because it has no z and z² coefficients to accommodate the expansion. The non-hierarchical model is **not invariant under linear transformations of the input** — changing units or centering the data produces a fundamentally different model.

Hierarchical models are invariant: y = w₀ + w₁x + w₂x² + w₃x³ transforms cleanly under z = x+c, with the new lower-degree coefficients absorbing the expansion terms.

**Part 3:**
**Why extrapolation with polynomials is extremely hazardous:**

**The fundamental problem:**
Polynomial models are fit to data in a specific range [x_min, x_max]. Within this range, the polynomial has been "trained" by the data. Outside this range, the polynomial's behavior is governed entirely by its highest-degree terms — it follows the mathematical function, not the data.

**What can go wrong:**

1. **Polynomial growth behavior:** A degree-k polynomial grows as xᵏ for large |x|. For k=5, the polynomial might grow as x⁵ — extremely rapidly. If the true function levels off (saturates) outside the training range, the polynomial will dramatically overshoot.

2. **Unexpected turns:** A polynomial can curve in completely unanticipated directions near and outside the boundary of the training data. The Runge Phenomenon already shows wild oscillations near boundaries for training data — this only gets worse beyond the training range.

3. **No physical constraint:** A polynomial has no knowledge of physical or domain constraints. A model predicting population growth might give negative populations for future dates, or a model of drug concentration might predict arbitrarily large concentrations.

**Contrast with simpler models:**
- A **linear model** extrapolates by continuing its constant slope — wrong, but at least predictably wrong in the same direction
- A **constant model** (just the mean) extrapolates by predicting the mean — boring but safe
- A **polynomial** can extrapolate in ANY direction — wildly upward, wildly downward, or oscillating — with no way to predict which without examining the function analytically

**Practical advice:**
Never trust polynomial regression predictions for inputs more than a small fraction of the training range beyond the training data. If extrapolation is necessary, use models with theoretically motivated functional forms (e.g., exponential decay for physical processes, logistic growth for populations).

---

# SECTION 7: Regularization

---

**Question 8**
Regularization addresses overfitting in generalized linear regression.

1. Write the complete regularized error function combining the data-dependent error ED(w) and the regularization term EW(w). Explain every component and the role of the regularization coefficient λ.
2. Derive the closed-form solution for the regularized (Ridge/L2) regression. Show how λ modifies the Normal Equations and explain what the added λI term does geometrically.
3. Explain why L2 regularization is called "weight decay." In what sense do the weights "decay toward zero"?

### ✅ Answer

**Part 1:**
**Regularized error function:**

E(w) = ED(w) + λEW(w)

Where:
- **ED(w) = (1/2)Σₙ(tₙ - y(xₙ,w))²:** The **data-dependent error** — measures how well the model fits the training data (sum of squared residuals). We want this small.
- **EW(w) = (1/2)wᵀw = (1/2)||w||²:** The **regularization term** — measures the magnitude (complexity) of the weight vector. We want this small.
- **λ:** The **regularization coefficient** — controls the relative importance of fitting the data vs. keeping weights small. λ ≥ 0.

**Detailed role of λ:**
- **λ = 0:** No regularization → minimize only ED(w) → standard least squares (can overfit)
- **Small λ:** Data fit dominates → model can have large weights → flexible, may overfit
- **Large λ:** Regularization dominates → forces weights small → simple model, may underfit
- **λ → ∞:** Forces all weights to zero → model predicts only the bias w₀ → extreme underfitting

λ is the **Bias-Variance tradeoff knob**:
- Increasing λ → increases bias (simpler model), decreases variance (more stable)
- Decreasing λ → decreases bias (more flexible), increases variance (less stable)

Optimal λ found by cross-validation.

**Part 2:**
**Closed-form solution for Ridge Regression:**

The total error function:
E(w) = (1/2)Σₙ(tₙ - wᵀφ(xₙ))² + (λ/2)||w||²

Taking gradient with respect to w:
∇E(w) = -Σₙ(tₙ - wᵀφ(xₙ))φ(xₙ) + λw

In matrix form:
∇E(w) = -(Φᵀt - ΦᵀΦw) + λw = ΦᵀΦw - Φᵀt + λw

Setting to zero:
(ΦᵀΦ + λI)w = Φᵀt

**Solution:**
**w_ridge = (ΦᵀΦ + λI)⁻¹Φᵀt**

**What the added λI term does:**

1. **Fixes singularity:** ΦᵀΦ might be singular (non-invertible). Adding λI > 0 to the diagonal makes all eigenvalues at least λ > 0, guaranteeing the matrix is **positive definite and always invertible** for any λ > 0.

2. **Reduces condition number:** If ΦᵀΦ has eigenvalues [ε, 1, 2, ..., 100] (where ε ≈ 0 causes ill-conditioning), then ΦᵀΦ + λI has eigenvalues [ε+λ, 1+λ, 2+λ, ..., 100+λ]. For λ=1: [1+ε, 2, 3, ..., 101]. Condition number falls from 100/ε → ~101/(1+ε) ≈ 100. Much better conditioned.

3. **Geometric interpretation:** λI "inflates" the eigenvalues of ΦᵀΦ uniformly in all directions. It shrinks the effective influence of individual data points by making the matrix "stiffer" — less responsive to any particular direction in weight space. This is equivalent to adding N virtual extra "data points" that all pull the weights toward zero.

**Part 3:**
**Why "weight decay":**

The term "weight decay" comes from the sequential/online learning perspective.

In stochastic gradient descent, the weight update rule with L2 regularization is:

w_new = w_old - η∇ED(w) - ηλw_old
= w_old(1 - ηλ) - η∇ED(w)

The factor **(1 - ηλ)** multiplies the current weights before adding the gradient step. For ηλ > 0 (which is always true for positive learning rate and regularization):
- (1 - ηλ) < 1
- This **multiplies all weights by a number less than 1** at every update step
- Weights are **"decayed" toward zero** at each iteration unless the data gradient pulls them away

**Intuition:**
At every step, all weights are slightly reduced (decay toward zero), and only the component that the data "supports" (gradient term) prevents them from reaching zero. Weights that are not supported by the training data signal decay away to zero over iterations. Weights that genuinely help reduce training error receive a strong gradient signal that counteracts the decay.

**In batch gradient descent:**
The same intuition applies — the regularization penalty in the loss function creates a constant downward pull on all weights toward zero. Only weights that reduce the data fit loss (ED) more than the regularization cost are maintained at non-zero values in the optimal solution.

**Connection to statistics:** In Bayesian terms, L2 regularization is equivalent to placing a **zero-mean Gaussian prior** on the weights: p(w) = N(0, α⁻¹I). The regularization coefficient λ = α/β corresponds to the ratio of noise precision to prior precision. MAP (Maximum A Posteriori) estimation under this prior gives exactly the Ridge solution.

---

**Question 9**
L1 vs L2 Regularization produce fundamentally different solutions.

1. Write the general regularized error with the q-norm penalty. Explain what happens for q=2 (Ridge) and q=1 (Lasso). What is the key qualitative difference in the solutions they produce?
2. Explain WHY L1 (Lasso) produces sparse solutions (some weights exactly zero) while L2 (Ridge) produces small but non-zero weights. Use the geometric argument involving the constraint regions.
3. Given a polynomial regression model with degree 15 (16 coefficients) fit to a dataset where you suspect only 3 polynomial terms are actually relevant, which regularization would you choose and why?

### ✅ Answer

**Part 1:**
**General regularized error with q-norm:**

E(w) = ED(w) + (λ/q)Σⱼ|wⱼ|^q

**q = 2 (Ridge/L2 regularization):**
E(w) = (1/2)Σₙ(tₙ-ŷₙ)² + (λ/2)Σⱼwⱼ²

Penalty: sum of SQUARED weights
Effect on solution: All weights are **shrunk toward zero proportionally** — each weight is multiplied by a factor less than 1, but no weight is exactly eliminated. All 16 polynomial coefficients remain in the model, just smaller.

**q = 1 (Lasso/L1 regularization):**
E(w) = (1/2)Σₙ(tₙ-ŷₙ)² + λΣⱼ|wⱼ|

Penalty: sum of ABSOLUTE weights
Effect on solution: Some weights are driven to **exactly zero** — the model becomes sparse, effectively performing **feature/term selection**. For sufficiently large λ, only the most important polynomial terms survive; the rest are completely eliminated.

**Key qualitative difference:**
- **Ridge:** All features "survive" but shrunk — good when all features are somewhat relevant
- **Lasso:** Sparse solution — some features completely eliminated — good when many features are irrelevant and you want automatic feature selection

**Part 2:**
**Geometric argument for sparsity:**

Both regularization methods can be formulated as **constrained optimization**:

- **Ridge:** Minimize ED(w) subject to Σwⱼ² ≤ t (w lies inside a sphere in weight space)
- **Lasso:** Minimize ED(w) subject to Σ|wⱼ| ≤ t (w lies inside a diamond/hypercube in weight space)

The unregularized solution (minimum of ED alone) lies at some point w* in weight space. Regularization constrains w to lie within the allowed region.

The regularized solution is where the **elliptical contours of ED(w)** first "touch" the constraint region as they expand outward from w*.

**Why L2 sphere doesn't produce sparsity:**
The sphere (L2 constraint region) has a smooth, curved surface with no corners or edges. When the expanding ellipse first touches the sphere, it almost always contacts the smooth surface — at a point where no coordinate is exactly zero. The optimal solution is at a generic point on the sphere where all wⱼ ≠ 0 (just smaller).

**Why L1 diamond produces sparsity:**
The diamond (L1 constraint region) has **corners and edges** aligned with the coordinate axes. In 2D, the diamond has 4 corners at (±t, 0) and (0, ±t) — points where one coordinate is exactly zero. In k dimensions, the diamond has 2k sharp corners aligned with the axes.

When the expanding ellipse first touches the diamond, it is most likely to contact one of these corners — which lie on the coordinate axes where some wⱼ = 0. The sharp corners "attract" the solution. As long as the gradient of ED at the corner does not point too far away from the corner, the corner remains the optimal solution.

**Probability of hitting a corner:**
In high dimensions, the fraction of the diamond's "surface" that lies at corners (where one or more coordinates = 0) is very high compared to smooth faces. The optimal solution is much more likely to land on a corner (sparse) than on a generic face (non-sparse).

**Mathematical formulation:**
Lasso's subdifferential at wⱼ = 0 includes a range [-λ, +λ], meaning wⱼ = 0 is optimal as long as the gradient of ED at wⱼ = 0 has magnitude less than λ. This allows exact zero solutions. Ridge's gradient at wⱼ = 0 is exactly zero (from the λwⱼ term) — there is no mechanism to maintain exact zeros.

**Part 3:**
**Choice for degree-15 polynomial with 3 relevant terms:**

**Choose L1 (Lasso) regularization.**

**Reasoning:**

1. **Automatic feature selection:** With 16 coefficients and only 3 genuinely relevant polynomial terms, you need a method that identifies and retains the 3 important terms while eliminating the 13 irrelevant ones. Lasso's sparsity property does exactly this — with appropriate λ, it drives the 13 irrelevant coefficients to exactly zero.

2. **Interpretability:** After Lasso, you have a sparse model with 3 non-zero coefficients. You can identify exactly which polynomial terms matter (degree 2? degree 5? degree 7?). Ridge would give 16 non-zero (but small) coefficients — you cannot determine which ones are actually important.

3. **Generalization:** A sparse model with only 3 relevant terms captures the true signal without overfitting to noise. Ridge with all 16 small coefficients might still overfit because 13 irrelevant terms contribute small amounts of noise to predictions.

4. **Ridge would not be ideal here:** Ridge shrinks all 16 coefficients but keeps all 16 non-zero. It will still use (slightly) all 15 polynomial terms including the 13 irrelevant ones. The model complexity is not reduced in terms of number of parameters — just in parameter magnitudes.

**Practical approach:**
1. Apply Lasso with several λ values
2. Use cross-validation to select the optimal λ
3. Examine the resulting sparse model — which 3 (or however many) terms survived?
4. This gives both the best predictive model AND the most interpretable model

**Caveat:** If you suspect multicollinearity among the relevant polynomial terms (which is very likely given the Vandermonde structure), consider **Elastic Net** regularization, which combines L1 and L2: it provides sparsity (L1) while handling correlated features more gracefully than pure Lasso (L2). Pure Lasso tends to arbitrarily pick one of a set of correlated features, while Elastic Net can include groups of correlated features together.

---

## BONUS CHALLENGE QUESTIONS

---

**Question 10**
Cross-topic synthesis.

1. You fit degree-1, degree-5, degree-10, and degree-15 polynomial models to a dataset of N=20 points with x ∈ [0,1]. Predict what happens to: (a) training RSS, (b) test RSS, (c) condition number κ(ΦᵀΦ), (d) weight magnitudes ||w||, as degree increases from 1 to 15. Explain the mechanism behind each trend.
2. Connect the Runge Phenomenon, ill-conditioning, and overfitting as three aspects of the same underlying problem with high-degree polynomials. Explain how they are related and how regularization addresses all three simultaneously.
3. A colleague says: "I applied L2 regularization to my degree-10 polynomial and my weights got smaller, so my model is definitely better now." Write a complete response explaining what regularization actually does and doesn't guarantee, connecting MSE, Bias-Variance, cross-validation, and the condition number.

### ✅ Answer

**Part 1:**
Predictions as degree increases from 1 to 15 (N=20 points, x ∈ [0,1]):

**(a) Training RSS:**
**Monotonically decreases (or stays equal) as degree increases.**

Mechanism: Adding a higher-degree term gives the model one more parameter to minimize training error. At worst (if the new term has zero weight), RSS stays identical. In practice, any correlation between the new feature and the target reduces RSS. At degree N-1 = 19, RSS = 0 exactly (Vandermonde interpolation theorem: a degree-19 polynomial passes through all 20 points exactly). Training RSS cannot increase with model complexity.

**(b) Test RSS:**
**Initially decreases (as model captures real signal), then dramatically increases (as model overfits to training noise).**

Mechanism: The U-shaped test error curve. Low-degree models (degree 1-2) have high bias — they underfit and have high test error. As degree increases to ~3-5, the model captures more real structure → test RSS decreases. Beyond the optimal degree, the model starts fitting training noise → variance increases → test RSS increases. At degree 15-19, test RSS is catastrophically high — the model has essentially memorized 20 training points and predicts wildly wrong for any new point.

**(c) Condition number κ(ΦᵀΦ):**
**Increases dramatically (potentially to 10^16 or beyond) as degree increases.**

Mechanism: As degree increases, the Vandermonde matrix gains more polynomial columns. For x ∈ [0,1], high-degree columns (x¹⁰ ≈ x¹¹ ≈ x¹²) become nearly identical → ΦᵀΦ becomes nearly singular → its smallest eigenvalue σ_min → 0 → κ = σ_max/σ_min → ∞. Degree-15 polynomial on [0,1] can easily have κ > 10^16 — beyond double-precision floating point accuracy, making the matrix inversion numerically meaningless.

**(d) Weight magnitudes ||w||:**
**Increases dramatically as degree increases.**

Mechanism: High-degree polynomials require large alternating coefficients to produce the tight oscillations needed to pass through all training points. The coefficients engage in massive cancellation — e.g., w₁₀ = +10^6, w₁₁ = -10^6 — so that at the training points they cancel perfectly, but between points they create large oscillations. This is the algebraic manifestation of the Runge Phenomenon — the huge oscillations correspond directly to enormous weight magnitudes.

**Summary table:**

| Degree | Training RSS | Test RSS | Condition κ | Weight ||w|| |
|--------|-------------|---------|-------------|---------|
| 1 | High | Moderate (underfit) | ~1-10 (excellent) | Small |
| 3-5 | Medium | Low (good fit) | Moderate | Moderate |
| 10 | Very Low | High (overfit) | Very large | Large |
| 15-19 | ≈0 | Catastrophic | ~10^16 | Enormous |

**Part 2:**
**Runge Phenomenon, ill-conditioning, and overfitting as three aspects of one problem:**

All three arise from the same root cause: **using high-degree polynomial basis functions whose columns are nearly linearly dependent and whose fitting produces large, alternating coefficients.**

**The unified story:**

**Step 1 — Root cause:** High-degree polynomial on [0,1] → columns of Φ are nearly identical (x¹⁰ ≈ x¹¹) → **ill-conditioning** (κ(ΦᵀΦ) is enormous). This is the mathematical/numerical manifestation.

**Step 2 — Consequence in weight space:** To compensate for nearly identical columns, the optimizer needs enormous, alternating weights to distinguish them → ||w|| becomes huge → **Runge Phenomenon** (large weights mean the polynomial oscillates wildly between data points). This is the geometric manifestation in input space x.

**Step 3 — Consequence for generalization:** The wildly oscillating polynomial (Runge) fits training points exactly but produces wrong predictions everywhere else → **overfitting** (zero training error, catastrophic test error). This is the statistical manifestation.

**Three faces of the same phenomenon:**
- **Ill-conditioning:** visible in the mathematics (tiny eigenvalues of ΦᵀΦ)
- **Runge Phenomenon:** visible in the function plot (wild oscillations)
- **Overfitting:** visible in the train-test error gap (low training RSS, high test RSS)

**How regularization addresses all three simultaneously:**

**Ridge regularization modifies the system to: (ΦᵀΦ + λI)w = Φᵀt**

1. **Fixes ill-conditioning:** Adding λI > 0 increases all eigenvalues by at least λ → σ_min ≥ λ > 0 → κ = σ_max/σ_min ≤ (σ_max + λ)/λ — the condition number is bounded and decreasing in λ. Numerically stable.

2. **Reduces Runge oscillations:** The regularization penalizes large weights → forces ||w|| to be small → polynomial coefficients cannot be enormous and alternating → oscillations between data points are suppressed.

3. **Reduces overfitting:** Smaller weights → simpler, smoother polynomial → generalizes better to test data → test RSS decreases from catastrophic levels toward reasonable values.

Regularization is not three separate fixes — it is **one fix** (penalizing weight magnitude) that simultaneously addresses all three manifestations of the high-degree polynomial problem.

**Part 3:**
**Complete response to "smaller weights = definitely better model":**

"Your conclusion is partially right (regularization helped) but 'definitely better' cannot be guaranteed just from smaller weights. Here is a complete explanation:

**What L2 regularization actually does:**

1. **Shrinks weights toward zero:** Yes, ||w|| decreases with increasing λ. This is mathematically guaranteed.

2. **Reduces overfitting risk:** Smaller weights generally reduce variance — the model is less sensitive to individual training points. This is the desired effect.

3. **Fixes ill-conditioning:** Adding λI to ΦᵀΦ improves the condition number. The weights you compute are now numerically meaningful (not artifacts of floating-point instability). This is a genuine improvement independent of generalization.

**What regularization does NOT guarantee:**

1. **Lower test error is not guaranteed:** Regularization trades bias for variance. If you chose λ too large, you've overcorrected — the model now underfits, with high bias. Smaller weights with λ = 10^10 would give weights ≈ 0 and predict only the mean — terrible performance despite "small weights."

2. **"Better" is not defined by weight magnitude:** A model with smaller weights might have higher bias and worse test performance than one with larger weights. Weight magnitude is a regularization mechanism, not a performance metric.

3. **Optimal λ is not guaranteed without cross-validation:** Without systematically evaluating different λ values on held-out data, you don't know if your λ is too small (still overfit), optimal (best generalization), or too large (underfit).

**What you should actually do:**

1. **Train with multiple λ values:** Try λ ∈ {0.001, 0.01, 0.1, 1, 10, 100, ...}

2. **Cross-validate each λ:** For each λ, compute K-fold cross-validated test MSE. This gives a reliable estimate of actual generalization performance.

3. **Plot the validation curve:** Plot λ (x-axis) vs. cross-validated test MSE (y-axis). Expect a U-shape: too small λ = overfit (high test MSE due to variance), too large λ = underfit (high test MSE due to bias), optimal λ = minimum test MSE.

4. **Select optimal λ:** Choose λ* that minimizes cross-validated test MSE (optionally with the one-standard-error rule for a simpler model).

5. **Report test MSE, not weight magnitude:** The only reliable measure of 'better model' is lower error on held-out data.

**Summary:** Regularization is a tool, not a guarantee. Smaller weights indicate the regularization is active, but whether the regularized model is actually better depends on whether the chosen λ improved the Bias-Variance balance — and that can only be determined through cross-validation on test data."

---

## Complete Answer Reference Table

| Question | Core Concepts | Difficulty | Exam Priority |
|----------|--------------|------------|---------------|
| Q1 | Basis functions, why linear models fail, three basis types | Medium | Very High |
| Q2 | MLE framework, Gaussian noise, log-likelihood | Hard | Very High |
| Q3 | Precision parameter, noise variance estimation | Medium | High |
| Q4 | Design matrix, Vandermonde, multicollinearity | Medium-Hard | Very High |
| Q5 | Condition number, ill-conditioning vs overfitting | Hard | Very High |
| Q6 | Runge Phenomenon, global vs local basis | Hard | High |
| Q7 | Forward/backward selection, hierarchy, extrapolation | Medium | High |
| Q8 | Regularization formulation, Ridge closed-form, weight decay | Hard | Very High |
| Q9 | L1 vs L2, geometric argument for sparsity | Very Hard | Very High |
| Q10 | Cross-topic synthesis | Very Hard | Medium |

---

## Top 6 Most Likely Exam Questions From This Topic

1. **L1 vs L2 Regularization** — what each does to parameters, sparse vs. shrunk, when to use each (mirrors Q2 parts 2&3 from original exam exactly)
2. **Basis functions** — why we need them, what types exist, write the generalized model
3. **MLE = Least Squares** — show the equivalence, explain the Gaussian noise assumption
4. **Ill-conditioning** — explain condition number, why high-degree polynomials cause it, distinguish from overfitting
5. **Runge Phenomenon** — what it is, why polynomial basis is global, why this causes oscillations
6. **Regularized Normal Equations** — write w_ridge = (ΦᵀΦ + λI)⁻¹Φᵀt, explain what λI adds

**Send the next slides and I will build the complete exam for those topics too!**