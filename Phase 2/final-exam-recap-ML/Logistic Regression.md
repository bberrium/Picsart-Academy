- [[#Comprehensive Practice Exam WITH ANSWERS|Comprehensive Practice Exam WITH ANSWERS]]
	- [[#Comprehensive Practice Exam WITH ANSWERS#Logistic Regression — Complete Question Bank|Logistic Regression — Complete Question Bank]]
- [[#SECTION 1: From Regression to Classification|SECTION 1: From Regression to Classification]]
		- [[#Logistic Regression — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Logistic Regression — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 2: Optimization for Logistic Regression|SECTION 2: Optimization for Logistic Regression]]
		- [[#Logistic Regression — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Logistic Regression — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 3: Multiclass Logistic Regression|SECTION 3: Multiclass Logistic Regression]]
		- [[#Logistic Regression — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 4: Classification Approaches|SECTION 4: Classification Approaches]]
		- [[#Logistic Regression — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 5: Fisher's Linear Discriminant|SECTION 5: Fisher's Linear Discriminant]]
		- [[#Logistic Regression — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 6: Least Squares for Classification|SECTION 6: Least Squares for Classification]]
		- [[#Logistic Regression — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 7: Classification Metrics (Deep Dive)|SECTION 7: Classification Metrics (Deep Dive)]]
		- [[#Logistic Regression — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Logistic Regression — Complete Question Bank#✅ Answer|✅ Answer]]
	- [[#SECTION 7: Classification Metrics (Deep Dive)#BONUS CHALLENGE QUESTIONS|BONUS CHALLENGE QUESTIONS]]
		- [[#BONUS CHALLENGE QUESTIONS#✅ Answer|✅ Answer]]
	- [[#SECTION 7: Classification Metrics (Deep Dive)#Complete Answer Reference Table|Complete Answer Reference Table]]
	- [[#SECTION 7: Classification Metrics (Deep Dive)#Top 7 Most Likely Exam Questions From This Topic|Top 7 Most Likely Exam Questions From This Topic]]

# Comprehensive Practice Exam WITH ANSWERS
## Logistic Regression — Complete Question Bank

---

# SECTION 1: From Regression to Classification

---

**Question 1**
Understanding why linear regression cannot be directly applied to classification.

1. Explain precisely why linear regression is unsuitable for binary classification. What mathematical property of linear regression outputs makes it fundamentally incompatible with probability estimation?
2. Explain the role of the sigmoid (logistic) function in solving this problem. Write its formula, list its three key properties, and explain how each property makes it suitable for probability modeling.
3. Write the complete logistic regression model. Explain what p(y=1|x) = σ(wᵀx + b) means in plain English and describe what the decision boundary looks like geometrically.

### ✅ Answer

**Part 1:**
**Why linear regression fails for classification:**

Linear regression predicts: y = wᵀx + b, which outputs values in (-∞, +∞) — any real number.

**The fundamental incompatibility:**
Probabilities must lie in [0, 1]. Linear regression can output:
- Negative values (e.g., -3.7) — impossible as a probability
- Values greater than 1 (e.g., 4.2) — impossible as a probability
- Values that grow without bound as input magnitude increases

**Additional problems with linear regression for classification:**

1. **Violates Bernoulli assumption:** The target y ∈ {0, 1} is discrete. Linear regression assumes continuous Gaussian residuals — the distribution assumption is completely wrong for binary data.

2. **Sensitive to outliers in the wrong way:** A training point far from the decision boundary pulls the regression line toward it, distorting the boundary for all other points. In regression, outliers affect predictions everywhere.

3. **No probabilistic interpretation:** Even if you clip outputs to [0,1], the model was not trained to produce calibrated probabilities — the values between 0 and 1 have no valid probability interpretation.

4. **Non-constant variance:** For binary outcomes, variance of the outcome is p(1-p) which varies with x. Linear regression assumes constant variance (homoscedasticity) — another violated assumption.

**Part 2:**
**The Sigmoid Function:**

σ(x) = 1 / (1 + e⁻ˣ)

**Three key properties and why each matters:**

1. **Output always between 0 and 1:**
As x → +∞: e⁻ˣ → 0 → σ(x) → 1/(1+0) = 1
As x → -∞: e⁻ˣ → ∞ → σ(x) → 1/∞ = 0
For any finite x: 0 < σ(x) < 1

This guarantees the output is always a valid probability — regardless of how large or small wᵀx + b becomes, the sigmoid maps it to the unit interval.

2. **Smooth and differentiable everywhere:**
σ'(x) = σ(x)(1 - σ(x)) — a beautiful closed-form derivative that is always positive and bounded.

This allows gradient-based optimization (gradient descent) to work smoothly. There are no discontinuities, kinks, or undefined points. The loss function inherits this differentiability.

3. **Monotonically increasing:**
σ'(x) > 0 for all x — the sigmoid always increases as its input increases.

This means higher values of wᵀx + b always correspond to higher probability of class 1. The model's confidence in class 1 increases monotonically with the linear score — a natural and interpretable behavior.

**Part 3:**
**Complete logistic regression model:**

p(y=1|x) = σ(wᵀx + b) = 1 / (1 + exp(-(wᵀx + b)))

**Plain English interpretation:**
"The probability that input x belongs to class 1 is given by applying the sigmoid function to the linear combination of x's features. The weight vector w determines how much each feature contributes to the class 1 score, and b is a bias offset."

- wᵀx + b: The "linear score" — how strongly the features point toward class 1 (positive) or class 0 (negative)
- σ(·): Squashes this score into [0,1] as a calibrated probability

**Decision rule:**
- If p > 0.5 → predict class 1
- If p < 0.5 → predict class 0
- p = 0.5 occurs when wᵀx + b = 0 (because σ(0) = 0.5)

**Decision boundary geometry:**
The decision boundary is the set of all x where p(y=1|x) = 0.5, which occurs at:
wᵀx + b = 0

This is a **linear equation in x** — a hyperplane in D-dimensional space (a line in 2D, a plane in 3D). Despite using the nonlinear sigmoid function, the decision boundary is still **linear in x**. The sigmoid transforms predictions into probabilities but does not curve the boundary — the boundary remains the set of points where the linear score equals zero.

---

**Question 2**
The probabilistic foundation of logistic regression.

1. Explain the Bernoulli distribution assumption in logistic regression. Write p(y|x) using Bernoulli notation and explain why this is the correct distributional assumption for binary classification.
2. Derive the log-likelihood for N independently drawn observations. Explain every step from the joint probability to the final log-likelihood expression.
3. Explain Cross-Entropy Loss (Log Loss). Write the formula and explain its key intuition: "strongly penalizes confident wrong predictions." Give a numerical example.

### ✅ Answer

**Part 1:**
**Bernoulli distribution assumption:**

For binary classification y ∈ {0, 1}, each output follows a **Bernoulli distribution**:

p(y|x) = p^y × (1-p)^(1-y)

Where p = σ(wᵀx + b) = p(y=1|x)

**Verification this works:**
- When y = 1: p(y=1|x) = p¹ × (1-p)⁰ = p ✓ (probability of class 1)
- When y = 0: p(y=0|x) = p⁰ × (1-p)¹ = 1-p ✓ (probability of class 0)

**Why Bernoulli is the correct assumption:**
The Bernoulli distribution is exactly the distribution for a single binary random variable — a fair or unfair coin flip. Binary classification targets y ∈ {0,1} are precisely binary random variables. The probability p varies with x (the sigmoid maps each x to its own p), but the distributional family (Bernoulli) is appropriate for binary outcomes.

**Contrast with linear regression:** Linear regression assumes y = wᵀx + ε where ε ~ N(0, σ²) — a Gaussian error. For binary y, this Gaussian assumption is completely wrong: binary outcomes have no Gaussian distribution, and residuals have the "wrong" distribution. Bernoulli is the principled choice for binary outputs.

**Part 2:**
**Derivation of log-likelihood:**

**Assuming independence:** observations {(xₙ, yₙ)} are drawn independently.

**Likelihood (joint probability):**
L(w) = Πₙ₌₁ᴺ p(yₙ|xₙ) = Πₙ₌₁ᴺ pₙ^(yₙ) × (1-pₙ)^(1-yₙ)

Where pₙ = σ(wᵀxₙ + b)

**Log-likelihood (take logarithm of both sides):**
ℓ(w) = log L(w) = log Πₙ pₙ^(yₙ)(1-pₙ)^(1-yₙ)

Product becomes sum under log:
= Σₙ [yₙ log(pₙ) + (1-yₙ) log(1-pₙ)]

**Why log-likelihood:**
1. Converts product to sum (easier to differentiate)
2. Prevents numerical underflow from multiplying many small probabilities
3. log is monotonically increasing — maximizing ℓ(w) gives same w* as maximizing L(w)

**Part 3:**
**Cross-Entropy Loss:**

We **maximize** log-likelihood ↔ **minimize** negative log-likelihood:

L(w) = -ℓ(w) = -Σₙ [yₙ log(pₙ) + (1-yₙ) log(1-pₙ)]

Or per-sample: L = -(y log(p) + (1-y) log(1-p))

**Key intuition — strongly penalizes confident wrong predictions:**

Case 1: True label y=1, prediction p=0.99 (correct and confident)
Loss = -(1×log(0.99) + 0) = -log(0.99) ≈ **0.01** (tiny loss)

Case 2: True label y=1, prediction p=0.5 (uncertain)
Loss = -(1×log(0.5)) = -log(0.5) ≈ **0.693** (moderate loss)

Case 3: True label y=1, prediction p=0.01 (wrong and VERY confident)
Loss = -(1×log(0.01)) = -log(0.01) ≈ **4.605** (enormous loss!)

The loss grows as log(1/p) when the true class is 1. As p→0 while y=1, log(p)→-∞ and loss→+∞. A model that says "I'm 99% sure it's class 0" when it's actually class 1 is penalized enormously.

**Why this is the right behavior:**
We don't just want the model to get the labels right — we want **calibrated probability estimates**. A model that always predicts p=0.51 is technically "correct" (crosses the 0.5 threshold) but shows very low confidence in the right direction. Cross-entropy loss rewards high confidence in the right class and severely punishes high confidence in the wrong class — encouraging the model to produce well-calibrated probabilities, not just correct labels.

---

# SECTION 2: Optimization for Logistic Regression

---

**Question 3**
Unlike linear regression, logistic regression has no closed-form solution.

1. Explain why setting ∇_w L(w) = 0 (where L is the cross-entropy loss) yields a nonlinear system with no analytical solution. Contrast this with linear regression's Normal Equations.
2. Write the gradient of the cross-entropy loss for logistic regression. Explain the beautiful form of this gradient and what it means intuitively.
3. Write the gradient descent update rule for logistic regression. Explain every component and what "iterative" means in this context.

### ✅ Answer

**Part 1:**
**Why no closed-form solution:**

The cross-entropy loss for logistic regression:
L(w) = -Σₙ [yₙ log(σ(wᵀxₙ+b)) + (1-yₙ) log(1-σ(wᵀxₙ+b))]

Taking the gradient and setting to zero:
∇_w L = Σₙ (σ(wᵀxₙ+b) - yₙ) xₙ = 0

This requires: Σₙ σ(wᵀxₙ+b) xₙ = Σₙ yₙ xₙ

**The nonlinearity problem:**
σ(wᵀxₙ+b) = 1/(1+exp(-(wᵀxₙ+b))) is a **nonlinear function of w**. The equation Σₙ σ(wᵀxₙ+b) xₙ = Σₙ yₙ xₙ cannot be rearranged to isolate w algebraically — the sigmoid is "wrapped around" w in a way that cannot be undone with simple linear algebra.

**Contrast with linear regression:**
For linear regression: ∇_w L = Xᵀ(Xw - y) = 0
This gives: XᵀXw = Xᵀy — a **linear system** in w
Solution: w* = (XᵀX)⁻¹Xᵀy — closed form, computable in one step.

For logistic regression: the gradient equation is nonlinear in w — no closed form. Must use iterative optimization.

**However:** Despite no closed form, the loss IS convex in w — there is a unique global minimum with no local minima. Gradient descent is guaranteed to find it.

**Part 2:**
**Gradient of cross-entropy loss:**

∇_w L(w) = Σₙ (σ(wᵀxₙ+b) - yₙ) xₙ

= Σₙ (pₙ - yₙ) xₙ

**The beautiful form:**
The gradient is simply the sum over all training points of the **prediction error** (pₙ - yₙ) times the **feature vector** xₙ.

This has exactly the same structural form as the gradient for linear regression:
∇_w L_linear = Σₙ (wᵀxₙ - yₙ) xₙ

The only difference is: for linear regression, the "prediction" is the raw linear output wᵀxₙ. For logistic regression, the "prediction" is the sigmoid-transformed output pₙ = σ(wᵀxₙ).

**Intuitive interpretation:**
Each training point xₙ "votes" on which direction to adjust w:
- If pₙ > yₙ (predicted too high): the gradient contribution (pₙ-yₙ)xₙ is positive → gradient descent subtracts this → w moves to predict lower probability for similar x
- If pₙ < yₙ (predicted too low): (pₙ-yₙ) is negative → w moves to predict higher probability
- If pₙ = yₙ (perfect prediction): zero contribution — this point doesn't change w

The gradient is zero only when all predictions match the true labels — the perfect model.

**Part 3:**
**Gradient descent update rule:**

w^(t+1) = w^(t) - η × ∇_w L(w^(t))
= w^(t) - η × Σₙ (σ(wᵀ^(t)xₙ + b) - yₙ) xₙ

Every component:
- **w^(t):** Current weight vector at iteration t
- **η (eta):** The **learning rate** — a small positive scalar controlling step size
- **∇_w L(w^(t)):** The gradient of the loss at current weights — points in the direction of steepest increase of loss
- **- η × ∇_w L:** Subtracting the scaled gradient moves w in the direction of steepest DECREASE — toward lower loss
- **w^(t+1):** Updated weight vector after one step

**What "iterative" means:**
Starting from random initialization w^(0), we repeatedly:
1. Compute predictions pₙ = σ(w^(t)ᵀxₙ + b) for all training points
2. Compute the gradient = Σₙ(pₙ - yₙ)xₙ
3. Update w by stepping in the negative gradient direction
4. Repeat until convergence (gradient ≈ 0, or loss stops decreasing)

Unlike linear regression (one matrix operation), logistic regression requires many iterations — typically hundreds or thousands — before converging to the optimal weights.

---

**Question 4**
Gradient Descent: learning rate and convergence.

1. Explain the role of the learning rate η. What happens in each of the three problematic cases: η too large, η too small, and η just right?
2. Explain the two mathematical requirements a function must satisfy for gradient descent to be applicable. Why does the cross-entropy loss satisfy both?
3. Explain the difference between Batch Gradient Descent, Stochastic Gradient Descent (SGD), and Mini-Batch Gradient Descent. What are the tradeoffs of each?

### ✅ Answer

**Part 1:**
**Role of learning rate η:**

The gradient ∇L(w) tells us the direction to move, but not how far. The learning rate η scales the step size.

**η too large (diverges or oscillates):**
- Each step "overshoots" the minimum — the algorithm jumps past the optimal point to the other side
- The loss oscillates: step left of minimum → gradient points right → step right past minimum → gradient points left → oscillates indefinitely
- In severe cases, the loss INCREASES with each step → **divergence** — the algorithm flies off to infinity
- On a quadratic loss, divergence occurs when η > 2/λ_max (where λ_max is the largest eigenvalue of the Hessian)

**η too small:**
- Each step is tiny — the algorithm slowly crawls toward the minimum
- Convergence is achieved but requires an enormous number of iterations
- Computationally wasteful — each iteration computes the gradient (expensive) for negligible progress
- Can get "stuck" in flat regions where the gradient is small — taking forever to escape

**η just right (Goldilocks learning rate):**
- Steps are large enough to make meaningful progress per iteration
- Small enough to not overshoot the minimum
- Converges in a reasonable number of iterations
- The loss decreases consistently and smoothly toward the minimum

**Practical solution:** Adaptive learning rate methods (Adam, RMSProp) automatically adjust η during training. Learning rate schedules start large (fast initial progress) and decay over time (fine-tuning near minimum).

**Part 2:**
**Two requirements for gradient descent:**

**Requirement 1 — Differentiability:**
Gradient descent requires computing ∂L/∂wⱼ at every step. The function must be differentiable — a well-defined gradient must exist at every point.

**Why cross-entropy satisfies this:**
L(w) = -Σₙ[yₙ log(σ(wᵀxₙ+b)) + (1-yₙ)log(1-σ(wᵀxₙ+b))]

The sigmoid σ is infinitely differentiable. The log function is differentiable for positive arguments (σ always outputs values in (0,1), so log(σ) and log(1-σ) are always defined). The composition of differentiable functions is differentiable. ✓

**Requirement 2 — Convexity:**
Convexity guarantees that any local minimum is the global minimum — gradient descent won't get stuck in a suboptimal solution.

**Why cross-entropy is convex:**
The negative log-likelihood for logistic regression can be shown to be a convex function of w. Specifically:
- The Hessian ∇²L(w) = Σₙ pₙ(1-pₙ)xₙxₙᵀ
- Since pₙ(1-pₙ) > 0 and xₙxₙᵀ is positive semidefinite, the Hessian is positive semidefinite
- A function with positive semidefinite Hessian is convex ✓

This is a crucial advantage — gradient descent is guaranteed to find the global optimum for logistic regression (unlike neural networks where the loss is non-convex).

**Part 3:**
**Three variants of gradient descent:**

**Batch Gradient Descent:**
Update rule: w ← w - η × (1/N) Σₙ₌₁ᴺ ∇Lₙ(w)
Uses ALL N training points to compute one gradient update.

Pros: Stable, smooth convergence; exact gradient direction; can use large learning rates
Cons: Very slow for large N (must process all N points per step); requires all data in memory; no way to escape saddle points through noise

**Stochastic Gradient Descent (SGD):**
Update rule: w ← w - η × ∇Lₙ(w) (for one randomly chosen example n)
Uses ONE randomly chosen training point per update.

Pros: Very fast per update; can handle online (streaming) data; noise helps escape local minima/saddle points; memory efficient
Cons: Very noisy updates — gradient direction is a noisy estimate; may never fully converge, oscillating around minimum; requires smaller, often decaying learning rate

**Mini-Batch Gradient Descent (Industry Standard):**
Update rule: w ← w - η × (1/B) Σₙ∈batch ∇Lₙ(w)
Uses a random subset (batch) of B examples (typically B=32, 64, 128, 256).

Pros: 
- Balances noise (good for escaping local regions) and stability (less chaotic than SGD)
- Leverages GPU parallelism — modern GPUs excel at matrix operations on batches
- Moderate memory requirement
- Faster convergence than batch GD with large N

Cons: Adds hyperparameter B (batch size) to tune; still noisier than batch GD

This is the standard in all modern deep learning and large-scale ML.

---

# SECTION 3: Multiclass Logistic Regression

---

**Question 5**
Logistic regression extends naturally to multiple classes.

1. Explain the problem formulation for multiclass classification. How does the output change from binary classification and what mathematical challenge arises?
2. Write the Softmax function formula. Explain its three properties and why it is described as a "generalization of sigmoid to multiple classes."
3. Write the multiclass cross-entropy loss. Explain every component including the one-hot encoding yᵢₖ.

### ✅ Answer

**Part 1:**
**Multiclass classification problem:**

Binary logistic regression: p(y=1|x) ∈ [0,1] — one probability suffices (p(y=0|x) = 1 - p(y=1|x))

Multiclass with K classes: We need probabilities for each class k = 1,...,K:
p(y=1|x), p(y=2|x), ..., p(y=K|x)

**Mathematical challenge:**
These K probabilities must:
1. Each be in [0,1] — valid probabilities
2. Sum to exactly 1 — they represent a complete probability distribution over classes

Simply applying sigmoid to K separate linear scores doesn't guarantee they sum to 1. We need a function that:
- Outputs K positive values
- Normalizes them to sum to 1
- Preserves the ordering (higher score → higher probability)

The solution: compute a **score** for each class using a separate weight vector wₖ, then normalize using the Softmax function.

**Part 2:**
**Softmax Function:**

p(y=k|x) = exp(wₖᵀx) / Σⱼ₌₁ᴷ exp(wⱼᵀx)

**Three properties:**

1. **Outputs are all positive:**
exp(·) is always positive for any real input. Since numerator and denominator are both positive, each output is positive. No probability can be negative. ✓

2. **Sum of probabilities = 1:**
Σₖ p(y=k|x) = Σₖ exp(wₖᵀx) / Σⱼ exp(wⱼᵀx) = [Σₖ exp(wₖᵀx)] / [Σⱼ exp(wⱼᵀx)] = 1 ✓
The normalization by the sum of all exponentials ensures they sum to exactly 1.

3. **Higher score → higher probability:**
Since exp is monotonically increasing, the class with the highest score wₖᵀx gets the highest probability. The softmax amplifies differences (via exponentiation) and normalizes.

**Connection to sigmoid (why it's a generalization):**
For K=2 with scores w₁ᵀx and w₂ᵀx:
p(y=1|x) = exp(w₁ᵀx) / (exp(w₁ᵀx) + exp(w₂ᵀx))
= 1 / (1 + exp(-(w₁-w₂)ᵀx))
= σ((w₁-w₂)ᵀx)

This is exactly the sigmoid of the difference in scores — so binary logistic regression with sigmoid is a special case of multiclass logistic regression with softmax for K=2.

**Part 3:**
**Multiclass cross-entropy loss:**

L(W) = -Σᵢ₌₁ᴺ Σₖ₌₁ᴷ yᵢₖ log(p(y=k|xᵢ))

Every component:
- **N:** Number of training examples
- **K:** Number of classes
- **yᵢₖ:** **One-hot encoding** — equals 1 if sample i belongs to class k, 0 otherwise. For each sample i, exactly one yᵢₖ = 1 (the true class) and all others are 0.
- **p(y=k|xᵢ):** The softmax probability that sample i belongs to class k
- **log(p(y=k|xᵢ)):** Log probability — heavily penalizes low probability for the true class
- **yᵢₖ × log(pᵢₖ):** Only the term for the TRUE class (k where yᵢₖ=1) contributes — all other terms are zero

**Simplification using one-hot encoding:**
Since only one yᵢₖ = 1 per sample (the true class tᵢ), the double sum simplifies:
L(W) = -Σᵢ log(p(y=tᵢ|xᵢ))

The loss is just the negative log probability of the TRUE class for each sample — we want to maximize the model's confidence in the correct class.

**Prediction:** class = argmaxₖ p(y=k|x) — simply pick the class with highest softmax probability.

---

# SECTION 4: Classification Approaches

---

**Question 6**
Classification problems can be approached in multiple fundamentally different ways.

1. Explain the difference between directly predicting a class label vs. estimating class membership probabilities first. Why does estimating probabilities first have "many advantages"?
2. Explain the distinction between the "direct" approach (discriminative) and the "generative" approach for finding probabilities p(Cₖ|x). What does each model, and how does each use the training data?
3. Explain what decision boundaries and decision regions are. Why are the decision regions of a K-class linear discriminant function always "singly connected and convex"?

### ✅ Answer

**Part 1:**
**Direct class prediction vs. probability estimation:**

**Direct approach:** Build y(x) → {C₁, C₂, ..., Cₖ} directly — the function outputs a class label with no intermediate probability estimate. Example: Perceptron (outputs sign of wᵀx, no probability).

**Probability-first approach:** Build p(Cₖ|x) → [0,1] for each class k, then decide the class as argmaxₖ p(Cₖ|x).

**Advantages of probability-first:**

1. **Optimal decision-making under asymmetric costs:** If missing a cancer diagnosis (false negative) costs 100× more than a false alarm, you should predict "cancer" whenever p(cancer|x) > 0.01 (not 0.5). With only class labels, you cannot make this adjustment. With probabilities, you adjust the threshold.

2. **Calibrated uncertainty:** p(y=1|x) = 0.52 tells you the model is very uncertain — act cautiously. p(y=1|x) = 0.999 tells you the model is very confident — act decisively. Class labels alone convey no uncertainty.

3. **Combining with other information:** If you have two independent classifiers, you can combine their probabilities multiplicatively. Class labels cannot be combined this way.

4. **Reject option:** In safety-critical systems, you can say "if p < 0.9 for all classes, don't classify — ask a human." Impossible with direct label prediction.

5. **Downstream probabilistic models:** Probability outputs can feed into other probabilistic models as inputs. Class labels cannot.

**Part 2:**
**Discriminative vs. Generative approaches:**

**Direct (Discriminative) approach:**
Models p(Cₖ|x) directly — learns the decision boundary in input space.
- Asks: "Given input x, what is the probability it belongs to class k?"
- Uses training data to directly optimize the mapping x → p(Cₖ|x)
- Examples: Logistic Regression, SVM, Neural Networks

**Generative approach:**
Models p(x|Cₖ) (the distribution of x within each class) and p(Cₖ) (class priors). Then uses Bayes' theorem to recover p(Cₖ|x):
p(Cₖ|x) = p(x|Cₖ)p(Cₖ) / p(x)

- Asks: "What does data from class k look like?" and "How common is class k?"
- Uses training data to model the distribution of features within each class
- Examples: Naive Bayes, Linear Discriminant Analysis, Gaussian Mixture Models

**Key difference:**
- Discriminative: directly learns the boundary between classes
- Generative: learns what each class looks like, then derives the boundary from class models

Generative models can generate new synthetic data samples; discriminative models typically cannot.

**Part 3:**
**Decision boundaries and convexity of decision regions:**

**Decision boundary:** The set of input points x where two or more class scores are equal — the boundary between decision regions. For a K-class linear discriminant, the boundary between classes Cₖ and Cⱼ is: yₖ(x) = yⱼ(x) — a hyperplane.

**Decision region:** The set of all inputs assigned to class Cₖ — all x where yₖ(x) > yⱼ(x) for all j ≠ k.

**Why decision regions are singly connected and convex:**

**Proof from the slides:**
Take any two points xA and xB that both lie inside decision region Rₖ (both classified as Cₖ). This means: yₖ(xA) > yⱼ(xA) and yₖ(xB) > yⱼ(xB) for all j ≠ k.

For any point x̂ on the line segment between xA and xB:
x̂ = λxA + (1-λ)xB for λ ∈ [0,1]

For linear discriminant functions yⱼ(x) = wⱼᵀx + wⱼ₀:
yₖ(x̂) = yₖ(λxA + (1-λ)xB) = λyₖ(xA) + (1-λ)yₖ(xB)

Since λ, (1-λ) ≥ 0, yₖ(xA) > yⱼ(xA), and yₖ(xB) > yⱼ(xB):
yₖ(x̂) = λyₖ(xA) + (1-λ)yₖ(xB) > λyⱼ(xA) + (1-λ)yⱼ(xB) = yⱼ(x̂)

Therefore x̂ ∈ Rₖ — any point on the line between two points in Rₖ is also in Rₖ. This is exactly the definition of a **convex set**. ✓

**Why this matters practically:**
Convex decision regions are geometrically simple — you can always draw a straight path between any two correctly classified points of the same class without leaving the region. This makes the classifier predictable and well-behaved. Non-convex regions (from nonlinear classifiers) can have "islands" and "holes" that make behavior less predictable.

---

# SECTION 5: Fisher's Linear Discriminant

---

**Question 7**
Fisher's Linear Discriminant (FLD) approaches classification through dimensionality reduction.

1. Explain the core idea of Fisher's Linear Discriminant. What is it projecting, and what is the initial naive approach, and why does it fail?
2. Explain the Fisher Criterion J(w). Write its formula, define every component, and explain why it captures "large separation with small within-class variance" as a ratio.
3. Derive or explain the optimal projection direction for FLD. What is the final result and what does it say about the optimal direction relative to the class means and within-class scatter?

### ✅ Answer

**Part 1:**
**Core idea of Fisher's Linear Discriminant:**

**What it projects:**
FLD projects high-dimensional data points x ∈ ℝᴰ onto a single scalar y = wᵀx (1D projection). The goal is to find the projection direction w that best separates the classes in 1D.

**The naive approach — maximize projected class mean separation:**
Choose w to maximize m₂ - m₁ where mₖ = wᵀμₖ (projected class means, μₖ = class centroids).

**Why the naive approach fails:**
Maximizing mean separation alone ignores the spread of each class. Consider:
- Class means are far apart (good projected mean separation)
- But each class has huge variance — the projected distributions completely overlap

A projection direction where the means are far apart but both classes spread over the entire line provides no discrimination ability. You can always increase mean separation arbitrarily by scaling w — the naive criterion is unbounded.

**The additional problem (from slides):**
Even with unit-constrained w ∝ (μ₂ - μ₁), if the class distributions have strong non-diagonal covariance (they are elongated in different directions), projecting onto the between-class axis still produces overlapping distributions. The direction connecting the class means is not necessarily the best discriminating direction.

**Part 2:**
**Fisher Criterion:**

J(w) = (m₁ - m₂)² / (s₁² + s₂²) = wᵀSBw / wᵀSww

Where:

**Numerator — between-class variance:**
(m₁ - m₂)² = (wᵀ(μ₁ - μ₂))² = wᵀ SB w

SB = (μ₂ - μ₁)(μ₂ - μ₁)ᵀ — the **between-class scatter matrix**

This measures how far apart the projected class means are. Large numerator → well-separated classes in the projected 1D space.

**Denominator — within-class variance:**
sₖ² = Σₙ∈Cₖ (wᵀxₙ - mₖ)² — variance of projected class k

s₁² + s₂² = wᵀ Sw w where Sw = Σₖ Σₙ∈Cₖ (xₙ - μₖ)(xₙ - μₖ)ᵀ — the **within-class scatter matrix**

This measures how spread out each class is around its own mean in the projected space. Small denominator → compact, tight clusters.

**Why ratio captures the goal:**
J(w) = (separation of means)² / (spread within classes)

A high J(w) requires BOTH:
- Large (m₁-m₂)² — well-separated means after projection
- Small (s₁²+s₂²) — compact classes after projection

A direction that separates means but produces overlapping distributions gives small J (large denominator). A direction that doesn't separate means gives small J (small numerator). Only directions that genuinely separate the classes while keeping them compact give high J — exactly what we want for discrimination.

**Part 3:**
**Optimal projection direction:**

Differentiating J(w) with respect to w and setting to zero:

The condition for maximum is: **SB w = (wᵀSBw / wᵀSww) × Sw w**

Noting that SBw = (μ₂-μ₁)(μ₂-μ₁)ᵀw is always in the direction of (μ₂-μ₁) (it is a rank-1 matrix — any vector multiplied by it gives a scalar times (μ₂-μ₁)):

(μ₂-μ₁)[(μ₂-μ₁)ᵀw] = λ Sw w

Where λ = wᵀSBw / wᵀSww is a scalar.

Multiplying both sides by Sw⁻¹ (assuming Sw is invertible):

**w ∝ Sw⁻¹(μ₂ - μ₁)**

**Fisher's Linear Discriminant solution:**

The optimal projection direction is the **inverse within-class scatter of the difference in class means**.

**Interpretation:**
- (μ₂ - μ₁): The direction pointing from class 1 mean to class 2 mean — the "naive" direction
- Sw⁻¹: **Whitens** the data by the within-class scatter — it accounts for how the classes are spread and rotates the direction to better account for the actual geometry of the class distributions

If the within-class covariance is isotropic (Sw = I), then w ∝ (μ₂-μ₁) — same as the naive approach. But if the classes are elongated (non-spherical), Sw⁻¹ compensates by rotating the projection away from directions of high within-class variance.

---

# SECTION 6: Least Squares for Classification

---

**Question 8**
Least squares can also be applied to classification problems.

1. Explain the Least Squares classification approach. How does it differ from Logistic Regression in what it minimizes?
2. Explain the 1-of-K coding scheme for multiclass classification. Why is this encoding necessary and what does each element of the target vector represent?
3. The slides show that Least Squares for classification fails on certain datasets. What fundamental limitation of least squares causes it to produce poor classification boundaries in some cases?

### ✅ Answer

**Part 1:**
**Least Squares for Classification:**

In K-class classification with 1-of-K coding, each class Cₖ has its own linear model:
yₖ(x) = wₖᵀx + wₖ₀

Stacking all K models: y(x) = Wᵀx̃ where W is a (D+1) × K weight matrix

**What it minimizes:**
Least Squares minimizes the **sum-of-squares error** between the model output y(x) and the target vector t (1-of-K coded):

min_W Σₙ ||tₙ - Wᵀx̃ₙ||²

Setting derivative to zero: W* = (X̃ᵀX̃)⁻¹X̃ᵀT

Where T is the N×K target matrix (each row is a 1-of-K encoded class label).

**Discriminant function:** Assign x to class Cₖ if yₖ(x) = max_j yⱼ(x)

**Contrast with Logistic Regression:**
- **Logistic Regression** minimizes **cross-entropy loss** — specifically designed for probability estimation, connected to maximum likelihood under Bernoulli assumption
- **Least Squares** minimizes **squared error** between binary targets {0,1} and continuous outputs — a regression approach applied to classification

Least Squares treats the 0/1 targets as continuous values and finds the hyperplane that minimizes the squared distance from these targets. It is not derived from a principled probabilistic model for classification.

**Part 2:**
**1-of-K coding scheme:**

For K classes, the target for a sample from class Cⱼ is a binary vector t ∈ {0,1}ᴷ where:
- tⱼ = 1 (the true class)
- tₖ = 0 for all k ≠ j

**Example with K=5, sample from class 2:**
t = (0, 1, 0, 0, 0)ᵀ

**Why necessary:**
We need to train K separate linear models (one per class) simultaneously. A single integer label (y ∈ {1,...,K}) cannot be directly used because:
1. It implies an ordering (class 3 is "between" classes 2 and 4) that doesn't exist
2. It implies equal spacing (the "distance" from class 1 to class 2 equals the distance from class 4 to class 5) which is meaningless for categories
3. With a single integer, least squares minimizes (yₙ - ŷₙ)² which treats the classes as points on a number line

The 1-of-K (one-hot) encoding:
- Makes all K classes equally "far" from each other
- Allows training K separate binary classifiers simultaneously
- The matrix formulation W*ᵀx gives K scores, one per class, without assuming ordering

**Part 3:**
**Fundamental limitation of least squares for classification:**

**The outlier masking problem:**
Least squares is designed to minimize the distance between continuous predictions and targets. But classification targets are binary (0 or 1), and when a class has points scattered far from the decision boundary, the squared error from those distant points pulls the fitted boundary toward them.

**The "masking" failure mode:**
With three or more classes, least squares can produce a decision boundary that fails to separate a class entirely — the class gets "masked." This occurs because:

1. Least Squares tries to predict the target value 1 for class 1 and 0 for other classes
2. When data points of one class are on the "far" side of another class, the least squares solution finds a compromise that predicts intermediate values for these distant points
3. The optimal squared-error solution might assign no region of the input space to a class that exists in the data

**Why cross-entropy loss doesn't have this problem:**
Cross-entropy loss is derived from the correct probabilistic model for discrete outcomes. It specifically rewards confident correct classifications and penalizes confident wrong classifications — it is calibrated for the binary/categorical nature of the targets. Least squares was designed for continuous regression problems and does not have this calibration for discrete outputs.

---

# SECTION 7: Classification Metrics (Deep Dive)

---

**Question 9**
Application of classification metrics to real scenarios.

1. A model for detecting COVID-19 achieves: TP=85, FP=200, TN=9700, FN=15 on a test set of 10,000 patients (1% prevalence). Calculate Accuracy, Precision, Recall, F1 Score, Specificity, and FPR. Interpret each in the context of COVID detection.
2. You are building a spam filter. The cost of a false positive (legitimate email classified as spam) is 10× more serious than a false negative (spam reaching inbox). Which metric should you optimize and why? What threshold adjustment would you make?
3. Explain why Precision and Recall are inversely related at a fundamental level. Use a concrete scenario where trying to maximize Recall leads to near-zero Precision and vice versa.

### ✅ Answer

**Part 1:**
**Calculations:**

Dataset: 10,000 patients, 100 COVID-positive (1%), 9,900 COVID-negative
Results: TP=85, FP=200, TN=9700, FN=15

**Accuracy:**
= (TP+TN)/(TP+FP+TN+FN) = (85+9700)/10000 = 9785/10000 = **97.85%**

Interpretation: "97.85% of patients were correctly classified." Sounds impressive. But even predicting everyone as negative gives 99% accuracy! This metric is misleading here.

**Precision:**
= TP/(TP+FP) = 85/(85+200) = 85/285 = **29.8%**

Interpretation: "Of the 285 patients flagged as COVID-positive, only 30% actually have COVID. 70% are false alarms." Every positive prediction causes quarantine, isolation, and anxiety — 70% of this is unnecessary.

**Recall (Sensitivity):**
= TP/(TP+FN) = 85/(85+15) = 85/100 = **85%**

Interpretation: "Of 100 actual COVID patients, we correctly identified 85. 15 sick patients were sent home. In a pandemic context, these 15 missed cases each spread disease to many others."

**F1 Score:**
= 2×(Precision×Recall)/(Precision+Recall) = 2×(0.298×0.85)/(0.298+0.85) = 2×0.253/1.148 = **0.441**

Interpretation: Poor overall balance — the model has okay Recall but terrible Precision.

**Specificity (True Negative Rate):**
= TN/(TN+FP) = 9700/(9700+200) = 9700/9900 = **98.0%**

Interpretation: "Of 9,900 healthy patients, 98% were correctly cleared." Most healthy patients are correctly handled.

**FPR (False Positive Rate):**
= FP/(FP+TN) = 200/9900 = **2.0%**

Interpretation: "2% of healthy patients were incorrectly flagged as COVID-positive." 200 healthy people sent into quarantine unnecessarily.

**Overall clinical assessment:**
This model is moderately useful. Good at clearing healthy people (98% specificity), decent at catching COVID cases (85% recall), but very imprecise — for every real COVID case it catches, it also creates 2.4 false alarms. In a pandemic, the 85% recall is valuable but the 15 missed cases are critical. Need to improve recall by lowering threshold, accepting even more false alarms.

**Part 2:**
**Spam filter with asymmetric costs:**

False Positive (ham → spam): 10× more costly — important emails are lost/missed
False Negative (spam → inbox): 1× cost — annoying spam gets through

**Optimize: Precision**

Reasoning: We want to minimize False Positives (legitimate emails wrongly classified as spam). Precision = TP/(TP+FP) measures exactly this — what fraction of spam predictions are actually spam.

**Threshold adjustment:**
Increase the classification threshold τ above 0.5 — only classify an email as spam when the model is VERY confident it's spam (e.g., τ = 0.9 or 0.95).

Effect:
- Fewer emails classified as spam overall
- Those classified as spam are more reliably actual spam (higher Precision)
- More spam reaches inbox (lower Recall) — acceptable since this is less costly
- Fewer legitimate emails are misclassified as spam (fewer FP) — the primary goal

**Formal approach:**
Define utility: U = -10×FP - 1×FN
Optimize threshold τ to maximize U on validation data. The optimal τ will be higher than 0.5 (closer to the "spam" class boundary) to minimize the more costly FP errors.

**Part 3:**
**Fundamental inverse relationship:**

At any threshold τ, the model classifies all examples with score ≥ τ as positive. As τ decreases (predict positive more aggressively):

**Extreme: τ = 0 (predict EVERYTHING positive)**
- TP = all true positives = total positive count
- FN = 0 (no positive is missed — everything is predicted positive)
- **Recall = TP/(TP+0) = 1.0 ← MAXIMUM RECALL**
- FP = all actual negatives (everything incorrectly gets positive prediction)
- **Precision = TP/(TP + total negatives) ≈ class prevalence ← NEAR ZERO for rare class**

For 1% prevalence: Precision = 1% while Recall = 100%

**Extreme: τ = 1 (predict ALMOST NOTHING positive)**
- Only the most extreme high-score examples get positive prediction
- These few examples are almost certainly actually positive
- **Precision → 1.0 ← MAXIMUM PRECISION** (when you predict positive, you're almost always right)
- But you miss almost all actual positives
- **Recall → 0 ← NEAR ZERO RECALL**

**Why fundamental:**
There is a fixed number of actual positives in the data. Recall = TP/total_positives. To increase Recall, you must increase TP — classify more examples as positive. But each new example you add to the positive prediction set is less likely to be a true positive (you've already classified the high-confidence cases). So FP grows faster than TP as you lower the threshold, and Precision = TP/(TP+FP) falls.

The only way to have both perfect Precision and perfect Recall simultaneously is a perfect model that assigns high scores to all positives and low scores to all negatives — which requires zero classification error.

---

**Question 10**
ROC Curves in depth.

1. Explain how a ROC curve is constructed step by step. What computations must be performed for each threshold point?
2. An AUC of 0.5 means random performance. But a classifier that always predicts the majority class on an imbalanced dataset also achieves high accuracy. Explain why AUC correctly handles imbalance while accuracy does not.
3. You have two models: Model A has AUC=0.85, Model B has AUC=0.78. In a high-stakes medical screening application requiring at least 95% Recall, which model is better? How would you determine this?

### ✅ Answer

**Part 1:**
**ROC curve construction:**

**Step 1:** Train the model and obtain probability scores for all test examples.

**Step 2:** Sort all test examples by their predicted probability score (descending).

**Step 3:** For each possible threshold τ (sweeping from high to low, or equivalently, going down the sorted list one example at a time):
- Classify all examples with score ≥ τ as positive, all others as negative
- Build the confusion matrix: count TP, FP, TN, FN
- Compute: TPR = TP/(TP+FN) and FPR = FP/(FP+TN)
- Plot the point (FPR, TPR) on the graph

**Step 4:** Connect all points with line segments.

**Special points:**
- τ = 1 (nothing classified positive): Point (0, 0) — bottom-left corner
- τ = 0 (everything classified positive): Point (1, 1) — top-right corner
- Diagonal line from (0,0) to (1,1): The "random classifier" baseline

**Step 5:** Compute AUC = area under the resulting curve (numerically via trapezoidal rule or Mann-Whitney U statistic).

**Interpretation of the curve shape:**
A good classifier bows toward the top-left corner — achieving high TPR at low FPR. The curve "hugs" the top-left corner for a good model.

**Part 2:**
**Why AUC handles imbalance correctly:**

Consider an imbalanced dataset: 990 negatives, 10 positives (1% prevalence).

**Accuracy on classifier that always predicts negative:**
- TP=0, FP=0, TN=990, FN=10
- Accuracy = 990/1000 = **99%** ← misleadingly excellent

**AUC on classifier that always predicts negative:**
- At every threshold: TPR = 0/10 = 0 (catches no positives)
- At every threshold: FPR = 0/990 = 0 (never false alarms on negatives)
- But since all scores are identical (all predicted "negative" = 0), the ROC "curve" is just the single point (0,0)
- **AUC = 0.5** ← correctly identifies as no-better-than-random

**Why the difference:**
- **Accuracy** counts: (990 correct negatives + 0 correct positives) / 1000 = 99%. The 990 correct negatives overwhelm the 10 missed positives.
- **TPR** measures: correct positives / total positives = 0/10 = 0%. Completely blind to the class imbalance — it measures performance WITHIN the positive class.
- **FPR** measures: false alarms / total negatives = 0/990 = 0%. Performance WITHIN the negative class.
- AUC is based on TPR and FPR — both measured within their respective classes, automatically normalizing for class size. A classifier that ignores the positive class has TPR=0 regardless of how many negatives exist.

**Part 3:**
**Model comparison with operating constraint:**

The question "which model is better" with the constraint "at least 95% Recall" cannot be answered from AUC alone. AUC is a summary of performance across ALL thresholds — but we need performance at a SPECIFIC operating point.

**How to determine which is better:**

1. **Plot both ROC curves** on the same graph

2. **Find the operating point for each model at TPR = 0.95:**
   - On Model A's ROC curve, find the point where TPR = 0.95 → read the corresponding FPR value (say FPR_A)
   - On Model B's ROC curve, find the point where TPR = 0.95 → read FPR_B

3. **Compare FPR values at the required TPR:**
   - Lower FPR at TPR=0.95 = better model for this application
   - The model that achieves 95% recall with fewer false alarms is superior

**Why AUC alone is insufficient:**
Model A (AUC=0.85) is better overall across all thresholds. But it's possible that:
- Model A at 95% Recall achieves FPR=0.40
- Model B at 95% Recall achieves FPR=0.25

In this case, Model B is actually better for this specific operating requirement even though its overall AUC is lower. High AUC means good average performance — but at a specific critical operating point, the actual ROC curve shape matters.

**The correct answer depends on the curve shape at TPR=0.95.** Generally (but not always), higher AUC implies better performance at any specific operating point — so Model A is likely better, but confirmation requires examining the actual ROC curves at TPR=0.95.

---

## BONUS CHALLENGE QUESTIONS

---

**Question 11**
Cross-topic synthesis.

1. Connect Logistic Regression to Linear Regression, SVM, and the Perceptron. All four are linear classifiers or can be used for binary classification. Compare them on: what they optimize, what the decision boundary looks like, whether they produce probabilities, and when to use each.
2. A logistic regression model achieves 99% accuracy on a dataset with 99% negative examples. Compute the confusion matrix if the model predicts everything as negative. Calculate Precision, Recall, F1, and AUC. Explain why this model is useless despite high accuracy.
3. You have 10,000 training examples and want to train logistic regression. Compare: (a) using the closed-form Normal Equations (not applicable), (b) Batch Gradient Descent, (c) Mini-Batch SGD with batch size 32. Analyze each in terms of computational cost per update, convergence behavior, memory requirements, and practical recommendation.

### ✅ Answer

**Part 1:**
**Comparison of four linear classifiers:**

| Property | Logistic Regression | Linear Regression (for classification) | SVM | Perceptron |
|----------|--------------------|-----------------------------------------|-----|-----------|
| **What it optimizes** | Cross-entropy (max likelihood) | Sum of squared errors | Margin maximization | Mistake-driven updates |
| **Loss function** | -Σ[y log p + (1-y) log(1-p)] | Σ(y - wᵀx)² | (1/2)||w||² + C Σξₙ | No explicit loss |
| **Decision boundary** | Linear hyperplane (wᵀx+b=0) | Linear hyperplane | Maximum margin hyperplane | Any separating hyperplane |
| **Produces probabilities** | Yes (calibrated via sigmoid) | Only with heuristic clipping | No (distances, not probabilities) | No |
| **Handles non-separable data** | Yes (naturally) | Yes (but wrong model) | Yes (soft margin, C parameter) | No (loops forever) |
| **Unique solution** | Yes (convex, unique global min) | Yes (Normal Equations) | Yes (unique maximum margin) | No (depends on initialization/order) |
| **Training** | Iterative (gradient descent) | Closed-form or iterative | Quadratic programming | Iterative (online) |
| **Theoretical justification** | Max likelihood (Bernoulli) | Max likelihood (Gaussian) | VC theory (margin bounds) | Convergence theorem (separable data) |

**When to use each:**
- **Logistic Regression:** Default for binary classification, need probabilities, interpretable coefficients, moderate data size
- **Linear Regression for classification:** Generally avoid — use logistic instead
- **SVM:** Maximum generalization guarantee, small-medium datasets, non-linear data with kernel trick, don't need probabilities
- **Perceptron:** Historical interest; for large-scale online learning with simple updates

**Part 2:**
**Model that predicts everything negative (99% negative dataset, N=10,000):**

Dataset: 100 positive, 9,900 negative

Confusion matrix (all predicted negative):
- TP = 0 (no positives caught)
- FP = 0 (never predicted positive)
- TN = 9,900 (all negatives correctly predicted)
- FN = 100 (all positives missed)

**Accuracy = (0+9900)/10000 = 99%** ← deceptively excellent

**Precision = TP/(TP+FP) = 0/0 = undefined** (never predicts positive, so precision is undefined)

**Recall = TP/(TP+FN) = 0/(0+100) = 0%** ← captures zero actual positives

**F1 = 2×(P×R)/(P+R) = 0** (since R=0, F1=0 regardless of P)

**AUC:**
Since the model assigns the same score (0 or "negative") to every example, there is no threshold that changes performance — the ROC "curve" is just the diagonal:
- At threshold τ: TPR = 0, FPR = 0 for τ=1
- Since all scores are identical, we can't sweep through thresholds meaningfully
- **AUC = 0.5** ← correctly indicates random performance

**Why useless:**
- **F1 = 0:** The model catches exactly zero positive cases — it completely fails at the primary task
- **Recall = 0%:** Every single positive example is missed
- **AUC = 0.5:** No discriminative ability — equivalent to random guessing

The 99% accuracy is entirely due to the model correctly classifying the overwhelming majority class (negatives) while completely ignoring the minority class. This is the classic imbalanced dataset accuracy trap — the metric rewards the model for doing the easy thing (always predicting the common class) while providing no measure of performance on the actually important (positive) class.

**Part 3:**
**Training comparison for N=10,000:**

**(a) Closed-form Normal Equations:**
Not applicable for logistic regression — the gradient equation is nonlinear in w (because of the sigmoid). No closed-form solution exists. Unlike linear regression where setting the gradient to zero gives a linear system solvable as w* = (XᵀX)⁻¹Xᵀy, logistic regression's gradient equation Σₙ σ(wᵀxₙ)xₙ = Σₙ yₙxₙ cannot be algebraically solved for w.

**(b) Batch Gradient Descent:**
- **Computational cost per update:** O(N×D) = O(10,000×D) — must process all 10,000 examples to compute one gradient and take one step. Very expensive per step.
- **Convergence:** Smooth, stable convergence. Gradient direction is exact (uses all data). Large learning rates possible without instability. Monotonically decreasing loss (for appropriate η).
- **Memory:** Must hold all N=10,000 examples in memory simultaneously. For D features: O(N×D) memory.
- **Practical issue:** Each "update" uses all the data — for 1,000 iterations, you process 10M examples. Slow overall despite stable per-step behavior.

**(c) Mini-Batch SGD (batch size B=32):**
- **Computational cost per update:** O(B×D) = O(32×D) — only process 32 examples per update. 312× cheaper per step than batch GD.
- **Convergence:** Noisier than batch GD but practically faster overall. One "epoch" (all data processed once) = 10,000/32 ≈ 313 updates. If batch GD needs 1,000 iterations (each seeing all data), mini-batch needs ~313,000 updates for the same "epochs" of data — but converges much faster in wall-clock time.
- **Memory:** O(B×D) per step — only hold 32 examples in memory at once. Much more memory efficient than batch GD.
- **GPU efficiency:** Modern GPUs are optimized for matrix operations on batches. B=32 to B=256 leverages GPU parallelism optimally.
- **Noise benefit:** Random sampling introduces noise that can escape shallow local regions and saddle points in more complex loss landscapes.

**Practical recommendation:**
**Mini-Batch SGD with B=32-256 is the clear winner** for logistic regression with N=10,000:
- ~10-100× faster per parameter update than batch GD
- Full GPU utilization
- Memory efficient
- Converges to good solution in practice
- The slight noise from mini-batching is beneficial or neutral for convex problems like logistic regression
- Industry standard for all scale of ML problems

For N=10,000, batch GD is actually feasible and has the advantage of exact gradients. For N > 1,000,000, batch GD becomes completely impractical and mini-batch SGD is mandatory.

---

## Complete Answer Reference Table

| Question | Core Concepts | Difficulty | Exam Priority |
|----------|--------------|------------|---------------|
| Q1 | Sigmoid, logistic model, decision boundary | Medium | Very High |
| Q2 | Bernoulli assumption, log-likelihood, cross-entropy | Hard | Very High |
| Q3 | No closed-form, gradient formula, update rule | Hard | Very High |
| Q4 | Learning rate, gradient descent requirements, variants | Medium | High |
| Q5 | Softmax, multiclass CE loss, one-hot encoding | Medium | Very High |
| Q6 | Classification approaches, discriminative vs generative | Hard | High |
| Q7 | Fisher's LDA, Fisher criterion, optimal direction | Very Hard | High |
| Q8 | Least squares for classification, 1-of-K coding | Medium | Medium |
| Q9 | Full metric calculation and interpretation | Medium | Very High |
| Q10 | ROC construction, AUC vs accuracy, operating constraints | Hard | Very High |
| Q11 | Cross-topic synthesis | Very Hard | Medium |

---

## Top 7 Most Likely Exam Questions From This Topic

1. **Why accuracy fails on imbalanced data** — concrete example showing 99% accuracy with 0% recall (mirrors Q3 from original exam exactly)
2. **Precision vs Recall** — tradeoff, when to prioritize each, cancer vs wine examples (mirrors Q3 from original exam)
3. **Sigmoid function** — formula, three properties, why needed for logistic regression
4. **Cross-entropy loss** — formula, why it strongly penalizes confident wrong predictions
5. **Logistic regression vs linear regression** — why linear regression fails for classification, how sigmoid fixes it
6. **Gradient descent** — learning rate effects, requirements for applicability, update rule
7. **F1 Score** — formula, why harmonic mean, when to use vs individual metrics

**Send the next slides and I will build the complete exam for those topics too!**