- [[#Comprehensive Practice Exam WITH ANSWERS|Comprehensive Practice Exam WITH ANSWERS]]
	- [[#Comprehensive Practice Exam WITH ANSWERS#Generative Modeling — Complete Question Bank|Generative Modeling — Complete Question Bank]]
- [[#SECTION 1: Two Philosophies of Machine Learning|SECTION 1: Two Philosophies of Machine Learning]]
		- [[#Generative Modeling — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 2: Generative vs. Discriminative Classifiers|SECTION 2: Generative vs. Discriminative Classifiers]]
		- [[#Generative Modeling — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 3: Naive Bayes|SECTION 3: Naive Bayes]]
		- [[#Generative Modeling — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 4: Gaussian Discriminant Analysis|SECTION 4: Gaussian Discriminant Analysis]]
		- [[#Generative Modeling — Complete Question Bank#✅ Answer|✅ Answer]]
		- [[#Generative Modeling — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 5: LDA vs. QDA vs. RDA|SECTION 5: LDA vs. QDA vs. RDA]]
		- [[#Generative Modeling — Complete Question Bank#✅ Answer|✅ Answer]]
- [[#SECTION 6: GDA vs. Logistic Regression|SECTION 6: GDA vs. Logistic Regression]]
		- [[#Generative Modeling — Complete Question Bank#✅ Answer|✅ Answer]]
	- [[#SECTION 6: GDA vs. Logistic Regression#BONUS CHALLENGE QUESTIONS|BONUS CHALLENGE QUESTIONS]]
		- [[#BONUS CHALLENGE QUESTIONS#✅ Answer|✅ Answer]]
	- [[#SECTION 6: GDA vs. Logistic Regression#Complete Answer Reference Table|Complete Answer Reference Table]]
	- [[#SECTION 6: GDA vs. Logistic Regression#Top 7 Most Likely Exam Questions From This Topic|Top 7 Most Likely Exam Questions From This Topic]]

# Comprehensive Practice Exam WITH ANSWERS
## Generative Modeling — Complete Question Bank

---

# SECTION 1: Two Philosophies of Machine Learning

---

**Question 1**
Machine learning can be approached through two fundamentally different philosophical lenses.

1. Explain the Geometric School (Empirical Risk Minimization) and the Statistical School (Probabilistic Modeling). For each, describe its core philosophy, the language it uses, and give two examples of algorithms from each school.
2. Explain how Linear Regression demonstrates that both philosophies can lead to identical mathematical results. Walk through the statistical derivation showing that assuming Gaussian noise and applying MLE produces exactly the least squares objective.
3. Explain Vapnik's Principle and its three justifications for why the geometric approach was taught first. For each justification, explain the underlying reasoning.

### ✅ Answer

**Part 1:**
**Two philosophical schools:**

**The Geometric School (Empirical Risk Minimization):**
Core philosophy: We do not ask how the universe generated the data. We only care about finding boundaries, surfaces, or functions that minimize a penalty (loss function) on the observed data. The data is taken as given; our goal is purely to draw the best separating line or fit the best curve.

Language: Distances, margins, hyperplanes, gradients, loss functions, decision boundaries.

Examples: Support Vector Machines (maximize the margin between classes), Decision Trees (minimize impurity at each split), standard Logistic Regression (minimize cross-entropy loss), Linear Regression (minimize sum of squared distances).

**The Statistical School (Probabilistic Modeling):**
Core philosophy: We assume the data is the OUTPUT of a hidden random process — there is an underlying probability distribution generating what we observe. We model these distributions and find parameters that make our observations most likely (Maximum Likelihood). The data is evidence about an underlying generative mechanism.

Language: Maximum Likelihood Estimation, prior distributions, variance, covariance, Bayes' Theorem, posterior probabilities, marginal distributions.

Examples: Naive Bayes (models the joint distribution of features and class), Linear Discriminant Analysis (assumes Gaussian class-conditional distributions), Gaussian Mixture Models (models data as mixture of Gaussians).

**Part 2:**
**Linear Regression — two schools, identical result:**

**Geometric view:**
Find w that minimizes: Σₙ(yₙ - wᵀxₙ)² — the sum of squared residuals. Direct, no probabilistic assumptions.

**Statistical view:**
Assume: yₙ = wᵀxₙ + εₙ where εₙ ~ N(0, σ²)

This means: yₙ|xₙ ~ N(wᵀxₙ, σ²)

The likelihood of observing one data point:
p(yₙ|xₙ, w) = (1/√(2πσ²)) × exp(-(yₙ - wᵀxₙ)²/(2σ²))

The joint likelihood for all N independent observations:
L(w) = Πₙ p(yₙ|xₙ, w) = Πₙ (1/√(2πσ²)) × exp(-(yₙ - wᵀxₙ)²/(2σ²))

Log-likelihood:
ℓ(w) = log L(w) = -N/2 × log(2πσ²) - (1/(2σ²)) × Σₙ(yₙ - wᵀxₙ)²

**Maximizing ℓ(w) with respect to w:**
The first term is a constant (doesn't depend on w). The second term: -(1/(2σ²)) × Σₙ(yₙ - wᵀxₙ)²

Maximizing this = minimizing +Σₙ(yₙ - wᵀxₙ)²

This is EXACTLY the least squares objective! The Gaussian noise assumption and the squared penalty are mathematically identical.

**Takeaway:** The Gaussian distribution's exponent contains a squared term — taking the negative log eliminates the exponential and cancels the constants, leaving the squared error. This is not a coincidence: the Gaussian is the natural distribution associated with squared error loss.

**Part 3:**
**Vapnik's Principle and three justifications:**

Vapnik's famous principle: "When solving a problem of interest, do not solve a more general problem as an intermediate step."

**Justification 1 — Problem Difficulty:**
Estimating the full joint distribution P(x, y) requires modeling the entire data-generating mechanism — a vastly harder problem than just learning the conditional P(y|x) needed for classification/regression. For classification, we only need to know which class is more likely — we don't need to know how the data was generated in all its detail. Starting with generative models would be solving a much harder problem to get to the same destination. Direct discriminative approaches (SVMs, logistic regression) take the most direct path.

**Justification 2 — Deep Learning Bridge:**
The entire modern deep learning revolution is built on empirical risk minimization — define a loss, compute gradients, update parameters via backpropagation. By spending the course learning how to define loss functions, optimize them via gradient descent, understand the bias-variance tradeoff, and select hyperparameters via cross-validation, students acquired the exact mathematical toolkit needed for neural networks. The probabilistic framework is a different language; the optimization framework is universal.

**Justification 3 — Curse of Dimensionality:**
Generative models must estimate probability distributions over the feature space. In high dimensions, this requires exponentially more data to estimate reliably (the curse of dimensionality). Naive Bayes partially avoids this with independence assumptions, but more sophisticated generative models like QDA break down when features > examples. Geometric models like SVMs with the Kernel Trick and ensemble methods handle thousands of features natively through their optimization objectives without estimating full distributions. Teaching the scalable approaches first ensures students can work with real-world high-dimensional data.

---

# SECTION 2: Generative vs. Discriminative Classifiers

---

**Question 2**
Generative and discriminative classifiers embody the two philosophical approaches.

1. Explain precisely the difference between what a generative classifier learns vs. what a discriminative classifier learns. Write the mathematical objects each models.
2. Explain the classification procedure for a generative classifier using Bayes' Theorem. Starting from P(X, G), show how to compute P(G|X) for classification.
3. Explain the practical tradeoffs: when does a generative classifier outperform a discriminative one, and when does a discriminative classifier win? Connect this to assumptions, data quantity, and data quality.

### ✅ Answer

**Part 1:**
**What each type of classifier learns:**

**Generative Classifier:**
Models the joint distribution P(X, G) where X are features and G is the class label. Equivalently, models:
- P(X|G=k): The class-conditional distribution — "what does data from class k look like?"
- P(G=k): The prior probability of each class — "how common is class k?"

From these, it derives P(G|X) via Bayes' Theorem. The generative model can answer: "Given I'm generating data from class k, what feature values are likely?" — this is why it's called "generative" — it models the data generation process for each class.

**Discriminative Classifier:**
Models P(G|X) directly — "given features X, what is the probability of each class?" It doesn't model P(X) or P(X|G) at all — it never asks how the features were generated, only which class they predict.

**Mathematical objects:**
- Generative: Learns P(X|G=k) and P(G=k) separately → derives P(G|X)
- Discriminative: Directly learns P(G|X) from data (e.g., logistic regression learns w such that σ(wᵀx) ≈ P(G=1|X=x))

**Part 2:**
**Classification using Bayes' Theorem:**

Given learned P(X|G) and P(G), classify using:

P(G=k|X=x) = P(X=x|G=k) × P(G=k) / P(X=x)

Where: P(X=x) = Σₖ P(X=x|G=k) × P(G=k) (normalizing constant)

**Decision rule:**
Assign x to class k* = argmaxₖ P(G=k|X=x)
= argmaxₖ [P(X=x|G=k) × P(G=k)] (the normalizing denominator P(X=x) is the same for all classes, so it doesn't affect argmax)

**Practical computation:**
1. Estimate P(G=k) from training data: proportion of class k examples
2. Estimate P(X|G=k) using a distributional assumption (Gaussian for LDA/QDA, product of marginals for Naive Bayes)
3. For a new point x: evaluate P(X=x|G=k) × P(G=k) for all k, predict the class with highest product

**Part 3:**
**When each type wins:**

**Generative wins when:**
1. **Small data:** Generative models incorporate structural assumptions (e.g., Gaussianity) that reduce the number of parameters to estimate. With 50 training examples, a generative model's distributional assumption acts as strong regularization — the model knows the "shape" of the distribution. A discriminative model would overfit.

2. **Correct distributional assumption:** If data genuinely follows the assumed distribution (e.g., truly Gaussian features), the generative model extracts all available information efficiently. LDA is optimal when data is Gaussian — it finds the perfect boundary faster (with less data) than logistic regression.

3. **Interpretability of distributions:** When you need to generate new data, detect anomalies, or understand what each class "looks like."

**Discriminative wins when:**
1. **Large data:** With sufficient data, discriminative models learn the true boundary without assuming a specific distribution. Their "no assumption" approach pays off when data is plentiful.

2. **Wrong distributional assumption:** If data is skewed, has outliers, or doesn't match the assumed distribution, the generative model's assumptions are actively harmful. Logistic regression with no distributional assumptions adapts to arbitrary data shapes — it will wildly outperform LDA on non-Gaussian data.

3. **Only the boundary matters:** If you only care about classification accuracy (not generation, anomaly detection, or probability calibration), the discriminative model takes the most direct path to the optimal boundary without the overhead of modeling the full distribution.

---

# SECTION 3: Naive Bayes

---

**Question 3**
Naive Bayes is the simplest generative classifier.

1. State the "naive" assumption of Naive Bayes. Write the mathematical simplification it enables and explain what this assumption means geometrically in terms of the covariance matrix.
2. Write the Naive Bayes classification rule. Explain each term and describe the training procedure: what must be estimated from the training data and how?
3. Explain Laplace Smoothing. What specific problem does it solve, why does it occur, and how does the smoothing formula fix it?

### ✅ Answer

**Part 1:**
**The naive assumption:**

The "naive" assumption is **conditional independence of features given the class:**

P(X|G=k) = P(X₁,X₂,...,Xₚ|G=k) = Π_{j=1}^{p} P(Xⱼ|G=k)

In plain English: Knowing the class, each feature provides information about the class independently — no feature pair is correlated with each other beyond what their class membership already explains.

**Why "naive":** This assumption is almost never exactly true in reality. Features are almost always correlated — knowing someone's height tells you something about their weight even within the same class (men of the same demographic). Calling it "naive" acknowledges this assumption's simplicity and likely falsehood.

**Mathematical simplification:** Instead of estimating the full joint distribution P(X₁,...,Xₚ|G=k) — which in general requires 2^p - 1 parameters for binary features, or a full p×p covariance matrix for continuous features — we only need to estimate p separate marginal distributions P(Xⱼ|G=k) for j=1,...,p.

**Geometric/covariance interpretation:**
In the GDA family tree, all methods assume features within a class follow a Multivariate Normal distribution characterized by a covariance matrix Σ. The covariance matrix captures correlations between features:
- Off-diagonal entry Σᵢⱼ measures the correlation between feature i and feature j
- Naive Bayes forces ALL off-diagonal entries to zero: Σ = diag(σ₁², σ₂², ..., σₚ²)

Setting off-diagonal terms to zero = assuming features are uncorrelated (within a class). This is the "diagonal covariance" restriction in the GDA family tree.

**Part 2:**
**Naive Bayes classification rule:**

Predict class: k* = argmaxₖ P(G=k|X=x)
= argmaxₖ P(G=k) × Πⱼ P(Xⱼ=xⱼ|G=k)

(Using Bayes' Theorem and the naive independence assumption, dropping the constant denominator P(X))

**In log form** (practical implementation):
k* = argmaxₖ [log P(G=k) + Σⱼ log P(Xⱼ=xⱼ|G=k)]

Using logs converts products to sums — numerically stable and computationally efficient.

**Each term:**
- **P(G=k):** Prior probability of class k — estimated as (number of class k examples / total training examples)
- **P(Xⱼ=xⱼ|G=k):** Conditional probability of feature j having value xⱼ given class k

**Training procedure — what to estimate:**

For each class k = 1,...,K and each feature j = 1,...,p:

**Categorical features:** Count the frequency of each value of feature j in class k:
P(Xⱼ=v|G=k) = count(Xⱼ=v AND G=k) / count(G=k)

**Continuous features:** Fit a Gaussian to feature j within class k:
P(Xⱼ|G=k) ~ N(μⱼₖ, σ²ⱼₖ)
Estimate: μ̂ⱼₖ = mean of Xⱼ in class k, σ̂²ⱼₖ = variance of Xⱼ in class k

This is Maximum Likelihood Estimation applied to each feature-class combination separately.

**Part 3:**
**Laplace Smoothing:**

**The problem:** Suppose "Free" never appears in any spam email in the training data. Then:
P(Xⱼ = "Free" | G = spam) = 0/count(spam) = 0

When we encounter a test email containing "Free", we compute:
P(spam) × P(Xⱼ₁|spam) × P(Xⱼ₂|spam) × ... × P(Xⱼ="Free"|spam) × ...
= P(spam) × (other terms) × 0 × ... = **0**

One zero count in training DESTROYS the entire probability calculation due to the product rule. The model is absolutely certain "this email is not spam" purely because it never saw "Free" in training spam — but "Free" might be a new spam term not in training data. This is catastrophic overfitting to training set vocabulary.

**Why it occurs:** Small training datasets can't cover every possible feature value. Any unseen feature value gets probability 0, which propagates through the product to make entire class probabilities 0. The "zero frequency problem."

**Laplace Smoothing formula:**
P_smoothed(Xⱼ=v|G=k) = (count(Xⱼ=v AND G=k) + α) / (count(G=k) + α×|V|)

Where:
- α is a small constant (typically α=1 for "add-one smoothing")
- |V| is the number of possible values feature j can take (vocabulary size for text)

**How it fixes the problem:**
- Every count is increased by α → no count is ever 0 → no probability is ever 0
- The denominator increases by α×|V| to maintain proper normalization
- For α=1: a feature seen 0 times gets probability 1/(count(G=k) + |V|) — small but non-zero
- For α=1: a feature seen many times: its probability barely changes (adding 1 to a large count is negligible)

Laplace smoothing is Bayesian regularization — it adds a uniform prior (pseudo-counts) to the MLE estimate, preventing overfitting to zero-count events.

---

# SECTION 4: Gaussian Discriminant Analysis

---

**Question 4**
The GDA family uses multivariate Gaussian distributions as class-conditional densities.

1. Write the Multivariate Normal distribution formula. Explain every component including the covariance matrix Σ, its inverse Σ⁻¹, and its determinant |Σ|. What do the diagonal vs. off-diagonal elements of Σ represent?
2. Explain the GDA family tree: QDA, LDA, Naive Bayes, and RDA. What specific assumption does each make about the covariance matrix, and how does this tradeoff affect bias and variance?
3. Derive why LDA produces LINEAR decision boundaries while QDA produces QUADRATIC ones. Show the mathematical step where LDA's equal-covariance assumption cancels out the quadratic terms.

### ✅ Answer

**Part 1:**
**Multivariate Normal Distribution:**

P(x|μ, Σ) = (2π)^(-p/2) |Σ|^(-1/2) exp(-1/2 × (x-μ)ᵀΣ⁻¹(x-μ))

**Every component:**
- **x:** A p-dimensional feature vector (the data point)
- **μ:** The mean vector — p-dimensional, representing the center of the distribution (average feature values)
- **Σ:** The p×p covariance matrix — captures the spread and correlations of features
- **(x-μ):** The deviation of x from the mean
- **(x-μ)ᵀΣ⁻¹(x-μ):** The Mahalanobis distance squared — a generalized distance that accounts for feature correlations and scales. This is what appears in the exponent.
- **Σ⁻¹:** The precision matrix — scales the distance by the inverse covariance
- **|Σ|^(-1/2):** Normalizing factor ensuring the distribution integrates to 1; |Σ| is the determinant of Σ
- **(2π)^(-p/2):** Another normalizing constant

**Covariance matrix Σ structure:**
- **Diagonal entry Σᵢᵢ = σᵢ²:** The variance of feature i — how spread out feature i is around its mean
- **Off-diagonal entry Σᵢⱼ:** The covariance between feature i and feature j — positive means they tend to be large/small together; negative means they move in opposite directions; zero means they are uncorrelated
- **Σ = diag(σ₁², ..., σₚ²) (all off-diagonal = 0):** Features are uncorrelated — this is the Naive Bayes assumption

**Part 2:**
**GDA family tree:**

**QDA (Quadratic Discriminant Analysis):**
Assumption: Each class k has its OWN covariance matrix Σₖ — no sharing.
Parameters: μₖ (p per class) + Σₖ (p(p+1)/2 per class) → K×p(p+1)/2 covariance parameters total
For p=50, K=10: 50×51/2 × 10 = 12,750 covariance parameters alone!
Bias-Variance: Low Bias (each class has its own shape → can fit any class geometry) / High Variance (too many parameters, needs large N per class to estimate reliably)

**LDA (Linear Discriminant Analysis):**
Assumption: ALL classes share the SAME covariance matrix Σ (pooled across classes)
Parameters: μₖ (p per class) + ONE Σ (p(p+1)/2 total) → dramatically fewer parameters
For p=50, K=10: 1,275 covariance parameters — 10× fewer than QDA
Bias-Variance: High Bias (all classes must have same shape — wrong if they differ) / Low Variance (fewer parameters, stable estimates even with small N)

**Naive Bayes:**
Assumption: Σ is diagonal — all off-diagonal covariance terms are zero (feature independence)
Parameters: μₖ (p per class) + p variances per class → much fewer than LDA
Bias-Variance: Highest Bias (strongest assumption: independence) / Lowest Variance (fewest parameters, works even in very high dimensions with few examples)

**RDA (Regularized Discriminant Analysis):**
Assumption: Σₖ(α) = αΣ̂ₖ + (1-α)Σ̂ — a convex blend of QDA and LDA covariances
Parameter α ∈ [0,1] found by cross-validation
At α=0: exactly LDA; At α=1: exactly QDA; 0 < α < 1: smooth interpolation
Bias-Variance: Tunable via α — cross-validation finds the optimal tradeoff

**Part 3:**
**Why LDA is linear and QDA is quadratic:**

**QDA Decision Boundary:**
Assign x to class k vs. class l if δₖ(x) > δₗ(x), where:

δₖ(x) = log P(G=k|X=x) = -1/2 log|Σₖ| - 1/2(x-μₖ)ᵀΣₖ⁻¹(x-μₖ) + log P(G=k)

The QDA boundary between classes k and l is: δₖ(x) = δₗ(x), or:

-1/2 log|Σₖ| - 1/2(x-μₖ)ᵀΣₖ⁻¹(x-μₖ) + log πₖ = -1/2 log|Σₗ| - 1/2(x-μₗ)ᵀΣₗ⁻¹(x-μₗ) + log πₗ

Rearranging:
xᵀΣₖ⁻¹x - xᵀΣₗ⁻¹x + (lower order terms) = constant

The term **xᵀ(Σₖ⁻¹ - Σₗ⁻¹)x** is a QUADRATIC function of x → quadratic decision boundary.

**LDA Decision Boundary — cancellation of quadratic terms:**

LDA assumes Σₖ = Σₗ = Σ (same covariance for all classes).

Setting Σₖ = Σₗ = Σ in the QDA boundary equation:
xᵀΣ⁻¹x - xᵀΣ⁻¹x = **0**

The quadratic terms cancel exactly! The remaining terms are:
-1/2 × [2xᵀΣ⁻¹μₖ - 2xᵀΣ⁻¹μₗ] + (constants)
= xᵀΣ⁻¹(μₖ - μₗ) + constant

This is a LINEAR function of x → LINEAR decision boundary.

**Physical interpretation:** QDA has different covariance matrices — they create "different shapes" for each class distribution. The boundary between two different-shaped distributions is naturally quadratic (like the intersection of two different ellipses). LDA forces all classes to have the same shape — the boundary between two identically-shaped but differently-centered Gaussians is the perpendicular bisector, which is a straight line.

---

**Question 5**
LDA has a specific mathematical form and geometric interpretation.

1. Write the LDA discriminant function δₖ(x) for class k. Explain every term and describe what each component measures geometrically.
2. Explain the pooled covariance matrix estimation in LDA. What is it, why is it "pooled," and what does it assume about the data?
3. Explain LDA as both a classifier AND a dimensionality reduction method (Fisher's criterion connection). What two objectives must the projection simultaneously achieve?

### ✅ Answer

**Part 1:**
**LDA discriminant function:**

δₖ(x) = xᵀΣ⁻¹μₖ - (1/2)μₖᵀΣ⁻¹μₖ + log P(G=k)

**Every term:**

**xᵀΣ⁻¹μₖ:** The "similarity score" between x and class k's mean μₖ, measured in the metric defined by Σ⁻¹. This is related to the Mahalanobis inner product — it asks "how much does x project onto the direction of class k's center, accounting for the shape of the distribution?" Higher value = x is more "aligned" with class k.

**(1/2)μₖᵀΣ⁻¹μₖ:** A normalization term — the "self-similarity" of class k's mean. It corrects for the fact that classes with means further from the origin would otherwise get artificially high δₖ scores.

**log P(G=k):** The log prior probability of class k. If class k is more common (larger P(G=k)), this adds a positive term — the classifier is biased toward predicting more frequent classes. If classes are equally likely (P(G=k) = 1/K for all k), this term is the same for all classes and cancels in comparisons.

**Combined interpretation:**
δₖ(x) measures how well x "belongs" to class k: it's high when x is close to μₖ (in Mahalanobis distance) AND class k is common. Assign x to the class with highest δₖ(x).

**Part 2:**
**Pooled covariance matrix:**

The pooled covariance matrix Σ̂ is estimated by combining (pooling) the within-class scatter matrices across all K classes:

Σ̂ = (1/(N-K)) × Σₖ Σ_{xᵢ∈class k} (xᵢ - μ̂ₖ)(xᵢ - μ̂ₖ)ᵀ

= (1/(N-K)) × Σₖ (Nₖ - 1) × Σ̂ₖ

Where Σ̂ₖ is the sample covariance within class k, Nₖ is the size of class k, and N-K are the total degrees of freedom.

**Why "pooled":**
The individual within-class sample covariance matrices Σ̂₁, Σ̂₂,...,Σ̂ₖ are "pooled" (combined as a weighted average) into one global estimate. The weight for each class is proportional to its number of degrees of freedom (Nₖ - 1).

**What it assumes:**
Pooling assumes ALL classes have the SAME underlying true covariance matrix Σ — the individual Σ̂ₖ are just different noisy estimates of the same truth. By pooling, we use ALL N-K training examples to estimate one covariance matrix rather than N₁-1 examples for class 1, N₂-1 for class 2, etc. — much more statistically efficient when the assumption is valid.

If classes truly have different covariance structures, pooling introduces bias — but it dramatically reduces variance. This is the fundamental LDA tradeoff.

**Part 3:**
**LDA as dimensionality reduction:**

LDA simultaneously achieves two objectives for a linear projection w:

1. **Maximize between-class separation:** The projected class means wᵀμₖ should be as far apart as possible — different classes should cluster in different regions of the projected space.

2. **Minimize within-class scatter:** The projected variance within each class should be small — examples within the same class should cluster tightly in the projected space.

**The Fisher criterion formalizes this:**
J(w) = (Between-class variance) / (Within-class variance) = wᵀSBw / wᵀSww

Where:
- Sᴮ = Σₖ Nₖ(μₖ - μ)(μₖ - μ)ᵀ (between-class scatter matrix)
- Sᵥ = Σₖ Σᵢ∈ₖ (xᵢ - μₖ)(xᵢ - μₖ)ᵀ (within-class scatter matrix)

The optimal projection w maximizing J(w) is the eigenvector of Sw⁻¹SB corresponding to the largest eigenvalue — which produces the same classifier as LDA applied to compute discriminant functions δₖ(x).

**Dimensionality reduction application:**
For K classes, LDA can project data to at most K-1 dimensions (since Sᴮ has rank at most K-1). This is much more efficient than PCA for labeled classification problems because LDA finds the directions that are most useful for CLASS DISCRIMINATION, not just maximum variance (which PCA finds). A single LDA dimension often captures classification information better than many PCA dimensions.

---

# SECTION 5: LDA vs. QDA vs. RDA

---

**Question 6**
Understanding when to use LDA vs. QDA vs. RDA requires understanding the bias-variance tradeoff in this context.

1. Explain why QDA "breaks down" when a class has fewer training examples than the number of features (Nₖ < p). What specific mathematical operation fails and why?
2. Explain Regularized Discriminant Analysis (RDA) as a bias-variance tradeoff mechanism. Write the shrinkage formula and explain what happens at α=0, α=1, and intermediate values.
3. Explain how RDA with a second parameter γ can produce Naive Bayes as a limiting case. What does the second shrinkage do geometrically?

### ✅ Answer

**Part 1:**
**Why QDA breaks down when Nₖ < p:**

QDA requires computing Σ̂ₖ⁻¹ — the inverse of the within-class sample covariance matrix.

**The sample covariance matrix:** Σ̂ₖ is estimated from Nₖ training examples in class k. This matrix is p×p.

**The rank limitation:** A p×p matrix estimated from Nₖ vectors can have rank at most min(Nₖ-1, p). When Nₖ < p (fewer training examples than features):

rank(Σ̂ₖ) ≤ Nₖ - 1 < p

The matrix is **rank deficient** — it does not span the full p-dimensional space. There are directions in feature space along which the class k distribution has zero estimated variance (because there were not enough training examples to "see" variation in those directions).

**The failure:** A rank-deficient matrix is **singular** — its determinant is zero — so Σ̂ₖ⁻¹ does not exist. The QDA discriminant function δₖ(x) requires computing Σₖ⁻¹, but this inverse is mathematically undefined.

**Intuition:** With p=100 features and Nₖ=50 examples, you're trying to estimate a 100×100 matrix with 5,050 unique parameters from only 50 data points. The matrix is statistically meaningless — any estimated covariance is noise. Inverting it amplifies this noise to infinity.

**Part 2:**
**RDA shrinkage formula:**

Σ̂ₖ(α) = (1-α)Σ̂ + αΣ̂ₖ

Where:
- Σ̂ₖ is the class-specific QDA covariance matrix for class k
- Σ̂ is the pooled LDA covariance matrix (same for all classes)
- α ∈ [0, 1] is the shrinkage hyperparameter

**At α = 0:**
Σ̂ₖ(0) = Σ̂ → ALL classes use the SAME pooled covariance → exactly **LDA**. Maximum shrinkage toward the pooled estimate. Lowest variance, highest bias.

**At α = 1:**
Σ̂ₖ(1) = Σ̂ₖ → each class uses its OWN covariance → exactly **QDA**. No shrinkage. Highest variance, lowest bias. May fail if Nₖ < p.

**Intermediate α ∈ (0,1):**
Each class's covariance is a weighted blend between the class-specific estimate and the pooled estimate. The decision boundary is curved (like QDA) but constrained and smoothed (like LDA). The boundary complexity is between a straight line (α=0) and a quadratic curve (α=1).

**How to choose α:**
Cross-validation on a validation set. Setting α on training data would always prefer α → 1 (more flexibility → lower training error) → overfitting. Cross-validation estimates test performance, revealing the optimal tradeoff for generalization.

**Why shrinkage works mathematically:**
Even if Σ̂ₖ is singular (rank deficient), Σ̂ is typically full rank (estimated from all N examples). The blend Σ̂ₖ(α) = (1-α)Σ̂ + αΣ̂ₖ inherits the full-rankness of Σ̂ for α < 1 — the singularity of Σ̂ₖ is "healed" by mixing in the full-rank pooled matrix.

**Part 3:**
**RDA with second parameter γ — shrinking toward Naive Bayes:**

RDA can apply a second shrinkage parameter γ that shrinks the pooled covariance Σ̂ toward a purely diagonal matrix:

Σ̂(γ) = (1-γ)Σ̂ + γ × diag(Σ̂)

Where diag(Σ̂) keeps only the diagonal elements of Σ̂ (setting all off-diagonal covariances to zero).

**Geometric effect:**
- Off-diagonal elements of Σ̂ represent correlations between features
- Multiplying by (1-γ) shrinks these correlations toward zero
- At γ=1: all correlations are zero → diagonal covariance matrix → feature independence within each class

**The limiting case:**
Combined with α=1 (class-specific matrices) and γ=1 (diagonal matrices):
Σ̂ₖ(α=1, γ=1) = diag(Σ̂ₖ) = class-specific diagonal covariance

This is exactly the **Naive Bayes** assumption: each class has its own variance per feature, but no inter-feature correlations. The entire GDA family tree is unified — through the two parameters (α, γ), any point between full QDA and Naive Bayes can be reached.

---

# SECTION 6: GDA vs. Logistic Regression

---

**Question 7**
LDA and Logistic Regression both produce linear boundaries but optimize fundamentally different objectives.

1. Explain "The Paradox" — LDA and Logistic Regression both produce wᵀx + b = 0 decision boundaries. How can two different algorithms produce the same type of boundary?
2. Explain precisely what each algorithm optimizes. What quantity does LDA maximize, and what quantity does Logistic Regression maximize? Why do these different objectives lead to different solutions even when the boundary form is the same?
3. Give a concrete scenario where LDA outperforms Logistic Regression, and a scenario where Logistic Regression dramatically outperforms LDA. Connect each to the underlying statistical assumptions.

### ✅ Answer

**Part 1:**
**The Paradox:**

Both LDA and Logistic Regression are linear classifiers — they separate classes using a hyperplane of the form wᵀx + b = 0. For binary classification:
- LDA: Assigns to class 1 if δ₁(x) > δ₂(x), which simplifies to wᵀx + b > 0 where w = Σ⁻¹(μ₁ - μ₂) and b involves the means and log prior ratio
- Logistic Regression: Assigns to class 1 if σ(wᵀx + b) > 0.5, i.e., wᵀx + b > 0

Both produce a linear boundary — how can they both be optimal for the same classification task?

The answer is that the SAME FORM of boundary (linear) can be derived by completely different optimization principles. The form of the decision surface is constrained by the linear classifier assumption, but the LOCATION and ORIENTATION of the hyperplane differ because they are found by optimizing different objectives. Two algorithms looking at the same data through different lenses will draw the line in different locations within the space of all linear boundaries.

**Part 2:**
**What each algorithm maximizes:**

**LDA (Generative):**
Maximizes the **joint likelihood** P(X, G) = P(X|G) × P(G).

Specifically, it maximizes the likelihood of ALL the data — both features and labels — under the Gaussian class-conditional model. It estimates:
1. The class priors P(G=k) from class frequencies
2. The class means μₖ = sample means within each class
3. The pooled covariance Σ from within-class scatter

This is a FULL GENERATIVE model — it models how the data was generated. The decision boundary is derived FROM the fitted distributions, not optimized directly.

**Logistic Regression (Discriminative):**
Maximizes the **conditional likelihood** P(G|X) — directly optimizes how well it predicts the label GIVEN the features.

It maximizes: Πₙ P(G=yₙ|X=xₙ) = Πₙ σ(wᵀxₙ)^yₙ (1-σ(wᵀxₙ))^(1-yₙ)

This ignores P(X) entirely — it asks only "given these features, what label should I predict?" The boundary is found by direct optimization of classification accuracy.

**Why different solutions:**
LDA puts weight on fitting the entire data distribution, including regions far from the decision boundary. An outlier in feature space (far from the boundary) still influences the estimated μₖ and Σ — changing the boundary even though the outlier is easily classified.

Logistic Regression only cares about points near the boundary (where the prediction uncertainty is high) — points far from the boundary with near-zero or near-one probabilities contribute almost nothing to the gradient. The boundary is optimized purely for classification performance.

**Part 3:**
**Concrete scenarios:**

**LDA outperforms Logistic Regression:**

Scenario: Medical diagnosis with 20 patients per class, 15 blood test measurements.

Why LDA wins:
- With only 20 examples per class and 15 features, Logistic Regression has insufficient data to reliably estimate w (15-dimensional vector) — high variance, unstable estimates
- LDA's Gaussian assumption acts as strong regularization — the distributional structure constrains the model to a small number of parameters (class means + pooled covariance), effectively borrowing strength across the entire feature distribution
- If blood measurements are genuinely approximately Gaussian (which many biological measurements are), LDA is the maximum likelihood estimator — it uses data more efficiently
- Logistic Regression might overfit the training boundary, while LDA finds a more stable, regularized estimate

**Logistic Regression dramatically outperforms LDA:**

Scenario: Income prediction (right-skewed, bimodal), age prediction (approximately Gaussian but mixed), employment status (binary feature included), with significant outliers from data entry errors.

Why Logistic Regression wins:
- LDA assumes all features are jointly Gaussian — income is heavily right-skewed (log-normal at best), violating this assumption
- Binary employment status (0/1) is clearly not Gaussian — including it forces LDA to fit a Gaussian to a discrete variable, producing meaningless estimates
- Data entry outliers (e.g., income = $5,000,000 when it should be $50,000) severely distort the estimated mean μₖ and especially the covariance Σ — LDA's boundary shifts toward the outlier
- Logistic Regression makes NO distributional assumption about X — it doesn't care if income is skewed, age is bimodal, or employment is binary. It simply fits the best linear boundary for the classification task using gradient descent on cross-entropy loss
- Logistic Regression's cross-entropy loss is less sensitive to outliers than LDA's Gaussian likelihood (where a single extreme point has enormous influence on the covariance estimate)

---

## BONUS CHALLENGE QUESTIONS

---

**Question 8**
Cross-topic synthesis.

1. The slides state: "Assuming Gaussian noise in statistics is mathematically identical to applying a squared-error penalty in geometry." Extend this argument: what loss function would correspond to assuming Laplace (double exponential) noise? What well-known regularization does this correspond to, and why?
2. Compare the complete GDA family (Naive Bayes, LDA, QDA, RDA) and Logistic Regression on: (a) number of parameters estimated, (b) what distribution is modeled, (c) decision boundary shape, (d) performance with N << p (few examples, many features), (e) performance with non-Gaussian features.
3. You are building a spam classifier with 10,000 training emails, 50,000 unique words as features, and you know most words appear in fewer than 10% of emails. Explain which model you would choose from the GDA family or Logistic Regression, justify every aspect of your choice, and describe the specific modifications needed.

### ✅ Answer

**Part 1:**
**Laplace noise → L1 loss connection:**

**The Gaussian connection (review):**
Gaussian noise ε ~ N(0, σ²): log-likelihood → -(1/2σ²) Σ(yₙ - ŷₙ)² → minimizing L2 loss (squared error) = minimizing negative log-likelihood.

**Laplace noise extension:**
Laplace distribution: p(ε) = (1/2b) exp(-|ε|/b)

If we assume yₙ = ŷₙ + εₙ where εₙ ~ Laplace(0, b):

p(yₙ|xₙ, w) = (1/2b) exp(-|yₙ - ŷₙ|/b)

Log-likelihood:
ℓ(w) = Σₙ [-log(2b) - |yₙ - ŷₙ|/b]

Maximizing ℓ(w) w.r.t. w:
= minimizing Σₙ |yₙ - ŷₙ|

This is the **L1 loss (Mean Absolute Error)**! Assuming Laplace noise and maximizing likelihood is mathematically identical to minimizing MAE.

**The regularization connection:**
In regression with L1 penalty on parameters (Lasso): minimize Σ(yₙ - wᵀxₙ)² + λΣ|wⱼ|

The penalty Σ|wⱼ| is exactly a Laplace prior on the parameters: p(wⱼ) ∝ exp(-|wⱼ|/b). Lasso is MAP (Maximum A Posteriori) estimation with a Laplace prior — equivalent to assuming the parameters themselves are drawn from a Laplace distribution centered at zero.

**The beautiful symmetry:**
- Gaussian likelihood → L2 loss; Gaussian prior → Ridge (L2) regularization
- Laplace likelihood → L1 loss; Laplace prior → Lasso (L1) regularization

Every common loss function and regularizer has a probabilistic interpretation as a noise or parameter distribution assumption. The geometric and statistical views are always two sides of the same coin.

**Part 2:**
**Complete comparison table:**

| Property | Naive Bayes | LDA | QDA | RDA | Logistic Regression |
|----------|-------------|-----|-----|-----|---------------------|
| Parameters estimated | μₖⱼ, σ²ₖⱼ (2Kp) | μₖ, one Σ (Kp + p(p+1)/2) | μₖ, Σₖ each (Kp + Kp(p+1)/2) | μₖ, blended Σₖ(α) + α hyperparameter | w (p+1 weights) |
| Distribution modeled | P(X,G): class-conditional diag. Gaussian | P(X,G): Gaussian, shared Σ | P(X,G): Gaussian, separate Σₖ | P(X,G): Gaussian, blended Σₖ(α) | P(G|X): directly |
| Decision boundary | Linear (from diagonal covariance) | Linear (shared quadratic cancels) | Quadratic (class-specific covariance) | Between linear and quadratic | Linear |
| N << p (few examples, many features) | Works well — 2Kp parameters | Works if N > p(p+1)/2 total | FAILS when Nₖ < p per class | Works for α < 1 (regularized toward LDA) | Needs regularization (L1/L2) |
| Non-Gaussian features | Poor (wrong model) | Poor (wrong model) | Poor (wrong model) | Poor (wrong model) | Excellent (no assumption) |
| Can handle p > N | Yes (diagonal Σ) | Sometimes (if total N > p²) | No (singular Σₖ) | Yes (for α close to 0) | Yes (with regularization) |

**Part 3:**
**Spam classifier recommendation:**

**Setup:** N=10,000 emails, p=50,000 unique words, sparse features (most words in <10% of emails).

**Recommendation: Multinomial Naive Bayes (generative) with Laplace Smoothing**

**Justification for Naive Bayes over other GDA methods:**

1. **p >> N problem:** p=50,000 features, N=10,000 examples. LDA would require estimating a 50,000×50,000 covariance matrix — that's 1.25 × 10⁹ parameters from 10,000 examples. Completely infeasible. QDA is even worse (10 classes would need 12.5 billion parameters). Naive Bayes requires only 2×K×p = 2×2×50,000 = 200,000 parameters — feasible!

2. **Feature independence is approximately valid:** While words are correlated in general text, for spam detection the specific co-occurrence patterns within the spam/not-spam classes are much weaker. "Free" and "prize" both independently indicate spam — their conditional dependence within the spam class is minimal for this specific task.

3. **Sparsity is natural:** The independence assumption is perfect for sparse bag-of-words features. Each word's probability given spam/not-spam can be estimated independently from its frequency counts.

**Justification over Logistic Regression:**

4. **Online learning:** Email arrives continuously. Naive Bayes counts can be updated incrementally (add new email counts to existing counts) — O(p) per email. Logistic Regression requires retraining with gradient descent — O(N×p) per update.

5. **Computational efficiency at test time:** Naive Bayes prediction: O(p) multiplications. Logistic Regression: O(p) dot product — similar, but Naive Bayes parallelizes more naturally.

6. **Works with small class sizes:** If some spam types have few examples (new spam patterns), Naive Bayes with Laplace smoothing handles zero-counts gracefully. Regularized logistic regression could also handle this but requires tuning.

**Specific modifications needed:**

1. **Laplace Smoothing (α=1):** With 50,000 words and 10,000 emails, many words (new or rare ones) won't appear in the training spam set. Without smoothing: P(word|spam) = 0 → entire product = 0. Add α=1 to all word counts: P_smoothed(word|spam) = (count(word in spam) + 1) / (total words in spam + |vocabulary|)

2. **Multinomial Naive Bayes (not Gaussian):** Words are counts, not continuous values — use Multinomial distribution, not Gaussian. P(word j appears k times|spam) = Multinomial probability, estimated as P(word j|spam) = frequency of word j in spam emails. This handles the discrete, count-based nature of text.

3. **Log-space computation:** With 50,000 features, the product Π P(wordⱼ|spam) underflows to zero (even with non-zero probabilities, 50,000 small numbers multiplied together = 0 in floating point). Always compute in log space: Σ log P(wordⱼ|spam) — sums, not products.

4. **TF-IDF preprocessing:** Consider downweighting common words (the, and, is) that appear equally in spam and not-spam — they add noise without contributing discrimination. This is implicitly handled if these words have near-equal P(word|spam) and P(word|not-spam), but explicit TF-IDF weighting can improve performance.

5. **Binary vs. Multinomial:** Consider Bernoulli Naive Bayes (did the word appear? yes/no) vs. Multinomial (how many times did it appear?). For spam detection, Bernoulli often performs similarly with much lower computational overhead — presence/absence of "Free" is more informative than its count.

---

## Complete Answer Reference Table

| Question | Core Concepts | Difficulty | Exam Priority |
|----------|--------------|------------|---------------|
| Q1 | Two philosophies, Gaussian-MSE equivalence, Vapnik's principle | Medium | Very High |
| Q2 | Generative vs. discriminative, Bayes classifier, tradeoffs | Medium | Very High |
| Q3 | Naive Bayes assumption, training, Laplace smoothing | Medium | Very High |
| Q4 | MVN distribution, GDA family tree, LDA linear derivation | Hard | Very High |
| Q5 | LDA discriminant function, pooled covariance, dimensionality reduction | Hard | High |
| Q6 | QDA breakdown, RDA shrinkage, Naive Bayes limiting case | Hard | Very High |
| Q7 | LDA vs Logistic Regression paradox, optimization differences | Hard | Very High |
| Q8 | Cross-topic synthesis | Very Hard | Medium |

---

## Top 7 Most Likely Exam Questions From This Topic

1. **Generative vs. Discriminative** — what each learns (P(X,G) vs P(G|X)), fundamental difference, when each wins (mirrors Q6 from original exam)
2. **Naive Bayes assumption** — what "naive" means, conditional independence, why it's called naive (mirrors Q6 part 2 from original exam)
3. **LDA vs. QDA** — equal vs. separate covariance matrices, why LDA is linear and QDA is quadratic
4. **Gaussian-MSE equivalence** — show that Gaussian noise assumption = squared error loss
5. **RDA** — shrinkage formula, α parameter, how it interpolates between LDA and QDA
6. **Laplace Smoothing** — why needed, what problem it solves, the formula
7. **LDA vs Logistic Regression** — same boundary form but different objectives, when each wins

